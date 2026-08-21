"""Reading the decoded cache into training batches.

The cache holds one ``(6, N_SLICES, SIZE, SIZE)`` uint8 array per study, in :data:`dicom.SLOTS`
order, with absent slots zeroed and recorded in the manifest. This module turns that into batches and
does the augmentation — where one operation is dangerous enough to be the reason this file exists.

**Mirroring a knee is not a horizontal flip.** Four of the twelve targets are medial/lateral pairs and
MCL is named for its side, so mirroring has to be applied coherently across planes *and* to the
labels:

* coronal and axial hold medial/lateral as an **in-plane** axis, so they flip along columns;
* sagittal holds it as the **slice** axis — mirroring a sagittal image does nothing to it, and the
  stack must be reversed instead;
* and the ``Medial/Lateral`` label pairs must swap with the pixels.

Doing any one of those and not the others silently corrupts a third of the metric, which is why
:func:`mirror_study` does all three together and there is no public "flip the images" function.

Everything here is plain numpy so it can be tested without a GPU; :class:`KneeStudies` is a thin
torch wrapper over it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS, LATERALITY_PAIRS, STUDY_ID
from .dicom import SLOTS

#: Indices into :data:`dicom.SLOTS` whose medial/lateral axis lies in the image plane.
IN_PLANE_SLOTS = tuple(i for i, (plane, _) in enumerate(SLOTS) if plane in ("Coronal", "Axial"))

#: Indices whose medial/lateral axis is the slice axis.
THROUGH_PLANE_SLOTS = tuple(i for i, (plane, _) in enumerate(SLOTS) if plane == "Sagittal")


def label_swap_index(labels: list[str] | None = None) -> np.ndarray:
    """Permutation that exchanges each medial/lateral pair, for use with :func:`mirror_study`."""
    labels = labels or LABELS
    order = list(range(len(labels)))
    position = {name: i for i, name in enumerate(labels)}
    for a, b in LATERALITY_PAIRS:
        if a in position and b in position:
            order[position[a]], order[position[b]] = position[b], position[a]
    return np.asarray(order)


def mirror_study(
    pixels: np.ndarray,
    targets: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Mirror a study to the other knee, pixels and labels together.

    ``pixels`` is ``(6, n_slices, h, w)``. Returns mirrored copies; inputs are untouched.

    The asymmetry between planes is the whole point. Flipping every slot along its columns would
    leave the sagittal slots unchanged in the only respect that matters, so the model would receive a
    study whose coronal and axial views say "left knee" while its sagittal views still say "right" —
    and the swapped labels would then be wrong for half the slots.
    """
    out = pixels.copy()
    for i in IN_PLANE_SLOTS:
        out[i] = out[i][..., ::-1]
    for i in THROUGH_PLANE_SLOTS:
        out[i] = out[i][::-1]

    swap = label_swap_index()
    return (
        out,
        None if targets is None else np.asarray(targets)[swap],
        None if weights is None else np.asarray(weights)[swap],
    )


def sample_slice_window(n_available: int, n_take: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for a random contiguous-ish subset of slices, evenly spread.

    A random *contiguous* window would bias toward one part of the joint; a fully random subset
    destroys the slice ordering the model can otherwise use. This jitters an even spread instead:
    the same coverage as inference, with per-epoch variation.
    """
    if n_take >= n_available:
        return np.arange(n_available)
    base = np.linspace(0, n_available - 1, n_take)
    jitter = rng.uniform(-0.5, 0.5, size=n_take) * (n_available - 1) / max(n_take - 1, 1)
    return np.clip(np.round(base + jitter), 0, n_available - 1).astype(int)


def intensity_jitter(
    pixels: np.ndarray, rng: np.random.Generator, gain: float = 0.15, bias: float = 0.08
) -> np.ndarray:
    """Per-slot brightness/contrast jitter, applied in float on a 0-1 scale.

    Per *slot* rather than per study or per slice: the slots come from different sequences whose
    relative brightness is real information, but each one's absolute level is an artefact of the
    percentile window it was normalised with, so that is the level worth perturbing.
    """
    out = pixels.astype(np.float32) / 255.0
    for i in range(out.shape[0]):
        g = 1.0 + rng.uniform(-gain, gain)
        b = rng.uniform(-bias, bias)
        out[i] = np.clip(out[i] * g + b, 0.0, 1.0)
    return out


def build_index(cache_dirs: list[str | Path]) -> pd.DataFrame:
    """One row per cached study: id, path, shard. Duplicates across shards are an error.

    The shards are meant to be disjoint, so a study appearing twice means a rebuild wrote into a
    stale directory — worth failing on rather than silently preferring one copy.
    """
    rows = []
    for directory in cache_dirs:
        directory = Path(directory)
        for path in sorted(directory.glob("*.npy")):
            rows.append({STUDY_ID: path.stem, "path": str(path), "shard": directory.name})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no .npy found under {[str(d) for d in cache_dirs]}")
    duplicated = frame[frame[STUDY_ID].duplicated(keep=False)]
    if len(duplicated):
        raise ValueError(f"{duplicated[STUDY_ID].nunique()} studies appear in more than one shard")
    return frame


class KneeStudies:
    """Map-style dataset over the decoded cache.

    Yields ``(pixels, targets, weights, present)`` with pixels ``(6, n_slices, size, size)`` float32
    in 0-1. Arrays are opened with ``mmap_mode`` so a random slice subset reads only the slices it
    takes — the cache is 10.8 MB per study and an epoch touches all of it, so this is the difference
    between streaming from page cache and copying 48 GB per epoch.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        targets: pd.DataFrame,
        weights: pd.DataFrame,
        present: pd.Series | None = None,
        n_slices: int = 8,
        train: bool = True,
        mirror_prob: float = 0.5,
        seed: int = 0,
        labels: list[str] | None = None,
    ) -> None:
        self.labels = labels or LABELS
        self.index = index.reset_index(drop=True)
        self.targets = targets.reindex(columns=self.labels)
        self.weights = weights.reindex(columns=self.labels)
        self.present = present
        self.n_slices = n_slices
        self.train = train
        self.mirror_prob = mirror_prob if train else 0.0
        self.seed = seed

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        study = row[STUDY_ID]
        # Seeded per (epoch-less) item so a worker's augmentation is reproducible from the index.
        rng = np.random.default_rng((self.seed, i))

        array = np.load(row["path"], mmap_mode="r")
        n_available = array.shape[1]
        take = (
            sample_slice_window(n_available, self.n_slices, rng)
            if self.train
            else np.linspace(0, n_available - 1, min(self.n_slices, n_available)).round().astype(int)
        )
        pixels = np.asarray(array[:, take])

        targets = self.targets.loc[study].to_numpy(dtype=np.float32)
        weights = self.weights.loc[study].to_numpy(dtype=np.float32)

        if self.present is not None and study in self.present.index:
            mask = np.array([c == "1" for c in str(self.present.loc[study])], dtype=bool)
        else:
            # Fall back to "a slot with no signal was not acquired". Correct for this cache, since
            # absent slots are written as exact zeros.
            mask = pixels.reshape(pixels.shape[0], -1).any(axis=1)

        if self.train and rng.random() < self.mirror_prob:
            pixels, targets, weights = mirror_study(pixels, targets, weights)
            mask = mask.copy()  # slot identity is unchanged by mirroring; planes map onto themselves

        pixels = (
            intensity_jitter(pixels, rng)
            if self.train
            else pixels.astype(np.float32) / 255.0
        )
        return pixels, targets, weights, mask
