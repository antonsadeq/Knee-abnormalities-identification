"""Turning a study's DICOM into a fixed-shape uint8 tensor the encoder can read.

Four things in this data make the naive version wrong, and each is handled here rather than worked
around in the model:

1. **Pixel spacing varies across the 16 sites**, so resizing by pixel count hands the encoder
   anatomy at different physical scales. Everything is cropped to a constant *physical* extent
   first — see :func:`crop_to_extent`.
2. **Laterality is absent on roughly half the studies**, by whole vendors, and five of the twelve
   targets are named for a side. It is recovered from geometry instead — see :func:`laterality`.
3. **Slice and in-plane axis directions are vendor-dependent.** Two scanners imaging the same knee
   can hand you the stack reversed and the rows upside down. Both are normalised from the direction
   cosines, not assumed — see :func:`canonical_flips` and :func:`order_slices`.
4. **MRI has no absolute intensity scale**, unlike CT's Hounsfield units, so windowing is derived
   per series from percentiles over the sampled stack — see :func:`to_uint8`.

The output convention, fixed once here and depended on everywhere downstream:

    **medial at low column index, and for sagittal series, slice 0 is the medial-most slice.**

Reading is deliberately two-pass. Headers are parsed with ``stop_before_pixels`` to establish
geometry and choose which slices to keep, and only those slices are decoded. A series holds ~30
files and the model wants ~16 slices, so this halves the I/O — and I/O, not arithmetic, is what the
efficiency track is won on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Physical extent of the crop, in millimetres. Covers the joint and sits inside almost every
#: acquired field of view — the sampled Siemens axial series is 960 px at 0.1562 mm, i.e. 150 mm, so
#: 130 mm leaves a margin. **A crop larger than the FOV silently does nothing**, which is why
#: :func:`crop_to_extent` reports whether it actually cropped.
CROP_MM = 130.0

#: Output edge length. At a 130 mm crop this is 0.387 mm/px, under the 0.5 mm/px a 1 mm meniscal
#: tear needs to survive resampling; 224 px would give 0.580 and cap the meniscus classes before any
#: weight is trained. See ``docs/architecture.md``.
SIZE = 336

#: Slices kept per series. Linear in decode cost and sublinear in accuracy cost, so this is the
#: first dial to turn for the efficiency submission.
N_SLICES = 16

#: Below this many millimetres, an image centre is too close to the midline to call a side.
MIDLINE_MM = 15.0

PLANES = ("Sagittal", "Coronal", "Axial")

#: Patient axis each plane's slice normal runs along, in DICOM's LPS frame (+x left, +y posterior,
#: +z superior). Used to orient the normal consistently before sorting slices.
_NORMAL_AXIS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}

#: Canonical in-plane directions, as (axis, sign) that the column index and the row index should
#: increase along. Columns run toward the patient's left where medial/lateral is in-plane, so the
#: laterality flip has a well-defined meaning; rows run toward inferior so the images are upright.
_COL_DIR = {"Sagittal": (1, +1), "Coronal": (0, +1), "Axial": (0, +1)}
_ROW_DIR = {"Sagittal": (2, -1), "Coronal": (2, -1), "Axial": (1, +1)}


@dataclass(frozen=True)
class SliceGeometry:
    """Everything needed to place one slice in the patient frame, from headers alone."""

    path: Path
    rows: int
    cols: int
    row_cos: np.ndarray
    col_cos: np.ndarray
    spacing: tuple[float, float]
    position: np.ndarray

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.row_cos, self.col_cos)

    @property
    def centre(self) -> np.ndarray:
        """Centre of the image in patient coordinates.

        Not the same as ``ImagePositionPatient``, which is the centre of the *first voxel*. On the
        sampled series the two differ by 63 mm in x — enough that using the corner would weaken a
        laterality call that the centre makes decisively.
        """
        return (
            self.position
            + (self.cols - 1) / 2 * self.spacing[1] * self.row_cos
            + (self.rows - 1) / 2 * self.spacing[0] * self.col_cos
        )


def read_geometry(path: str | Path) -> SliceGeometry | None:
    """Parse one slice's geometry without decoding its pixels.

    Returns None when the tags that place the slice in space are missing, since a slice that cannot
    be positioned cannot be ordered or side-resolved and is better dropped than guessed at.
    """
    import pydicom

    path = Path(path)
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        spacing = tuple(float(v) for v in ds.PixelSpacing)
        rows, cols = int(ds.Rows), int(ds.Columns)
    except Exception:  # noqa: BLE001 — a malformed slice is skipped, never fatal
        return None
    if iop.shape != (6,) or ipp.shape != (3,) or len(spacing) != 2:
        return None

    return SliceGeometry(
        path=path,
        rows=rows,
        cols=cols,
        row_cos=iop[:3],
        col_cos=iop[3:],
        spacing=(spacing[0], spacing[1]),
        position=ipp,
    )


def plane_of(geom: SliceGeometry) -> str:
    """Which anatomical plane a slice lies in, from its normal's dominant axis.

    Computed rather than taken from ``train_series.csv`` so the two can be cross-checked; the CSV is
    authoritative for slot assignment, but a mismatch is a sign of a mislabelled series.
    """
    axis = int(np.argmax(np.abs(geom.normal)))
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[axis]


def order_slices(geoms: list[SliceGeometry], plane: str) -> list[SliceGeometry]:
    """Sort a series along its normal, oriented consistently in the patient frame.

    Sorting by ``InstanceNumber`` would inherit whatever direction the scanner happened to acquire
    in, so two sites imaging the same knee can hand you the stack reversed. Projecting onto a normal
    that is forced to point along +x / +y / +z removes the vendor from the answer.
    """
    if not geoms:
        return []
    axis = _NORMAL_AXIS[plane]
    reference = geoms[0].normal
    direction = reference if reference[axis] >= 0 else -reference
    return sorted(geoms, key=lambda g: float(np.dot(g.position, direction)))


def laterality(geoms: list[SliceGeometry]) -> str:
    """``"L"``, ``"R"`` or ``"unknown"`` from the sign of the mean image-centre x.

    The DICOM ``Laterality`` tag is Type 2C and missing on roughly half of these studies, by whole
    vendors, so it cannot be relied on — but the geometry always can. In DICOM's LPS frame +x is the
    patient's left, so a knee imaged left of the midline sits at positive x.

    Studies whose centre falls within :data:`MIDLINE_MM` of the midline are left ``"unknown"`` rather
    than guessed. A wrong side is worse than an unnormalised one: it does not merely fail to help the
    five side-named targets, it teaches them the opposite.
    """
    if not geoms:
        return "unknown"
    x = float(np.mean([g.centre[0] for g in geoms]))
    if abs(x) < MIDLINE_MM:
        return "unknown"
    return "L" if x > 0 else "R"


def canonical_flips(geom: SliceGeometry, plane: str) -> tuple[bool, bool]:
    """``(flip_rows, flip_cols)`` bringing a slice's in-plane axes to the canonical directions.

    Read off the direction cosines rather than assumed, because the sign of the row and column
    directions is a vendor and protocol choice. Without this, half the studies arrive mirrored and
    the encoder has to spend capacity undoing it.
    """
    col_axis, col_sign = _COL_DIR[plane]
    row_axis, row_sign = _ROW_DIR[plane]
    flip_cols = float(geom.row_cos[col_axis]) * col_sign < 0
    flip_rows = float(geom.col_cos[row_axis]) * row_sign < 0
    return flip_rows, flip_cols


def crop_to_extent(
    image: np.ndarray,
    spacing: tuple[float, float],
    mm: float = CROP_MM,
) -> tuple[np.ndarray, bool]:
    """Centre-crop to a constant physical extent. Returns ``(cropped, did_crop)``.

    This is the step that makes 16 sites comparable: pixel spacing runs from roughly 0.15 to 0.6 mm
    across this data, so a fixed-pixel crop would hand the encoder anatomy at a three-fold range of
    scales. ``did_crop`` is returned rather than logged because a crop wider than the field of view
    is a silent no-op, and a study that never got cropped is at the wrong scale for the whole run.
    """
    h, w = image.shape[-2:]
    want_r = int(round(mm / spacing[0]))
    want_c = int(round(mm / spacing[1]))
    if want_r >= h and want_c >= w:
        return image, False

    want_r, want_c = min(want_r, h), min(want_c, w)
    r0 = (h - want_r) // 2
    c0 = (w - want_c) // 2
    return image[..., r0 : r0 + want_r, c0 : c0 + want_c], True


def resize(image: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Resize a single 2-D slice, area-averaging when shrinking to avoid aliasing.

    Nearest or bilinear downsampling of a 960 px slice to 336 discards most pixels rather than
    averaging them, which is exactly how a 1 mm feature disappears despite the pitch being adequate.
    """
    import cv2

    interp = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(image.astype(np.float32), (size, size), interpolation=interp)


def to_uint8(stack: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Window a whole series to uint8 using percentiles over the **stack**, not per slice.

    MRI carries no absolute scale, so a window has to come from the data. Taking it per slice would
    normalise away the genuine intensity differences between slices — an effusion bright on one
    slice and absent on the next becomes equally bright on both — so the percentiles are computed
    once over every sampled slice together.
    """
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return np.zeros(stack.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(stack.shape, dtype=np.uint8)
    scaled = (stack - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def sample_indices(n: int, k: int) -> list[int]:
    """``k`` evenly spaced indices over ``n`` slices, repeating the ends when ``n < k``.

    Even spacing across the whole stack rather than the middle ``k``: the findings are spread through
    the joint, and a Baker's cyst or a collateral ligament sits at the periphery a centre crop of the
    slice axis would drop.
    """
    if n <= 0 or k <= 0:
        return []
    return [int(round(i)) for i in np.linspace(0, n - 1, k)]


def read_pixels(geom: SliceGeometry) -> np.ndarray | None:
    """Decode one slice to float32, applying rescale and photometric inversion."""
    import pydicom

    try:
        ds = pydicom.dcmread(str(geom.path), force=True)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:  # noqa: BLE001 — one unreadable slice must not lose the series
        return None

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept
    if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


@dataclass
class DecodedSeries:
    """A decoded series plus the facts needed to audit how it was built."""

    pixels: np.ndarray  # (n_slices, SIZE, SIZE) uint8
    plane: str
    side: str
    n_available: int
    cropped: bool
    spacing: tuple[float, float]

    @property
    def mm_per_px(self) -> float:
        return CROP_MM / self.pixels.shape[-1]


def decode_series(
    paths: list[str | Path],
    plane: str | None = None,
    n_slices: int = N_SLICES,
    size: int = SIZE,
    mm: float = CROP_MM,
    side: str | None = None,
) -> DecodedSeries | None:
    """Decode one series into ``(n_slices, size, size)`` uint8 in the canonical orientation.

    ``side`` overrides the per-series laterality call, and should be passed when a study-level value
    is available: a single series can sit near the midline and come back ``"unknown"`` while its
    siblings resolve confidently, and a study cannot be half-normalised.
    """
    geoms = [g for g in (read_geometry(p) for p in paths) if g is not None]
    if not geoms:
        return None

    plane = plane or plane_of(geoms[0])
    ordered = order_slices(geoms, plane)
    resolved = side or laterality(ordered)

    # Medial-to-lateral for a left knee runs along +x, so a right knee is the one that needs
    # reversing to reach the convention "slice 0 is medial".
    if plane == "Sagittal" and resolved == "R":
        ordered = list(reversed(ordered))

    keep = [ordered[i] for i in sample_indices(len(ordered), n_slices)]

    slices, cropped_any = [], False
    for geom in keep:
        arr = read_pixels(geom)
        if arr is None:
            slices.append(np.zeros((size, size), dtype=np.float32))
            continue

        flip_rows, flip_cols = canonical_flips(geom, plane)
        if flip_rows:
            arr = arr[::-1, :]
        if flip_cols:
            arr = arr[:, ::-1]
        # In-plane laterality applies only where medial/lateral is an in-plane axis. Mirroring a
        # sagittal slice does nothing to it, which is why that plane is handled by slice order.
        if plane in ("Coronal", "Axial") and resolved == "R":
            arr = arr[:, ::-1]

        arr, did = crop_to_extent(arr, geom.spacing, mm)
        cropped_any = cropped_any or did
        slices.append(resize(np.ascontiguousarray(arr), size))

    if not slices:
        return None

    return DecodedSeries(
        pixels=to_uint8(np.stack(slices)),
        plane=plane,
        side=resolved,
        n_available=len(ordered),
        cropped=cropped_any,
        spacing=keep[0].spacing,
    )


#: The six slots a study is reduced to: plane x fluid-sensitivity. Series metadata carries exactly
#: these two degrees of freedom (``Fat_Suppression`` is bit-identical to ``Fluid_Sensitive`` in all
#: 24,371 rows, so it adds nothing), and every study has all three planes.
SLOTS: tuple[tuple[str, int], ...] = tuple(
    (plane, flag) for plane in PLANES for flag in (0, 1)
)


def select_slots(series: "object") -> dict[tuple[str, int], str]:
    """Choose one series per (plane, fluid-sensitive) slot for one study.

    ``series`` is that study's rows from ``train_series.csv`` / ``test_series.csv``. Studies carry
    3–14 series against 6 slots, so most slots have a choice to make; the series with the most slices
    wins, since slice count is the axis that costs cache linearly rather than quadratically and more
    coverage is the cheaper kind of information. Ties break on the series id so a rebuild is
    reproducible.

    Slots with no series are simply absent from the result — the caller masks them rather than
    substituting a zero volume, because feeding a zero vector into the slot attention asks the model
    to distinguish "black" from "not acquired".
    """
    from .config import SEQUENCE_FLAG, SERIES_ID

    chosen: dict[tuple[str, int], str] = {}
    for (plane, flag), group in series.groupby(["Anatomical_Plane", SEQUENCE_FLAG]):
        key = (str(plane), int(flag))
        if key not in {(p, f) for p, f in SLOTS}:
            continue
        counts = group.get("n_files")
        if counts is not None:
            group = group.sort_values(["n_files", SERIES_ID], ascending=[False, True])
        else:
            group = group.sort_values(SERIES_ID)
        chosen[key] = str(group.iloc[0][SERIES_ID])
    return chosen


def decode_study(
    root: str | Path,
    series: "object",
    n_slices: int = N_SLICES,
    size: int = SIZE,
    mm: float = CROP_MM,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode one study into ``(pixels, present, info)``.

    ``pixels`` is ``(6, n_slices, size, size)`` uint8 in :data:`SLOTS` order and ``present`` is a
    ``(6,)`` bool mask. ``root`` is the study directory holding one subdirectory per series.

    **Laterality is resolved once for the study, not per series.** A single series can sit close
    enough to the midline to come back ``"unknown"`` while its siblings resolve confidently, and
    normalising some planes but not others would be worse than normalising none: the medial/lateral
    targets would then disagree with each other across slots of the same knee.
    """
    root = Path(root)
    slots = select_slots(series)

    paths: dict[tuple[str, int], list[Path]] = {}
    for key, series_id in slots.items():
        found = sorted((root / series_id).glob("*.dcm"))
        if found:
            paths[key] = found

    # One study-level side, taken from whichever slot resolves it, so every slot is normalised the
    # same way. Sagittal is preferred: its normal runs along the medial-lateral axis, so its centre
    # x is the least ambiguous of the three.
    side = "unknown"
    for key in sorted(paths, key=lambda k: (k[0] != "Sagittal",)):
        geoms = [g for g in (read_geometry(p) for p in paths[key][:4]) if g is not None]
        found = laterality(geoms)
        if found != "unknown":
            side = found
            break

    pixels = np.zeros((len(SLOTS), n_slices, size, size), dtype=np.uint8)
    present = np.zeros(len(SLOTS), dtype=bool)
    info: dict = {"side": side, "slots": {}}

    for i, key in enumerate(SLOTS):
        if key not in paths:
            continue
        decoded = decode_series(
            paths[key], plane=key[0], n_slices=n_slices, size=size, mm=mm, side=side
        )
        if decoded is None:
            continue
        pixels[i] = decoded.pixels
        present[i] = True
        info["slots"][f"{key[0]}_{key[1]}"] = {
            "n_available": decoded.n_available,
            "cropped": decoded.cropped,
            "mm_per_px": round(decoded.mm_per_px, 4),
            "spacing": [round(s, 4) for s in decoded.spacing],
        }

    return pixels, present, info
