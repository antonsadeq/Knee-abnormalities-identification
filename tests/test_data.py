"""The mirror augmentation is the one that can silently destroy a third of the metric.

Flipping images without swapping the medial/lateral label pairs corrupts four of twelve targets;
swapping the labels without mirroring every plane *coherently* corrupts them differently. Neither
raises. So both halves are pinned here, per plane.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.config import LABELS, LATERALITY_PAIRS, STUDY_ID
from rsna_knee.data import (
    IN_PLANE_SLOTS,
    THROUGH_PLANE_SLOTS,
    KneeStudies,
    build_index,
    intensity_jitter,
    label_swap_index,
    mirror_study,
    sample_slice_window,
)
from rsna_knee.dicom import SLOTS


def _study(n_slices=4, size=8):
    """A study whose every slot is distinguishable and asymmetric."""
    pixels = np.zeros((len(SLOTS), n_slices, size, size), dtype=np.uint8)
    for s in range(len(SLOTS)):
        for k in range(n_slices):
            pixels[s, k, :, : size // 2] = 10 * s + k + 1   # left half only, varies by slot & slice
    return pixels


# --- which slots hold laterality where ----------------------------------------------------------

def test_slot_partition_matches_the_planes():
    assert [SLOTS[i][0] for i in IN_PLANE_SLOTS] == ["Coronal", "Coronal", "Axial", "Axial"]
    assert [SLOTS[i][0] for i in THROUGH_PLANE_SLOTS] == ["Sagittal", "Sagittal"]
    assert set(IN_PLANE_SLOTS) | set(THROUGH_PLANE_SLOTS) == set(range(len(SLOTS)))


# --- the label permutation ----------------------------------------------------------------------

def test_label_swap_exchanges_exactly_the_pairs():
    order = label_swap_index()
    position = {name: i for i, name in enumerate(LABELS)}
    for a, b in LATERALITY_PAIRS:
        assert order[position[a]] == position[b]
        assert order[position[b]] == position[a]
    # Everything else is left alone.
    moved = {LABELS[i] for i in range(len(LABELS)) if order[i] != i}
    assert moved == {name for pair in LATERALITY_PAIRS for name in pair}


def test_label_swap_is_an_involution():
    order = label_swap_index()
    assert list(order[order]) == list(range(len(LABELS)))


# --- mirroring ----------------------------------------------------------------------------------

def test_mirror_flips_columns_only_for_in_plane_slots():
    pixels = _study()
    out, _, _ = mirror_study(pixels)
    for i in IN_PLANE_SLOTS:
        assert np.array_equal(out[i], pixels[i][..., ::-1]), SLOTS[i]


def test_mirror_reverses_slices_only_for_sagittal():
    pixels = _study()
    out, _, _ = mirror_study(pixels)
    for i in THROUGH_PLANE_SLOTS:
        # Slice order reversed, and the image itself untouched — mirroring a sagittal slice does
        # nothing to medial/lateral, so flipping its columns would be a pointless corruption.
        assert np.array_equal(out[i], pixels[i][::-1]), SLOTS[i]
        assert np.array_equal(out[i][0], pixels[i][-1]), SLOTS[i]


def test_mirror_swaps_the_labels_with_the_pixels():
    pixels = _study()
    targets = np.arange(len(LABELS), dtype=float)
    weights = np.arange(len(LABELS), dtype=float) * 0.1
    _, t, w = mirror_study(pixels, targets, weights)

    position = {name: i for i, name in enumerate(LABELS)}
    for a, b in LATERALITY_PAIRS:
        assert t[position[a]] == targets[position[b]]
        assert w[position[a]] == weights[position[b]]


def test_mirroring_twice_is_the_identity():
    pixels = _study()
    targets = np.random.default_rng(0).random(len(LABELS))
    once = mirror_study(pixels, targets)
    twice = mirror_study(once[0], once[1])
    assert np.array_equal(twice[0], pixels)
    assert np.allclose(twice[1], targets)


def test_mirror_does_not_mutate_its_input():
    pixels = _study()
    original = pixels.copy()
    mirror_study(pixels)
    assert np.array_equal(pixels, original)


# --- slice sampling and jitter ------------------------------------------------------------------

@pytest.mark.parametrize("available,take", [(16, 8), (30, 16), (8, 8), (4, 16)])
def test_slice_window_stays_in_range_and_right_length(available, take):
    rng = np.random.default_rng(0)
    idx = sample_slice_window(available, take, rng)
    assert idx.min() >= 0 and idx.max() < available
    assert len(idx) == (take if take < available else available)


def test_slice_window_spreads_over_the_whole_stack():
    # A contiguous window would bias toward one part of the joint; findings are spread through it.
    rng = np.random.default_rng(1)
    idx = sample_slice_window(32, 8, rng)
    assert idx.min() < 8 and idx.max() > 23


def test_intensity_jitter_stays_in_unit_range_and_varies_by_slot():
    pixels = _study()
    out = intensity_jitter(pixels, np.random.default_rng(0))
    assert out.dtype == np.float32 and out.min() >= 0.0 and out.max() <= 1.0
    means = [float(out[i].mean()) for i in range(len(SLOTS))]
    assert len(set(np.round(means, 6))) > 1


# --- index building -----------------------------------------------------------------------------

def test_build_index_rejects_a_study_present_in_two_shards(tmp_path):
    a, b = tmp_path / "shard0", tmp_path / "shard1"
    a.mkdir(); b.mkdir()
    np.save(a / "dup.npy", np.zeros((1,), dtype=np.uint8))
    np.save(b / "dup.npy", np.zeros((1,), dtype=np.uint8))
    with pytest.raises(ValueError, match="more than one shard"):
        build_index([a, b])


def test_build_index_errors_on_an_empty_cache(tmp_path):
    with pytest.raises(ValueError, match="no .npy"):
        build_index([tmp_path])


# --- dataset ------------------------------------------------------------------------------------

def _cache(tmp_path, ids):
    d = tmp_path / "shard"
    d.mkdir()
    for sid in ids:
        np.save(d / f"{sid}.npy", _study(n_slices=6, size=8))
    return d


def test_dataset_shapes_and_ranges(tmp_path):
    ids = ["s1", "s2"]
    index = build_index([_cache(tmp_path, ids)])
    y = pd.DataFrame(0.4, index=ids, columns=LABELS)
    w = pd.DataFrame(1.0, index=ids, columns=LABELS)

    ds = KneeStudies(index, y, w, n_slices=3, train=False)
    pixels, targets, weights, present = ds[0]
    assert pixels.shape == (6, 3, 8, 8) and pixels.dtype == np.float32
    assert 0.0 <= pixels.min() and pixels.max() <= 1.0
    assert targets.shape == (len(LABELS),) and weights.shape == (len(LABELS),)
    assert present.shape == (6,) and present.all()


def test_absent_slots_are_masked_from_the_manifest(tmp_path):
    ids = ["s1"]
    d = tmp_path / "shard"; d.mkdir()
    pixels = _study(n_slices=4, size=8)
    pixels[2] = 0            # Coronal_0 not acquired
    np.save(d / "s1.npy", pixels)
    index = build_index([d])
    y = pd.DataFrame(0.4, index=ids, columns=LABELS)
    w = pd.DataFrame(1.0, index=ids, columns=LABELS)

    ds = KneeStudies(index, y, w, present=pd.Series(["110111"], index=ids), n_slices=2, train=False)
    _, _, _, present = ds[0]
    assert list(present) == [True, True, False, True, True, True]


def test_eval_mode_is_deterministic_and_unaugmented(tmp_path):
    ids = ["s1"]
    index = build_index([_cache(tmp_path, ids)])
    y = pd.DataFrame(0.4, index=ids, columns=LABELS)
    w = pd.DataFrame(1.0, index=ids, columns=LABELS)
    ds = KneeStudies(index, y, w, n_slices=3, train=False)
    a, ta, _, _ = ds[0]
    b, tb, _, _ = ds[0]
    assert np.array_equal(a, b) and np.array_equal(ta, tb)


def test_train_mode_mirrors_some_items(tmp_path):
    ids = [f"s{i}" for i in range(24)]
    index = build_index([_cache(tmp_path, ids)])
    y = pd.DataFrame(0.4, index=ids, columns=LABELS)
    w = pd.DataFrame(1.0, index=ids, columns=LABELS)
    ds = KneeStudies(index, y, w, n_slices=4, train=True, mirror_prob=1.0, seed=3)
    reference = np.load(index.iloc[0]["path"])
    pixels, _, _, _ = ds[0]
    # With mirror_prob=1 the coronal slots must be column-flipped relative to the cache.
    i = IN_PLANE_SLOTS[0]
    assert not np.allclose(pixels[i], reference[i, : pixels.shape[1]] / 255.0)
