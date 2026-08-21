"""Geometry tests for the decode path, on synthesised DICOM.

The failures this guards against are all silent. An unrecovered side does not raise, it teaches five
of the twelve targets the opposite of the truth. A reversed slice stack does not raise, it makes
medial and lateral indistinguishable. A crop wider than the field of view does not raise, it leaves
that study at a different physical scale from every other. So each is pinned by a test on a slice
whose geometry is known exactly, rather than checked by eye on real data.

Standard orientations, in DICOM's LPS frame (+x left, +y posterior, +z superior):

    axial     rows -> posterior, columns -> left       normal +z
    coronal   rows -> inferior,  columns -> left       normal +y
    sagittal  rows -> inferior,  columns -> posterior  normal -x
"""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.dicom import (
    CROP_MM,
    SLOTS,
    canonical_flips,
    crop_to_extent,
    decode_series,
    laterality,
    order_slices,
    plane_of,
    read_geometry,
    resize,
    sample_indices,
    select_slots,
    to_uint8,
)

AXIAL = ([1, 0, 0], [0, 1, 0])
CORONAL = ([1, 0, 0], [0, 0, -1])
SAGITTAL = ([0, 1, 0], [0, 0, -1])


def write_slice(path, pixels, iop, ipp, spacing=(1.0, 1.0)):
    """Write a minimal but valid MR slice that pydicom can read back with pixels."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    pixels = np.ascontiguousarray(pixels).astype("<u2")
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = generate_uid()

    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "MR"
    ds.Rows, ds.Columns = pixels.shape
    ds.ImageOrientationPatient = [float(v) for v in (*iop[0], *iop[1])]
    ds.ImagePositionPatient = [float(v) for v in ipp]
    ds.PixelSpacing = [float(spacing[0]), float(spacing[1])]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


def make_series(tmp_path, orientation, centre_x, n=8, size=64, spacing=(1.0, 1.0), pattern=None):
    """A series of ``n`` parallel slices whose image centre sits at ``centre_x`` in patient x."""
    row_cos, col_cos = np.asarray(orientation[0], float), np.asarray(orientation[1], float)
    normal = np.cross(row_cos, col_cos)
    half = (size - 1) / 2
    # Solve for the corner that puts the image centre at the requested x.
    offset = half * spacing[1] * row_cos + half * spacing[0] * col_cos
    base = np.array([centre_x, 0.0, 0.0]) - offset

    paths = []
    for i in range(n):
        img = np.full((size, size), 100, dtype=np.uint16) if pattern is None else pattern(i, size)
        ipp = base + normal * (i * 3.0)
        paths.append(
            write_slice(tmp_path / f"s{i:03d}.dcm", img, orientation, ipp, spacing)
        )
    return paths


# --- geometry ----------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "orientation,expected",
    [(AXIAL, "Axial"), (CORONAL, "Coronal"), (SAGITTAL, "Sagittal")],
)
def test_plane_is_read_from_the_normal(tmp_path, orientation, expected):
    paths = make_series(tmp_path, orientation, centre_x=50.0, n=1)
    assert plane_of(read_geometry(paths[0])) == expected


def test_centre_is_not_the_corner(tmp_path):
    # 64 px at 2 mm is a 128 mm field of view, so the centre sits 63 mm from the corner. Using the
    # corner for the side call is what this guards against.
    paths = make_series(tmp_path, AXIAL, centre_x=40.0, n=1, size=64, spacing=(2.0, 2.0))
    geom = read_geometry(paths[0])
    assert geom.centre[0] == pytest.approx(40.0)
    assert geom.position[0] == pytest.approx(40.0 - 63.0)


@pytest.mark.parametrize(
    "centre_x,expected",
    [(80.0, "L"), (-80.0, "R"), (5.0, "unknown"), (-5.0, "unknown")],
)
def test_laterality_from_geometry(tmp_path, centre_x, expected):
    geoms = [read_geometry(p) for p in make_series(tmp_path, AXIAL, centre_x, n=3)]
    assert laterality(geoms) == expected


def test_laterality_of_nothing_is_unknown():
    assert laterality([]) == "unknown"


def test_slice_order_is_independent_of_acquisition_direction(tmp_path):
    paths = make_series(tmp_path, SAGITTAL, centre_x=80.0, n=6)
    geoms = [read_geometry(p) for p in paths]
    forward = order_slices(geoms, "Sagittal")
    reverse = order_slices(list(reversed(geoms)), "Sagittal")
    assert [g.path for g in forward] == [g.path for g in reverse]
    # ...and it runs along +x, so slice 0 is the medial-most slice of a left knee.
    xs = [float(g.position[0]) for g in forward]
    assert xs == sorted(xs)


@pytest.mark.parametrize(
    "orientation,plane", [(AXIAL, "Axial"), (CORONAL, "Coronal"), (SAGITTAL, "Sagittal")]
)
def test_standard_orientations_need_no_flip(tmp_path, orientation, plane):
    geom = read_geometry(make_series(tmp_path, orientation, 50.0, n=1)[0])
    assert canonical_flips(geom, plane) == (False, False)


def test_reversed_column_direction_is_detected(tmp_path):
    # A vendor writing columns toward the patient's right rather than left.
    geom = read_geometry(make_series(tmp_path, ([-1, 0, 0], [0, 1, 0]), -50.0, n=1)[0])
    flip_rows, flip_cols = canonical_flips(geom, "Axial")
    assert flip_cols and not flip_rows


def test_reversed_row_direction_is_detected(tmp_path):
    geom = read_geometry(make_series(tmp_path, ([1, 0, 0], [0, -1, 0]), 50.0, n=1)[0])
    flip_rows, flip_cols = canonical_flips(geom, "Axial")
    assert flip_rows and not flip_cols


# --- crop, resize, window ------------------------------------------------------------------------

def test_crop_takes_a_constant_physical_extent():
    # 0.15 mm/px over 960 px is 144 mm; 130 mm is 867 px.
    fine, _ = crop_to_extent(np.zeros((960, 960)), (0.15, 0.15))
    # 0.6 mm/px over 320 px is 192 mm; 130 mm is 217 px.
    coarse, _ = crop_to_extent(np.zeros((320, 320)), (0.6, 0.6))
    assert fine.shape == (867, 867)
    assert coarse.shape == (217, 217)
    # Different pixel counts, same physical extent — which is the entire point.
    assert fine.shape[0] * 0.15 == pytest.approx(CROP_MM, abs=0.5)
    assert coarse.shape[0] * 0.6 == pytest.approx(CROP_MM, abs=1.0)


def test_crop_reports_when_the_fov_is_too_small_to_crop():
    # 128 mm across, narrower than the 130 mm crop, so nothing is removed.
    out, did = crop_to_extent(np.zeros((64, 64)), (2.0, 2.0))
    assert out.shape == (64, 64)
    assert did is False


def test_crop_is_centred():
    img = np.zeros((100, 100))
    img[50, 50] = 1.0
    out, did = crop_to_extent(img, (2.0, 2.0), mm=100.0)
    assert did and out.shape == (50, 50)
    assert out[25, 25] == 1.0


def test_resize_area_averages_when_shrinking():
    # A single hot pixel must survive as mass, not vanish. Nearest-neighbour would drop it entirely
    # for most target grids, which is how a 1 mm tear disappears at an adequate pitch.
    img = np.zeros((960, 960), dtype=np.float32)
    img[480:484, 480:484] = 1000.0
    out = resize(img, 336)
    assert out.shape == (336, 336)
    assert out.sum() > 0


def test_window_uses_stack_percentiles_not_per_slice():
    # One bright slice and one dim one. Per-slice windowing would push both to full range and erase
    # the difference between them.
    stack = np.stack([np.full((8, 8), 10.0), np.full((8, 8), 200.0)])
    out = to_uint8(stack)
    assert out[0].max() < out[1].min()


def test_window_survives_a_constant_series():
    out = to_uint8(np.full((3, 8, 8), 42.0))
    assert out.shape == (3, 8, 8) and out.dtype == np.uint8


def test_window_survives_all_nan():
    out = to_uint8(np.full((2, 4, 4), np.nan))
    assert out.dtype == np.uint8 and int(out.max()) == 0


@pytest.mark.parametrize(
    "n,k,expected",
    [
        (16, 4, [0, 5, 10, 15]),
        (4, 4, [0, 1, 2, 3]),
        (2, 4, [0, 0, 1, 1]),   # fewer slices than wanted: each is repeated
        (1, 3, [0, 0, 0]),
        (0, 4, []),
    ],
)
def test_sample_indices_spans_the_whole_stack(n, k, expected):
    got = sample_indices(n, k)
    assert got == expected
    if n:
        assert got[0] == 0 and got[-1] == n - 1


# --- end to end --------------------------------------------------------------------------------

def _asymmetric(i, size):
    """A slice with all its signal on the low-column side, so a mirror is detectable."""
    img = np.zeros((size, size), dtype=np.uint16)
    img[:, : size // 4] = 900
    return img


def test_decode_series_shape_and_dtype(tmp_path):
    paths = make_series(tmp_path, AXIAL, 80.0, n=10, size=64, spacing=(2.0, 2.0))
    out = decode_series(paths, plane="Axial", n_slices=6, size=32)
    assert out is not None
    assert out.pixels.shape == (6, 32, 32)
    assert out.pixels.dtype == np.uint8
    assert out.side == "L"
    assert out.n_available == 10


def _two_knees(tmp_path, orientation, pattern, **kw):
    """The same pixel content acquired on a left knee and on a right knee."""
    (tmp_path / "L").mkdir()
    (tmp_path / "R").mkdir()
    left = make_series(tmp_path / "L", orientation, 80.0, pattern=pattern, **kw)
    right = make_series(tmp_path / "R", orientation, -80.0, pattern=pattern, **kw)
    return left, right


def test_axial_right_knee_is_mirrored_into_the_left_convention(tmp_path):
    left, right = _two_knees(tmp_path, AXIAL, _asymmetric, n=6, size=64)
    a = decode_series(left, plane="Axial", n_slices=4, size=32)
    b = decode_series(right, plane="Axial", n_slices=4, size=32)
    assert a.side == "L" and b.side == "R"

    # Identical pixels, opposite sides. Medial/lateral is an in-plane axis for axial, so exactly one
    # of the two must come back mirrored in the column axis — otherwise the four side-named targets
    # are learning from an axis the model cannot observe.
    assert not np.array_equal(a.pixels, b.pixels)
    assert np.array_equal(a.pixels, b.pixels[..., ::-1])


def test_sagittal_right_knee_reverses_slices_and_leaves_columns_alone(tmp_path):
    def ramp(i, size):
        img = np.zeros((size, size), dtype=np.uint16)
        img[:, : size // 4] = 40 + i * 30       # asymmetric, so a column flip would show
        return img

    left, right = _two_knees(tmp_path, SAGITTAL, ramp, n=6, size=32)
    a = decode_series(left, plane="Sagittal", n_slices=6, size=16)
    b = decode_series(right, plane="Sagittal", n_slices=6, size=16)

    # `order_slices` sorts by ascending patient x. The sagittal normal here runs along -x, so that
    # is the reverse of this synthetic acquisition order and the left knee's ramp reads descending.
    # What matters is that exactly one of the two stacks is additionally reversed, which is what puts
    # the medial-most slice at index 0 for both.
    a_profile = [int(s.mean()) for s in a.pixels]
    b_profile = [int(s.mean()) for s in b.pixels]
    assert a_profile == sorted(a_profile, reverse=True)
    assert b_profile == sorted(b_profile)
    assert a_profile == list(reversed(b_profile))

    # And the columns are untouched: mirroring a sagittal slice does nothing to medial/lateral, so
    # reversing the slice axis alone must recover one stack from the other exactly.
    assert np.array_equal(a.pixels, b.pixels[::-1])


def test_decode_series_returns_none_without_readable_geometry(tmp_path):
    junk = tmp_path / "junk.dcm"
    junk.write_bytes(b"not a dicom")
    assert decode_series([junk]) is None


def test_unknown_side_is_left_unnormalised(tmp_path):
    paths = make_series(tmp_path, AXIAL, 2.0, n=4, size=32, pattern=_asymmetric)
    out = decode_series(paths, plane="Axial", n_slices=4, size=16)
    assert out.side == "unknown"


# --- slot selection ----------------------------------------------------------------------------

def test_slots_are_the_six_plane_sequence_combinations():
    assert len(SLOTS) == 6
    assert set(SLOTS) == {(p, f) for p in ("Sagittal", "Coronal", "Axial") for f in (0, 1)}


def test_select_slots_prefers_the_series_with_most_slices():
    import pandas as pd

    series = pd.DataFrame(
        {
            "SeriesInstanceUID": ["a", "b", "c"],
            "Anatomical_Plane": ["Sagittal", "Sagittal", "Axial"],
            "Fluid_Sensitive": [1, 1, 0],
            "n_files": [12, 40, 20],
        }
    )
    chosen = select_slots(series)
    assert chosen[("Sagittal", 1)] == "b"
    assert chosen[("Axial", 0)] == "c"
    # Slots with no series are absent, so the caller masks them.
    assert ("Coronal", 0) not in chosen
