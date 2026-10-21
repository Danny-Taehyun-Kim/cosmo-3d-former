"""The .cosmo reader agrees with the header values of the file."""
import numpy as np

from cosmo3dformer.cosmo_io import SEGMENT_COLUMNS, cavity_area, read_cosmo


def test_segment_columns(demo_cosmo):
    """Seven columns are returned in the documented order."""
    segments = read_cosmo(demo_cosmo)
    assert segments.shape[1] == len(SEGMENT_COLUMNS) == 7


def test_area_matches_header(demo_cosmo):
    """Summed segment area matches the cavity area in the header."""
    text = open(demo_cosmo).read()
    segments = read_cosmo(demo_cosmo)
    assert abs(segments[:, 4].sum() - cavity_area(text)) / cavity_area(text) < 0.01


def test_surface_is_centred(demo_cosmo):
    """The area-weighted centroid sits at the origin after reading."""
    segments = read_cosmo(demo_cosmo)
    area = segments[:, 4]
    centroid = (segments[:, :3] * area[:, None]).sum(0) / area.sum()
    assert np.abs(centroid).max() < 1e-4


def test_atomic_numbers_are_known(demo_cosmo):
    """Every segment carries a recognised atomic number."""
    segments = read_cosmo(demo_cosmo)
    assert (segments[:, 6] > 0).all()
