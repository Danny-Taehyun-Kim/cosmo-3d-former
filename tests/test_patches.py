"""Patch aggregation conserves area and produces well-formed features."""
import numpy as np

from cosmo3dformer.cosmo_io import cavity_area, read_cosmo
from cosmo3dformer.patches import ENCODER_INPUT_COLS, N_DIM, PAD, build_patches, pad_patches


def test_area_is_conserved(demo_cosmo):
    """Summed patch area equals summed segment area."""
    segments = read_cosmo(demo_cosmo)
    patches, _, _, _ = build_patches(segments)
    relative = abs(patches[:, 3].sum() - segments[:, 4].sum()) / segments[:, 4].sum()
    assert relative < 1e-5


def test_element_fractions_sum_to_one(demo_cosmo):
    """The eight element area fractions of a patch add up to one."""
    segments = read_cosmo(demo_cosmo)
    patches, _, _, _ = build_patches(segments)
    assert np.abs(patches[:, 10:18].sum(1) - 1).max() < 1e-4


def test_solid_angle_is_a_fraction(demo_cosmo):
    """The burial measure lies between zero and one."""
    segments = read_cosmo(demo_cosmo)
    patches, _, _, _ = build_patches(segments)
    assert patches[:, 9].min() >= 0.0
    assert patches[:, 9].max() <= 1.0


def test_padding_shape(demo_cosmo):
    """Padding gives a fixed-size array and a matching mask."""
    segments = read_cosmo(demo_cosmo)
    patches, n_patch, _, _ = build_patches(segments)
    padded, mask = pad_patches(patches)
    assert padded.shape == (PAD, N_DIM)
    assert int(mask.sum()) == min(n_patch, PAD)
    assert len(ENCODER_INPUT_COLS) == 12
