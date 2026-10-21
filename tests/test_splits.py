"""Target transforms and the validation split behave as documented."""
import numpy as np

from cosmo3dformer.splits import TRANSFORM, Target, subset_val_indices, val_split


def test_transform_round_trip():
    """Forward followed by inverse returns the original values."""
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 900, 500)
    for mode in ("standard", "log1p", "log"):
        target = Target(mode).fit(y)
        assert np.allclose(target.inverse(target.forward(y)), y, rtol=1e-9)


def test_transform_map_covers_every_property():
    """Each benchmark property has a registered transform."""
    assert len(TRANSFORM) == 9
    assert set(TRANSFORM.values()) <= {"standard", "log1p", "log"}


def test_val_split_is_deterministic_and_disjoint():
    """The validation subset depends only on the pool and the fold index."""
    pool = list(range(1000))
    first = val_split(pool, 0)
    assert first == val_split(list(reversed(pool)), 0)
    assert first != val_split(pool, 1)
    assert len(first) == 125


def test_subset_validation_size():
    """A subset reserves 15 percent for validation, with a floor of eight."""
    assert len(subset_val_indices(50)) == 8
    assert len(subset_val_indices(400)) == 60
    assert len(subset_val_indices(1600)) == 240
