"""Block masking keeps the requested ratio, the floor and determinism."""
import torch

from cosmo3dformer.masking import block_mask, gather_masked_batch


def _surface(n=60, seed=0):
    torch.manual_seed(seed)
    xyz = torch.randn(1, n, 3) * 3
    pad = torch.ones(1, n, dtype=torch.bool)
    return torch.cdist(xyz, xyz), pad, xyz


def test_ratio_is_respected():
    """The hidden count is the ratio times the valid count, rounded."""
    dist, pad, _ = _surface()
    generator = torch.Generator().manual_seed(1)
    hidden, visible = block_mask(dist, pad, 0.9, generator=generator)
    assert int(hidden.sum()) == round(0.9 * int(pad.sum()))
    assert int((hidden & visible).sum()) == 0
    assert int((hidden | visible).sum()) == int(pad.sum())


def test_minimum_three_visible():
    """A surface is never left with fewer than three visible patches."""
    dist, pad, _ = _surface(n=12)
    generator = torch.Generator().manual_seed(2)
    hidden, visible = block_mask(dist, pad, 0.99, generator=generator)
    assert int(visible.sum()) >= 3


def test_same_rank_gives_same_mask():
    """Passing the ranks in reproduces the mask exactly."""
    dist, pad, _ = _surface()
    rank = torch.rand(1, dist.shape[1], generator=torch.Generator().manual_seed(3))
    a = block_mask(dist, pad, 0.9, rank=rank)
    b = block_mask(dist, pad, 0.9, rank=rank.clone())
    assert torch.equal(a[0], b[0])


def test_gather_round_trip():
    """Gathered visible and masked blocks cover the valid patches exactly once."""
    dist, pad, xyz = _surface(n=40)
    rank = torch.rand(1, 40, generator=torch.Generator().manual_seed(4))
    hidden, visible = block_mask(dist, pad, 0.9, rank=rank)
    features = torch.randn(1, 40, 12)
    fv, dvv, pv, dmv, pm, iv, im = gather_masked_batch(features, xyz, pad, hidden, visible)
    assert int(pv.sum()) == int(visible.sum())
    assert int(pm.sum()) == int(hidden.sum())
    assert dmv.shape == (1, im.shape[1], iv.shape[1])
