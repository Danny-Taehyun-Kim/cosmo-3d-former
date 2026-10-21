"""Tests for defects found during release preparation.

Each test pins down one behaviour that was wrong at some point and would be
easy to reintroduce.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmo3dformer.model import PatchEncoder, count_params
from cosmo3dformer.reconstruction import INTERPOLATION_METHODS, METHODS, evaluate_round
from cosmo3dformer.splits import CLEAN2_PROPERTIES, val_split

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_bias_scale_changes_the_output_and_not_the_state_dict():
    """A zero distance-bias scale must change the encoder output.

    The ablation encoder keeps non-zero bias_mlp weights because no gradient ever
    reached them, so loading it without the scale applies a bias that the trained
    model never saw. The scale must not enter the state dict, otherwise strict
    loading of the released weights breaks.
    """
    torch.manual_seed(0)
    features = torch.randn(1, 8, 12)
    xyz = torch.randn(1, 8, 3)
    pad = torch.ones(1, 8, dtype=torch.bool)
    dist = torch.cdist(xyz, xyz)

    on = PatchEncoder(n_in=12, d=64, L=2, h=4, p=0.0, encoder_bias_scale=1.0).eval()
    off = PatchEncoder(n_in=12, d=64, L=2, h=4, p=0.0, encoder_bias_scale=0.0).eval()
    off.load_state_dict(on.state_dict(), strict=True)
    with torch.no_grad():
        a = on(features, dist, pad)
        b = off(features, dist, pad)
    assert not torch.allclose(a, b), "the scale had no effect"
    assert set(on.state_dict()) == set(off.state_dict())
    assert count_params(on) == count_params(off)


def test_no_bias_config_carries_a_zero_scale():
    """The ablation configuration must request a zero scale, not a boolean flag."""
    import yaml
    config = yaml.safe_load(open(CONFIGS / "pretrain_M_no_bias.yaml"))
    assert config["encoder_bias_scale"] == 0.0
    baseline = yaml.safe_load(open(CONFIGS / "pretrain_M.yaml"))
    assert baseline["encoder_bias_scale"] == 1.0
    for key in ("d_model", "n_heads", "lr", "epochs", "batch_size", "seed"):
        assert config[key] == baseline[key], f"{key} should match the baseline"


def test_validation_split_depends_only_on_the_pool_it_is_given():
    """Filtering the pool first produces a different validation set.

    The published split is drawn from the full labelled pool and the consistency
    filter is applied afterwards. Drawing it from the filtered pool gives a
    different set, so the order matters and is pinned here.
    """
    rng = np.random.default_rng(0)
    cids = np.arange(4000)
    keep = rng.random(4000) > 0.3
    full = val_split(cids, 0)
    filtered = val_split(cids[keep], 0)
    assert full != filtered
    assert len(full) > len(filtered)
    assert CLEAN2_PROPERTIES == ("Pc_kPa", "Vc_L_per_mol")


def test_target_statistics_file_matches_the_channel_order():
    """The stored pre-training statistics must cover four channels in order."""
    stats = json.loads((CONFIGS / "pretrain_target_stats.json").read_text())
    assert stats["channels"] == ["sigma", "area", "potential", "burial"]
    assert len(stats["mean"]) == len(stats["std"]) == 4
    assert all(s > 0 for s in stats["std"])
    assert abs(stats["mean"][1] - 2.1) < 0.01, "the area mean should sit at the target area"


def test_interpolation_baselines_leave_burial_absent():
    """Burial is not an encoder input, so the interpolation baselines must not
    produce a number for it. A zero would read as the training-mean prediction."""
    from cosmo3dformer.reconstruction import baseline_predictions
    torch.manual_seed(0)
    visible = torch.randn(1, 6, 4)
    pv = torch.ones(1, 6, dtype=torch.bool)
    dmv = torch.rand(1, 9, 6) * 4 + 0.5
    out = baseline_predictions(visible, pv, dmv, 9)
    for name in INTERPOLATION_METHODS:
        assert torch.isnan(out[name][..., 3]).all(), f"{name} produced a burial value"
        assert torch.isfinite(out[name][..., :3]).all()
    assert set(INTERPOLATION_METHODS) < set(METHODS)


def test_reconstruction_accumulates_into_the_real_tensor():
    """Coverage must be non-zero after a round.

    Indexing the accumulator with a tensor returns a copy, so an in-place scatter
    would be discarded and every count would stay zero.
    """
    from cosmo3dformer.model import PatchMAE
    torch.manual_seed(0)
    model = PatchMAE(n_in=12, d=32, L=1, h=2, p=0.0, dd=16, dL=1, dh_=2).eval()
    n, p = 3, 20
    features = torch.randn(n, p, 12)
    xyz = torch.randn(n, p, 3) * 2
    pad = torch.ones(n, p, dtype=torch.bool)
    targets = torch.randn(n, p, 4)
    out = evaluate_round(model, features, xyz, pad, targets, [1, 2, 3], 0, 0.9,
                         "cpu", batch_size=2, autocast=False)
    assert int(out["coverage"].sum()) > 0
    assert int(out["hidden_first"].sum()) > 0
    assert torch.isfinite(out["mse"][:, METHODS.index("model"), :3]).all()
