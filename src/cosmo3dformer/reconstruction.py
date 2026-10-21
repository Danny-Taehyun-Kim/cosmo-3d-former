"""Population-level evaluation of masked surface reconstruction.

Each surface is masked repeatedly with molecule- and round-specific seeds. A
patch is scored only in the rounds in which it was hidden, so the ensemble
average never mixes in a round where the patch was visible. Patches that were
never hidden stay unpredicted and are not filled with a reference value.

Interpolation baselines see the same hidden and visible sets as the model. The
burial channel is not an encoder input and is therefore not given to the
interpolation baselines; a constant training-mean baseline covers it instead.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from .masking import block_mask, gather_masked_batch

METHODS = ("model", "train_mean", "visible_mean", "nearest_visible", "idw")
INTERPOLATION_METHODS = ("visible_mean", "nearest_visible", "idw")
CHANNELS = ("sigma", "area", "potential", "aux")
IDW_EPS = 1e-6
IDW_POWER = -2


def mask_seed(cid, round_index, namespace="recon_v1"):
    """Deterministic per-molecule, per-round mask seed."""
    key = f"{namespace}|{int(cid)}|{int(round_index)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def draw_ranks(cids, round_index, n_pad, device, generator=None, namespace="recon_v1"):
    """Uniform ranks for one masking round, one seed per molecule."""
    if generator is None:
        generator = torch.Generator(device=device)
    ranks = torch.empty(len(cids), n_pad, device=device)
    for i, cid in enumerate(cids):
        generator.manual_seed(mask_seed(cid, round_index, namespace))
        ranks[i] = torch.rand(1, n_pad, device=device, generator=generator)[0]
    return ranks


def baseline_predictions(visible_targets, pv, dmv, n_masked):
    """Interpolation baselines evaluated at the masked positions.

    visible_targets (B, Pv, C) are the standardised targets of the visible
    patches, pv (B, Pv) marks valid visible slots and dmv (B, Pm, Pv) holds the
    masked-to-visible distances.
    """
    weights = pv.float()[..., None]
    out = {}

    mean = (visible_targets * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp(min=1)
    out["visible_mean"] = mean.expand(-1, n_masked, -1).clone()

    masked_distance = dmv.masked_fill(~pv[:, None, :], float("inf"))
    nearest = masked_distance.argmin(-1)
    out["nearest_visible"] = torch.gather(
        visible_targets, 1, nearest[..., None].expand(-1, -1, visible_targets.shape[-1]))

    inverse = ((dmv + IDW_EPS) ** IDW_POWER).masked_fill(~pv[:, None, :], 0.0)
    inverse = inverse / inverse.sum(-1, keepdim=True).clamp(min=1e-30)
    out["idw"] = torch.einsum("bmv,bvc->bmc", inverse, visible_targets)

    for name in INTERPOLATION_METHODS:
        out[name][..., 3] = float("nan")
    return out


@torch.no_grad()
def evaluate_round(model, features, xyz, pad, targets_std, cids, round_index, ratio,
                   device, block_radius=2.0, batch_size=1, autocast=True):
    """Run one masking round and return per-patch sums and coverage counts.

    `targets_std` (N, P, 4) holds the standardised targets in the order sigma,
    area, potential, burial. Returns a dictionary with the accumulated
    predictions per method, the coverage count and the per-molecule squared
    error of each method and channel.
    """
    n_molecule, n_pad = pad.shape
    n_method, n_channel = len(METHODS), len(CHANNELS)
    sums = torch.zeros(n_method, n_channel, n_molecule, n_pad, device=device)
    coverage = torch.zeros(n_molecule, n_pad, dtype=torch.int32, device=device)
    mse = torch.full((n_molecule, n_method, n_channel), float("nan"), device=device)
    hidden_first = torch.zeros(n_molecule, n_pad, dtype=torch.bool, device=device)

    ranks = draw_ranks(cids, round_index, n_pad, device)
    for start in range(0, n_molecule, batch_size):
        stop = min(start + batch_size, n_molecule)
        block = slice(start, stop)          # a slice keeps the in-place writes on the real tensor
        coords = xyz[block]
        dist = torch.cdist(coords, coords)
        hidden, visible = block_mask(dist, pad[block], ratio, rank=ranks[block],
                                     block_radius=block_radius)
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if (autocast and device.startswith("cuda")) else _null())
        with context:
            fv, dvv, pv, dmv, pm, iv, im = gather_masked_batch(
                features[block], coords, pad[block], hidden, visible)
            recon, aux_pred, _ = model(fv, dvv, pv, dmv, pm)
        prediction = torch.cat([recon.float(), aux_pred.float()[..., None]], -1)
        truth = torch.gather(targets_std[block], 1, im[..., None].expand(-1, -1, n_channel))
        visible_truth = torch.gather(targets_std[block], 1, iv[..., None].expand(-1, -1, n_channel))

        candidates = {"model": prediction,
                      "train_mean": torch.zeros_like(prediction),
                      **baseline_predictions(visible_truth, pv, dmv, prediction.shape[1])}

        weight = pm.float()
        for mi, name in enumerate(METHODS):
            error = candidates[name] - truth
            finite = torch.isfinite(error).float()
            numerator = (torch.nan_to_num(error ** 2) * weight[..., None] * finite).sum(1)
            denominator = (weight[..., None] * finite).sum(1)
            mse[block, mi] = torch.where(denominator > 0, numerator / denominator.clamp(min=1),
                                         torch.full_like(numerator, float("nan")))
            source = (torch.nan_to_num(candidates[name]) * weight[..., None]).permute(2, 0, 1).contiguous()
            sums[mi, :, block].scatter_add_(2, im[None].expand(n_channel, -1, -1).contiguous(), source)
        coverage[block].scatter_add_(1, im, pm.int())
        if round_index == 0:
            hidden_first[block] = hidden

    return {"sums": sums, "coverage": coverage, "mse": mse, "hidden_first": hidden_first}


class _null:
    """Context manager that does nothing, used when autocast is off."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def sign_agreement(reference, prediction):
    """Fraction of entries where reference and prediction share a sign."""
    reference = np.asarray(reference, float)
    prediction = np.asarray(prediction, float)
    return float((np.sign(reference) == np.sign(prediction)).mean())


def amplitude_ratio(reference, prediction):
    """Standard deviation of the prediction divided by that of the reference."""
    return float(np.asarray(prediction, float).std() / np.asarray(reference, float).std())
