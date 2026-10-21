"""Spatial block masking used by the masked-autoencoder objective.

A uniform rank is drawn per patch. Each patch then takes the smallest rank
among the patches within BLOCK_RADIUS of it, so neighbouring patches share a
rank and are hidden together. The lowest-coverage patches up to the requested
ratio are masked. Molecules that would be left with fewer than three visible
patches get three restored.
"""
from __future__ import annotations

import torch

BLOCK_RADIUS = 2.0


def block_mask(dist, pad, ratio, generator=None, rank=None, block_radius=BLOCK_RADIUS):
    """Return (hidden, visible) boolean masks of shape (B, P).

    dist   (B, P, P) patch-centre distances in Angstrom
    pad    (B, P)    True for valid patches
    ratio  float     fraction of valid patches to hide
    generator        torch generator used to draw the ranks
    rank   (B, P)    optional pre-drawn uniform ranks, used instead of generator

    The first returned mask is the hidden set, the second the visible set.
    """
    batch, n_pad = pad.shape
    if rank is None:
        rank = torch.rand(batch, n_pad, device=dist.device, generator=generator)
    rank = rank.masked_fill(~pad, 2.0)

    adjacency = (dist <= block_radius) & pad[:, :, None] & pad[:, None, :]
    coverage = torch.where(adjacency, rank[:, :, None].expand(-1, -1, n_pad),
                           torch.full_like(dist, 2.0)).amin(1).masked_fill(~pad, 2.0)

    n_valid = pad.sum(1)
    need = (ratio * n_valid).round().long().clamp(min=1)
    ordered, _ = coverage.sort(1)
    threshold = ordered.gather(1, (need - 1).clamp(min=0)[:, None])
    hidden = (coverage <= threshold) & pad

    too_few = ((~hidden) & pad).sum(1) < 3
    if too_few.any():
        for b in torch.where(too_few)[0].tolist():
            available = torch.where(pad[b])[0]
            hidden[b, available[:3]] = False
    return hidden, (~hidden) & pad


def gather_masked_batch(features, xyz, pad, hidden, visible, targets=None, aux=None):
    """Reorder a batch into visible and masked blocks for the model forward pass.

    Returns fv, dvv, pv, dmv, pm and, when targets are given, the gathered
    targets together with the visible and masked index tensors.
    """
    device = features.device
    n_pad = pad.shape[1]
    n_vis = int(visible.sum(1).max())
    n_mask = int(hidden.sum(1).max())
    tiebreak = torch.arange(n_pad, device=device)[None, :] * 1e-6

    iv = torch.argsort((~visible).float() + tiebreak, dim=1)[:, :n_vis]
    im = torch.argsort((~hidden).float() + tiebreak, dim=1)[:, :n_mask]
    pv = torch.gather(visible, 1, iv)
    pm = torch.gather(hidden, 1, im)

    fv = torch.gather(features, 1, iv[..., None].expand(-1, -1, features.shape[-1]))
    xv = torch.gather(xyz, 1, iv[..., None].expand(-1, -1, 3))
    xm = torch.gather(xyz, 1, im[..., None].expand(-1, -1, 3))

    out = [fv, torch.cdist(xv, xv), pv, torch.cdist(xm, xv), pm]
    if targets is not None:
        out.append(torch.gather(targets, 1, im[..., None].expand(-1, -1, targets.shape[-1])))
    if aux is not None:
        out.append(torch.gather(aux, 1, im))
    out += [iv, im]
    return tuple(out)
