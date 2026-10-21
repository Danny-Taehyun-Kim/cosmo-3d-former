#!/usr/bin/env python
"""Hide 90 percent of the acetamide surface and reconstruct it.

Usage:
    python examples/02_acetamide_reconstruction.py --cosmo data/demo/178.cosmo \
        --model weights/M_full_ep150.pt [--n-masks 24]

Reports the sign agreement and the error of the reconstructed screening-charge
density. Runs on CPU in well under a minute.

The decoder was trained against targets standardised by the statistics of the
pre-training set, so those statistics are what the predictions must be returned
on. They are read from configs/pretrain_target_stats.json by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cosmo3dformer.cosmo_io import read_cosmo
from cosmo3dformer.model import PatchMAE
from cosmo3dformer.patches import ENCODER_INPUT_COLS, build_patches, pad_patches
from cosmo3dformer.pretrain import AUX_COL, TARGET_COLS
from cosmo3dformer.reconstruction import evaluate_round, sign_agreement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-masks", type=int, default=24)
    parser.add_argument("--ratio", type=float, default=0.9)
    parser.add_argument("--cid", type=int, default=None)
    parser.add_argument("--target-stats",
                        default=str(Path(__file__).resolve().parents[1]
                                    / "configs" / "pretrain_target_stats.json"),
                        help="JSON with the pre-training target mean and std; pass an "
                             "empty string to standardise by this molecule instead, which "
                             "is not the scale the decoder was trained on")
    args = parser.parse_args()

    cid = args.cid if args.cid is not None else int(args.cosmo.split("/")[-1].split(".")[0])
    segments = read_cosmo(args.cosmo)
    patches, n_patch, _, _ = build_patches(segments)
    padded, mask = pad_patches(patches)

    state = torch.load(args.model, map_location="cpu", weights_only=False)
    model = PatchMAE(n_in=int(state["n_in"]), d=int(state.get("d_model", 256)),
                     h=int(state.get("n_heads", 8)),
                     encoder_bias_scale=float(state.get("encoder_bias_scale", 1.0)))
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    features = torch.from_numpy(padded[None, :, ENCODER_INPUT_COLS])
    xyz = torch.from_numpy(padded[None, :, :3])
    pad = torch.from_numpy(mask[None])
    targets = torch.from_numpy(padded[None, :, TARGET_COLS])
    aux = torch.from_numpy(padded[None, :, AUX_COL])

    if args.target_stats:
        stats = json.loads(Path(args.target_stats).read_text())
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        mu, sd, amu, asd = mean[:3], std[:3], mean[3], std[3]
        source = args.target_stats
    else:
        valid = pad
        mu = torch.stack([targets[..., k][valid].mean() for k in range(3)])
        sd = torch.stack([targets[..., k][valid].std() for k in range(3)])
        amu, asd = aux[valid].mean(), aux[valid].std()
        source = "this molecule"
    standardised = torch.cat([(targets - mu) / sd, ((aux - amu) / asd)[..., None]], -1)

    total = None
    coverage = None
    for round_index in range(args.n_masks):
        result = evaluate_round(model, features, xyz, pad, standardised, [cid], round_index,
                                args.ratio, "cpu", batch_size=1, autocast=False)
        total = result["sums"] if total is None else total + result["sums"]
        coverage = result["coverage"] if coverage is None else coverage + result["coverage"]

    scored = mask & (coverage[0].numpy() > 0)
    prediction = np.divide(total[0, 0, 0].numpy(), np.maximum(coverage[0].numpy(), 1))
    prediction = prediction * sd[0].item() + mu[0].item()
    reference = padded[:, TARGET_COLS[0]]

    print(f"patches            {n_patch}")
    print(f"target statistics  {source}")
    print(f"masks              {args.n_masks} at ratio {args.ratio}")
    print(f"scored patches     {int(scored.sum())}")
    print(f"sign agreement     {sign_agreement(reference[scored], prediction[scored]):.4f} "
          f"({int((np.sign(reference[scored]) == np.sign(prediction[scored])).sum())}"
          f"/{int(scored.sum())})")
    rmse = float(np.sqrt(((prediction[scored] - reference[scored]) ** 2).mean()))
    print(f"sigma RMSE         {rmse:.6f} e/A^2")
    print(f"sd ratio           {prediction[scored].std() / reference[scored].std():.4f}")


if __name__ == "__main__":
    main()
