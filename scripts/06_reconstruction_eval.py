#!/usr/bin/env python
"""Evaluate masked surface reconstruction over a set of molecules.

Usage:
    06_reconstruction_eval.py --patches patches.h5 --model weights/M_full_ep150.pt \
        --out recon.npz [--n-masks 24] [--ratio 0.9] [--limit 0]

Writes the accumulated per-patch predictions, the coverage count, the
per-molecule squared error of every method and channel, and the mask-zero
hidden set. Statistics are produced by 07_make_figures.py.
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch

from cosmo3dformer.model import PatchMAE
from cosmo3dformer.pretrain import AUX_COL, TARGET_COLS
from cosmo3dformer.reconstruction import (CHANNELS, INTERPOLATION_METHODS, METHODS,
                                           evaluate_round)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-masks", type=int, default=24)
    parser.add_argument("--ratio", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-stats", default=None,
                        help="JSON with the pre-training target mean and std; without it "
                             "the statistics are taken from the molecules given here, which "
                             "is not the scale the decoder was trained on")
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    model = PatchMAE(n_in=int(state["n_in"]), d=int(state.get("d_model", 256)),
                     h=int(state.get("n_heads", 8)),
                     encoder_bias_scale=float(state.get("encoder_bias_scale", 1.0)))
    model.load_state_dict(state["model"], strict=True)
    model = model.to(args.device).eval()

    with h5py.File(args.patches, "r") as handle:
        cols = [int(c) for c in handle.attrs["encoder_input_cols"]]
        n_total = handle["cid"].shape[0]
        n = min(n_total, args.limit) if args.limit else n_total
        cid = handle["cid"][:n].astype(np.int64)
        n_patch = torch.from_numpy(handle["n_patch"][:n].astype(np.int64))
        patches = torch.from_numpy(handle["patches"][:n])
    pad = (torch.arange(patches.shape[1])[None, :] < n_patch[:, None]).to(args.device)
    features = patches[:, :, cols].to(args.device)
    xyz = patches[:, :, :3].to(args.device)
    targets = patches[:, :, TARGET_COLS].to(args.device)
    aux = patches[:, :, AUX_COL].to(args.device)
    del patches

    if args.target_stats:
        import json as _json
        stats = _json.loads(open(args.target_stats).read())
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=args.device)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=args.device)
        mu, sd, amu, asd = mean[:3], std[:3], mean[3], std[3]
    else:
        valid = pad
        mu = torch.stack([targets[..., k][valid].mean() for k in range(3)])
        sd = torch.stack([targets[..., k][valid].std() for k in range(3)])
        amu, asd = aux[valid].mean(), aux[valid].std()
    standardised = torch.cat([(targets - mu) / sd, ((aux - amu) / asd)[..., None]], -1)

    sums = None
    coverage = None
    mse = []
    hidden_first = None
    for round_index in range(args.n_masks):
        result = evaluate_round(model, features, xyz, pad, standardised, cid, round_index,
                                args.ratio, args.device, batch_size=args.batch_size)
        sums = result["sums"] if sums is None else sums + result["sums"]
        coverage = result["coverage"] if coverage is None else coverage + result["coverage"]
        mse.append(result["mse"].cpu().numpy())
        if round_index == 0:
            hidden_first = result["hidden_first"].cpu().numpy()
        print(f"  mask {round_index} done", flush=True)

    # Burial is not an encoder input, so it is not handed to the interpolation
    # baselines. Mark those cells as absent rather than leaving the zero that the
    # accumulation wrote, which would read as the training-mean prediction.
    sums = sums.cpu().numpy()
    burial = CHANNELS.index("aux")
    for name in INTERPOLATION_METHODS:
        sums[METHODS.index(name), burial] = np.nan
    np.savez(args.out, sums=sums, coverage=coverage.cpu().numpy(),
             mse=np.stack(mse, axis=1), hidden_first=hidden_first, cid=cid,
             not_given=np.array([f"{m}:aux" for m in INTERPOLATION_METHODS]),
             pad=pad.cpu().numpy(), targets_std=standardised.cpu().numpy(),
             mu=np.append(mu.cpu().numpy(), amu.item()),
             sd=np.append(sd.cpu().numpy(), asd.item()),
             methods=np.array(METHODS), channels=np.array(CHANNELS),
             n_masks=args.n_masks, ratio=args.ratio,
             checkpoint_epoch=state.get("epoch", -1),
             target_stats_source=(args.target_stats or "computed from the given molecules"))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
