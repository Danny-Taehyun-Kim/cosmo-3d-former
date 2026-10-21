#!/usr/bin/env python
"""Fine-tune a pre-trained encoder on one property and one fold.

Usage:
    05_finetune.py --patches patches.h5 --encoder weights/M_encoder_ep146.pt \
        --labels labels.csv --property Tb_K --fold 0 \
        --config configs/finetune.yaml --out fold0.csv
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from cosmo3dformer.finetune import FinetuneConfig, finetune
from cosmo3dformer.ridge import r2_score, rmse
from cosmo3dformer.splits import CLEAN2_PROPERTIES, TRANSFORM, val_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--property", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None, help="override for a smoke run")
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N molecules of the patch file")
    parser.add_argument("--train-cids", default=None,
                        help="text file of identifiers to train on, for limited-label runs")
    parser.add_argument("--val-cids", default=None,
                        help="text file of identifiers to validate on, for limited-label runs")
    parser.add_argument("--clean2-column", default="clean2",
                        help="restrict the critical-property runs to rows where this is true")
    args = parser.parse_args()

    with open(args.config) as handle:
        raw = yaml.safe_load(handle)
    known = set(FinetuneConfig.__dataclass_fields__)
    config = FinetuneConfig(**{k: v for k, v in raw.items() if k in known})
    if args.epochs is not None:
        config.epochs = args.epochs
        config.stage1_epochs = min(config.stage1_epochs, args.epochs)

    state = torch.load(args.encoder, map_location="cpu", weights_only=False)
    with h5py.File(args.patches, "r") as handle:
        cols = [int(c) for c in handle.attrs["encoder_input_cols"]]
        total = handle["cid"].shape[0]
        n = min(total, args.limit) if args.limit else total
        cid = handle["cid"][:n].astype(np.int64)
        n_patch = torch.from_numpy(handle["n_patch"][:n].astype(np.int64))
        patches = torch.from_numpy(handle["patches"][:n])
    pad = (torch.arange(patches.shape[1])[None, :] < n_patch[:, None]).to(args.device)
    features = patches[:, :, cols].to(args.device)
    xyz = patches[:, :, :3].to(args.device)
    del patches

    position = {int(c): i for i, c in enumerate(cid)}
    table = pd.read_csv(args.labels)
    table["cid"] = table.cid.astype(int)
    table = table.dropna(subset=[args.property])
    table = table[table.cid.isin(position)]
    # The validation split is drawn from the full labelled pool, before the
    # consistency filter, which is the order the published results used.
    unfiltered_pool = table.cid[table.fold != args.fold]
    if args.property in CLEAN2_PROPERTIES and args.clean2_column in table.columns:
        table = table[table[args.clean2_column].astype(bool)]

    test = table[table.fold == args.fold]
    pool = table[table.fold != args.fold]
    if args.train_cids and args.val_cids:
        train_set = {int(x) for x in open(args.train_cids).read().split()}
        val_set = {int(x) for x in open(args.val_cids).read().split()}
        train = pool[pool.cid.isin(train_set)]
        val = pool[pool.cid.isin(val_set)]
    else:
        val_cids = val_split(unfiltered_pool, args.fold)
        val = pool[pool.cid.isin(val_cids)]
        train = pool[~pool.cid.isin(val_cids)]

    to_index = lambda frame: torch.tensor([position[c] for c in frame.cid], device=args.device)
    pred, best_val = finetune(
        state, features, xyz, pad,
        train[args.property].to_numpy(float), val[args.property].to_numpy(float),
        to_index(train), to_index(val), to_index(test),
        TRANSFORM.get(args.property, "standard"), config, device=args.device, fold=args.fold)

    y_test = test[args.property].to_numpy(float)
    frame = pd.DataFrame([{"property": args.property, "fold": args.fold,
                           "n_train": len(train), "n_val": len(val), "n_test": len(test),
                           "val_loss": best_val, "R2": r2_score(y_test, pred),
                           "RMSE": rmse(y_test, pred)}])
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
