#!/usr/bin/env python
"""Evaluate a frozen representation with ridge regression over the five folds.

Usage:
    04_ridge_eval.py --embeddings emb.parquet --labels labels.csv \
        --property Tb_K --config configs/ridge.yaml --out results.csv

The labels file must hold cid, fold and the property column. Property values
are not distributed with this repository; see data/README.md.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from cosmo3dformer.ridge import fit_predict, r2_score, rmse
from cosmo3dformer.splits import val_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--property", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clean2-column", default="clean2",
                        help="restrict to rows where this column is true, when present")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config) as handle:
            config = yaml.safe_load(handle)
    alphas = tuple(config.get("alphas", (0.1, 1.0, 10.0, 100.0)))
    transform = config.get("transform", {}).get(args.property, "standard")
    clean2_properties = set(config.get("clean2_properties", []))

    emb = pd.read_parquet(args.embeddings)
    emb["cid"] = emb.cid.astype(int)
    columns = [c for c in emb.columns if c.startswith("emb_")]
    table = pd.read_csv(args.labels)
    table["cid"] = table.cid.astype(int)
    table = table.dropna(subset=[args.property])
    # The validation split is drawn from the full labelled pool of the property,
    # before any consistency filter, and the filter is applied afterwards. This
    # is the order the published results used.
    unfiltered = table[["cid", "fold"]].copy()
    if args.property in clean2_properties and args.clean2_column in table.columns:
        table = table[table[args.clean2_column].astype(bool)]
    table = table.merge(emb[["cid"] + columns], on="cid", how="inner")

    rows = []
    for fold in sorted(table.fold.unique()):
        test = table[table.fold == fold]
        pool = table[table.fold != fold]
        val_cids = val_split(unfiltered.cid[unfiltered.fold != fold], fold,
                             frac=config.get("validation_fraction", 1 / 8))
        val = pool[pool.cid.isin(val_cids)]
        train = pool[~pool.cid.isin(val_cids)]
        if len(train) < 20 or len(val) < 5 or len(test) < 5:
            continue
        pred, alpha, val_score = fit_predict(
            train[columns].to_numpy(float), train[args.property].to_numpy(float),
            val[columns].to_numpy(float), val[args.property].to_numpy(float),
            test[columns].to_numpy(float), transform=transform, alphas=alphas)
        y_test = test[args.property].to_numpy(float)
        rows.append({"property": args.property, "fold": int(fold), "alpha": alpha,
                     "n_train": len(train), "n_val": len(val), "n_test": len(test),
                     "val_R2": val_score, "R2": r2_score(y_test, pred),
                     "RMSE": rmse(y_test, pred)})

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    print(f"wrote {args.out}", flush=True)
    print(f"median RMSE {np.median(frame.RMSE):.4f}  median R2 {np.median(frame.R2):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
