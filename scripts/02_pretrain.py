#!/usr/bin/env python
"""Pre-train one encoder on a patch HDF5 file.

Usage:
    02_pretrain.py --patches patches.h5 --config configs/pretrain_M.yaml --out runs/M
    02_pretrain.py ... --max-steps 20 --limit 512     # short smoke run
"""
from __future__ import annotations

import argparse

import torch
import yaml

from cosmo3dformer.pretrain import PretrainConfig, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N molecules")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps")
    args = parser.parse_args()

    with open(args.config) as handle:
        raw = yaml.safe_load(handle)
    known = {f for f in PretrainConfig.__dataclass_fields__ if f != "extra"}
    config = PretrainConfig(**{k: v for k, v in raw.items() if k in known},
                            extra={k: v for k, v in raw.items() if k not in known})

    history = run(args.patches, config, args.out, device=args.device,
                  limit=args.limit, max_steps=args.max_steps)
    if history:
        last = history[-1]
        print(f"epochs {len(history)} last val_msm {last['val_msm']:.6f}", flush=True)


if __name__ == "__main__":
    main()
