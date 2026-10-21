#!/usr/bin/env python
"""Aggregate .cosmo surfaces into padded patch tensors and write an HDF5 file.

Usage:
    01_build_patches.py --cosmo-dir DIR --out patches.h5 [--workers N] [--limit N]

The output holds patches (M, 192, 18), patch_mask, n_patch, capped,
normal_fit_frac and cid, with the feature layout recorded in the attributes.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import h5py
import numpy as np

from cosmo3dformer.cosmo_io import read_cosmo
from cosmo3dformer.patches import (ENCODER_INPUT_COLS, N_DIM, N_RAY, PAD, R_AV2,
                                   TARGET_AREA, Z_LIST, build_patches, pad_patches)

MIN_SEGMENTS = 10


def one(path):
    """Build the patch array for a single .cosmo file."""
    cid = int(Path(path).stem)
    try:
        segments = read_cosmo(path)
        if segments is None or len(segments) < MIN_SEGMENTS:
            return cid, None, None, 0, False, 0.0, "too few segments"
        patches, n_patch, capped, fit = build_patches(segments)
        padded, mask = pad_patches(patches)
        return cid, padded, mask, n_patch, capped, fit, ""
    except Exception as exc:  # noqa: BLE001
        return cid, None, None, 0, False, 0.0, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmo-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(Path(args.cosmo_dir).glob("*.cosmo"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no .cosmo files under {args.cosmo_dir}")
    print(f"{len(paths)} molecules, {args.workers} workers", flush=True)

    n = len(paths)
    out = np.zeros((n, PAD, N_DIM), np.float32)
    mask = np.zeros((n, PAD), bool)
    n_patch = np.zeros(n, np.int32)
    capped = np.zeros(n, bool)
    fit = np.zeros(n, np.float32)
    cid = np.zeros(n, np.int64)
    failures = []

    with mp.Pool(args.workers) as pool:
        for i, (c, padded, m, np_, cp, ff, err) in enumerate(pool.imap(one, paths, chunksize=4)):
            cid[i] = c
            if err:
                failures.append((c, err))
                continue
            out[i], mask[i], n_patch[i], capped[i], fit[i] = padded, m, np_, cp, ff
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{n}", flush=True)

    with h5py.File(args.out, "w") as handle:
        handle.create_dataset("cid", data=cid)
        handle.create_dataset("patches", data=out, chunks=(64, PAD, N_DIM), compression="lzf")
        handle.create_dataset("patch_mask", data=mask, chunks=(512, PAD), compression="lzf")
        handle.create_dataset("n_patch", data=n_patch)
        handle.create_dataset("capped", data=capped)
        handle.create_dataset("normal_fit_frac", data=fit)
        handle.attrs["layout"] = ("0-2 xyz (not an encoder input) |3 area|4 sig_raw|5 sig_raw_var|"
                                  "6 sig_avg|7 sig_avg_var|8 potential|9 solid_angle (aux target)|"
                                  "10-17 Zfrac")
        handle.attrs["encoder_input_cols"] = np.array(ENCODER_INPUT_COLS, dtype=np.int32)
        handle.attrs["Z_list"] = np.array(Z_LIST, dtype=np.int32)
        handle.attrs["target_area"] = TARGET_AREA
        handle.attrs["pad"] = PAD
        handle.attrs["r_av2"] = R_AV2
        handle.attrs["units"] = "Angstrom"
        handle.attrs["n_ray"] = N_RAY

    ok = int((n_patch > 0).sum())
    print(f"wrote {args.out}: {ok}/{n} molecules, {len(failures)} failures", flush=True)
    for c, err in failures[:5]:
        print(f"  {c}: {err}", flush=True)


if __name__ == "__main__":
    main()
