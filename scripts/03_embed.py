#!/usr/bin/env python
"""Extract frozen embeddings for every molecule in a patch HDF5 file.

Usage:
    03_embed.py --patches patches.h5 --encoder weights/M_encoder_ep146.pt --out emb.parquet
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
import pandas as pd
import torch

from cosmo3dformer.embed import embed_patches, load_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cids", default=None,
                        help="text file with one identifier per line; embed only these")
    args = parser.parse_args()

    encoder, state = load_encoder(args.encoder, device=args.device)
    with h5py.File(args.patches, "r") as handle:
        cols = [int(c) for c in handle.attrs["encoder_input_cols"]]
        if list(state["cols"]) != cols:
            raise SystemExit("encoder input columns differ between checkpoint and HDF5")
        cid = handle["cid"][:]
        if args.cids:
            wanted = {int(x) for x in open(args.cids).read().split()}
            take = np.sort(np.flatnonzero(np.isin(cid, list(wanted))))
            if len(take) == 0:
                raise SystemExit("none of the requested identifiers are in the patch file")
            cid = cid[take]
            n_patch = torch.from_numpy(handle["n_patch"][:][take].astype(np.int64))
            patches = torch.from_numpy(handle["patches"][:][take])
        else:
            n_patch = torch.from_numpy(handle["n_patch"][:].astype(np.int64))
            patches = torch.from_numpy(handle["patches"][:])
    pad = (torch.arange(patches.shape[1])[None, :] < n_patch[:, None]).to(args.device)
    features = patches[:, :, cols].to(args.device)
    xyz = patches[:, :, :3].to(args.device)
    del patches

    emb = embed_patches(encoder, features, xyz, pad, batch_size=args.batch_size,
                        device=args.device)
    frame = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    frame.insert(0, "cid", cid.astype(np.int64))
    frame.to_parquet(args.out)
    print(f"wrote {args.out}: {len(frame)} molecules by {emb.shape[1]} dimensions", flush=True)


if __name__ == "__main__":
    main()
