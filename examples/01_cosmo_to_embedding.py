#!/usr/bin/env python
"""Turn a .cosmo surface into patches and then into a frozen embedding.

Usage:
    python examples/01_cosmo_to_embedding.py --cosmo data/demo/178.cosmo \
        [--encoder weights/M_encoder_ep146.pt]

Without an encoder the script stops after the patch stage, which needs no
weights and runs on CPU in a few seconds.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from cosmo3dformer.cosmo_io import cavity_area, read_cosmo
from cosmo3dformer.embed import embed_patches, load_encoder
from cosmo3dformer.patches import ENCODER_INPUT_COLS, build_patches, pad_patches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmo", required=True)
    parser.add_argument("--encoder", default=None)
    args = parser.parse_args()

    text = open(args.cosmo).read()
    segments = read_cosmo(args.cosmo)
    print(f"segments           {len(segments)}")
    print(f"cavity area        {cavity_area(text):.3f} A^2 (header)")
    print(f"segment area sum   {segments[:, 4].sum():.3f} A^2")

    patches, n_patch, capped, fit = build_patches(segments)
    print(f"patches            {n_patch}  capped={capped}  normal fit {fit:.3f}")
    print(f"patch area sum     {patches[:, 3].sum():.3f} A^2")
    print(f"mean patch area    {patches[:, 3].mean():.3f} A^2")
    print(f"sigma range        [{patches[:, 4].min():.5f}, {patches[:, 4].max():.5f}] e/A^2")

    if args.encoder is None:
        print("\nno encoder given, stopping after the patch stage")
        return

    padded, mask = pad_patches(patches)
    features = torch.from_numpy(padded[None, :, ENCODER_INPUT_COLS])
    xyz = torch.from_numpy(padded[None, :, :3])
    pad = torch.from_numpy(mask[None])
    encoder, state = load_encoder(args.encoder, device="cpu")
    embedding = embed_patches(encoder, features, xyz, pad, device="cpu")
    print(f"\nencoder            d={state.get('d_model')} epoch={state.get('epoch')}")
    print(f"embedding          {embedding.shape[1]} dimensions")
    print(f"norm               {np.linalg.norm(embedding):.4f}")


if __name__ == "__main__":
    main()
