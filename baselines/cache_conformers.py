#!/usr/bin/env python
"""Build the conformer cache used by the Uni-Mol baselines.

Usage:
    cache_conformers.py --smiles smiles.csv --out conformers.pkl [--workers N]

The input needs the columns cid and smiles. For each molecule ten conformers
are embedded with RDKit ETKDGv3 at random seed 42, optimised with MMFF, and the
lowest-energy one is kept. Embedding falls back to random coordinates when the
first attempt returns nothing. The result is deterministic, so every fold and
every budget sees the same geometry.
"""
from __future__ import annotations

import argparse
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

CONFORMER_SEED = 42
N_CONFORMERS = 10


def generate(item):
    """Return (cid, (atom symbols, coordinates)) or (cid, None) on failure."""
    cid, smiles = item
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return cid, None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = CONFORMER_SEED
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFORMERS, params=params)
        if len(ids) == 0:
            params.useRandomCoords = True
            ids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFORMERS, params=params)
            if len(ids) == 0:
                return cid, None
        energies = [value for _, value in AllChem.MMFFOptimizeMoleculeConfs(mol)]
        best = int(np.argmin(energies)) if energies else 0
        conformer = mol.GetConformer(ids[best])
        xyz = np.array([[conformer.GetAtomPosition(i).x,
                         conformer.GetAtomPosition(i).y,
                         conformer.GetAtomPosition(i).z]
                        for i in range(mol.GetNumAtoms())], dtype=np.float32)
        return cid, ([atom.GetSymbol() for atom in mol.GetAtoms()], xyz)
    except Exception:  # noqa: BLE001
        return cid, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True, help="CSV with cid and smiles")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    table = pd.read_csv(args.smiles)
    items = sorted((int(r.cid), r.smiles) for r in table.itertuples())
    print(f"{len(items)} molecules", flush=True)
    with Pool(args.workers) as pool:
        results = pool.map(generate, items, chunksize=20)
    cache = {cid: value for cid, value in results if value is not None}
    with open(args.out, "wb") as handle:
        pickle.dump(cache, handle)
    print(f"wrote {args.out}: {len(cache)} molecules, {len(items) - len(cache)} failures",
          flush=True)


if __name__ == "__main__":
    main()
