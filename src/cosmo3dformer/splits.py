"""Fold assignment, target transforms, validation splits and limited-data subsets.

The benchmark table supplies a molecule-disjoint five-fold split. For a given
test fold the remaining four folds form the pool; one eighth of the pool is set
aside as validation with a fold-dependent seed. Limited-data subsets are drawn
from the pool stratified by target decile and are nested, so the subset for a
smaller budget is a prefix of the subset for a larger one.

Property values are never stored in this repository. The split tables under
data/splits hold identifiers, fold indices and flags only.
"""
from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

TRANSFORM = {"Tb_K": "log1p", "Tc_K": "standard", "Pc_kPa": "log1p",
             "Vc_L_per_mol": "log", "Tm_K": "standard", "Tt_K": "standard",
             "flash_K": "standard", "autoign_K": "standard", "logKow": "standard"}

CLEAN2_PROPERTIES = ("Pc_kPa", "Vc_L_per_mol")
SUBSET_SIZES = (50, 100, 200, 400, 800, 1600)
SUBSET_SEEDS = (0, 1, 2)
N_DECILE = 10
VAL_FRACTION = 1 / 8
VAL_SEED_BASE = 42

PROPERTY_RNG_OFFSET = {"Tb_K": 1, "flash_K": 2, "Pc_kPa": 3, "Tm_K": 4,
                       "Tc_K": 5, "logKow": 6}
CLEAN2_RNG_OFFSET = {"Pc_kPa": 23, "Vc_L_per_mol": 24}

GAS_CONSTANT_KPA_L = 8.314
ZC_RANGE = (0.18, 0.35)
PC_RATIO_LIMIT = 2.0
VC_RATIO_LIMIT = 1.5


class Target:
    """Per-property target transform followed by standardisation.

    The non-linear step is one of standard, log1p or log. Mean and standard
    deviation are fitted on training labels only.
    """

    def __init__(self, mode="standard"):
        if mode not in ("standard", "log1p", "log"):
            raise ValueError(f"transform mode={mode}")
        self.mode = mode
        self.mean = None
        self.std = None

    def _pre(self, y):
        y = np.asarray(y, float)
        if self.mode == "log1p":
            if y.min() <= -1.0:
                raise ValueError(f"log1p invalid: min(y)={y.min()}")
            return np.log1p(y)
        if self.mode == "log":
            if y.min() <= 0.0:
                raise ValueError(f"log invalid: min(y)={y.min()}")
            return np.log(y)
        return y

    def fit(self, y):
        z = self._pre(y)
        self.mean = float(z.mean())
        self.std = float(z.std()) + 1e-9
        return self

    def forward(self, y):
        return (self._pre(y) - self.mean) / self.std

    def inverse(self, z):
        z = np.asarray(z, float) * self.std + self.mean
        if self.mode == "log1p":
            return np.expm1(z)
        if self.mode == "log":
            return np.exp(z)
        return z


def val_split(pool_cids, fold_k, frac=VAL_FRACTION):
    """Deterministic validation subset of the non-test pool for one fold."""
    rng = np.random.RandomState(VAL_SEED_BASE + int(fold_k))
    cids = np.array(sorted(int(c) for c in pool_cids))
    rng.shuffle(cids)
    n_val = int(round(len(cids) * frac))
    return {int(c) for c in cids[:n_val]}


def stratified_order(cids, y, rng, n_bin=N_DECILE):
    """Interleave molecules by target decile so any prefix stays stratified."""
    cids = np.asarray(cids)
    y = np.asarray(y, float)
    ranks = pd.Series(y).rank(method="first").values
    bins = np.minimum(((ranks - 1) * n_bin // len(y)).astype(int), n_bin - 1)
    per_bin = [list(cids[bins == i]) for i in range(n_bin)]
    for entries in per_bin:
        rng.shuffle(entries)
    pointer = [0] * n_bin
    order = []
    while len(order) < len(cids):
        for i in range(n_bin):
            if pointer[i] < len(per_bin[i]):
                order.append(int(per_bin[i][pointer[i]]))
                pointer[i] += 1
    return order


def subset_val_indices(size):
    """Positions inside a subset of `size` molecules that form its validation part."""
    n_val = min(max(8, ceil(0.15 * size)), size - 1)
    idx = set(np.unique(np.round(np.linspace(0, size - 1, n_val)).astype(int)).tolist())
    i = 0
    while len(idx) < n_val:
        if i not in idx:
            idx.add(i)
        i += 1
    return idx


def subset_rng(prop, fold_k, seed, clean2=False):
    """Random generator for one limited-data subset list."""
    offsets = CLEAN2_RNG_OFFSET if clean2 else PROPERTY_RNG_OFFSET
    if prop not in offsets:
        raise KeyError(f"no RNG offset registered for {prop} (clean2={clean2})")
    return np.random.default_rng(10000 * int(seed) + 100 * int(fold_k) + offsets[prop])


def zc_flags(tc_k, pc_kpa, vc_l_per_mol, pc_joback, vc_joback):
    """Consistency flags for critical-property labels.

    Zc is the compressibility factor built from the three critical properties.
    A label set is called clean2 when Zc lies in ZC_RANGE, the label critical
    pressure is not more than PC_RATIO_LIMIT times the Joback estimate, and the
    Joback critical volume is not more than VC_RATIO_LIMIT times the label.
    """
    zc = pc_kpa * vc_l_per_mol / (GAS_CONSTANT_KPA_L * tc_k)
    consistent = (zc >= ZC_RANGE[0]) & (zc <= ZC_RANGE[1])
    pc_bad = (pc_kpa / pc_joback) > PC_RATIO_LIMIT
    vc_bad = (vc_joback / vc_l_per_mol) > VC_RATIO_LIMIT
    return pd.DataFrame({"Zc": zc, "zc_consistent": consistent,
                         "flag_pc_bad": pc_bad, "flag_vc_bad": vc_bad,
                         "clean2": consistent & ~pc_bad & ~vc_bad})


def load_split_table(path, prop):
    """Read a split table from data/splits and return cid and fold for one property."""
    table = pd.read_csv(path)
    column = f"has_{prop}"
    if column not in table.columns:
        raise KeyError(f"{column} missing from {path}")
    sub = table.loc[table[column] == 1, ["cid", "fold"]].copy()
    sub["cid"] = sub.cid.astype(int)
    sub["fold"] = sub.fold.astype(int)
    return sub.reset_index(drop=True)
