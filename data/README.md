# Data

This directory holds identifiers, fold assignments and consistency flags only.
It contains no property values.

## splits/

| File | Contents |
|------|----------|
| `molecules.csv` | 8,280 rows. `cid`, `fold`, one `has_<property>` indicator per property, and the critical-property consistency flags |
| `limited_data_subsets.csv.gz` | membership of every limited-data subset: property, fold, budget `N`, subset seed, train or validation split, and `cid` |

The fold column is a molecule-disjoint five-fold split shared by every property
and every model. Do not re-split.

`has_<property>` is 1 when the molecule carries a label for that property in the
benchmark and 0 otherwise. The label itself is not distributed here.

The consistency flags follow the definition in `cosmo3dformer.splits.zc_flags`.
`clean2` is the conjunction of a compressibility factor inside [0.18, 0.35], a
label critical pressure no more than twice the Joback estimate, and a Joback
critical volume no more than 1.5 times the label. The compressibility factor
itself is a function of three property values and is therefore not stored; only
the resulting flags are.

## demo/

A handful of `.cosmo` files used by the examples and the tests. The full
surface corpus is not part of this repository.

## Obtaining the inputs

**COSMO surfaces.** The surface corpus comes from the PubChemQC COSMO
sigma-profile database. See the Zenodo deposit listed in the top-level README
for the `.cosmo` and `.sigma` files and for the molecule index.

**Property labels.** Pure-component property values come from the Korea
Thermophysical Properties Data Bank (KDB) and are subject to the KDB terms of
use, so the values are not included here. Obtain them from KDB
directly and join them to `splits/molecules.csv` on `cid`. A labels file for
the evaluation scripts needs the columns `cid`, `fold` and one column per
property, named as in `has_<property>` without the prefix.
