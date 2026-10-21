# COSMO-3D-Former

A patch-based encoder for COSMO molecular surfaces. The model is pre-trained by
masked surface reconstruction on 362,303 molecules and is used either as a
frozen representation or as a fine-tuned predictor of pure-component
thermophysical properties.

## Overview

A COSMO calculation produces a few thousand surface segments per molecule, each
carrying a screening-charge density, an area and a solute potential. The
segments are aggregated into roughly 120 patches of about 2.1 square Angstrom
each. A patch holds twelve rotation-invariant features: area, the mean and
variance of the raw screening-charge density, the solute potential, and the
area fraction of eight elements. Coordinates never enter the encoder; only the
patch-centre distance matrix does, so rotation invariance holds by
construction.

Pre-training hides 90 percent of the patches in spatial blocks and asks the
decoder to reconstruct the screening-charge density, the area and the solute
potential of the hidden patches, with a burial measure as an auxiliary target.
Masked queries are copies of one learned token and attend to visible patches
only, so no hidden feature can reach the decoder.

Three encoder widths are released. M is the model used throughout the paper.

| Model | Width | Layers | Heads | Encoder parameters | Embedding |
|-------|------:|-------:|------:|-------------------:|----------:|
| S | 256 | 8 | 8 | 6,338,496 | 512 |
| M | 480 | 8 | 8 | 22,192,096 | 960 |
| L | 688 | 8 | 8 | 45,539,472 | 1,376 |

The decoder adds 440,204, 468,876 and 495,500 parameters respectively.

## Repository contents

| Path | Contents |
|------|----------|
| `src/cosmo3dformer/` | library: surface reading, patch construction, masking, model, pre-training, embedding, ridge, fine-tuning, splits, reconstruction |
| `configs/` | pre-training, fine-tuning and ridge settings matching Supplementary Table 2 |
| `scripts/` | numbered pipeline drivers and the weight downloader |
| `baselines/` | wrappers, settings and conformer generation for the four comparison models |
| `data/` | fold assignments, limited-data subset membership, consistency flags and demo surfaces |
| `examples/` | two short demonstrations that run on CPU |
| `tests/` | unit tests that need no property labels |

## System requirements

Tested on Linux (Red Hat Enterprise Linux 9.6, kernel 5.14) with Python 3.10.20,
PyTorch 2.5.1 built against CUDA 12.1, and NVIDIA A100 PCIE 40 GB accelerators.

A GPU is required for pre-training and fine-tuning. Patch construction, frozen
embedding of a handful of molecules, ridge evaluation and both examples run on
CPU. No non-standard hardware is needed.

## Installation

    conda env create -f environment.yml
    conda activate cosmo3dformer
    pip install -e .

Installation takes about ten minutes on a normal desktop, dominated by the
PyTorch download. Installing into an existing environment with `pip install -e .`
takes under a minute.

## Demo

Build patches from a packaged surface and inspect them. No weights needed.

    python examples/01_cosmo_to_embedding.py --cosmo data/demo/887.cosmo

Expected output:

    segments           921
    cavity area        68.250 A^2 (header)
    segment area sum   68.251 A^2
    patches            33  capped=False  normal fit 1.000
    patch area sum     68.251 A^2
    mean patch area    2.068 A^2
    sigma range        [-0.01565, 0.01579] e/A^2

    no encoder given, stopping after the patch stage

Runtime is a few seconds on one CPU core. Adding `--encoder` prints the
960-dimensional embedding norm as well.

Reconstruct a masked surface:

    python examples/02_acetamide_reconstruction.py \
        --cosmo data/demo/178.cosmo --model weights/M_full_ep150.pt

This hides 90 percent of the patches, repeats the masking 24 times, and reports
the sign agreement of the reconstructed screening-charge density. Expected
output:

    patches            47
    target statistics  configs/pretrain_target_stats.json
    masks              24 at ratio 0.9
    scored patches     47
    sign agreement     0.9787 (46/47)
    sigma RMSE         0.002542 e/A^2
    sd ratio           0.7900

Runtime is under a minute on CPU. The example runs on CPU throughout, so these
values do not depend on whether a GPU is present. The same calculation on a GPU
gives a sigma RMSE of 0.002383 over the same patches: with 47 patches and 24
masks the ensemble is small enough that the order of summation matters. Figure
2b comes from the GPU path in `scripts/06_reconstruction_eval.py`, over the
whole validation set, where the per-patch counts are far larger and this
sensitivity washes out.

The standardisation is the one the decoder was trained with, read from
`configs/pretrain_target_stats.json`. Standardising by a single molecule
instead inflates the amplitude and is not a valid comparison.

## Usage

Build patches for your own surfaces:

    python scripts/01_build_patches.py --cosmo-dir my_cosmo --out my_patches.h5

Extract frozen embeddings:

    python scripts/03_embed.py --patches my_patches.h5 \
        --encoder weights/M_encoder_ep146.pt --out my_embeddings.parquet

Evaluate a property with ridge regression. The labels file needs `cid`, `fold`
and the property column; see `data/README.md`.

    python scripts/04_ridge_eval.py --embeddings my_embeddings.parquet \
        --labels my_labels.csv --property Tb_K \
        --config configs/ridge.yaml --out ridge_Tb.csv

Fine-tune on one property and one fold:

    python scripts/05_finetune.py --patches my_patches.h5 \
        --encoder weights/M_encoder_ep146.pt --labels my_labels.csv \
        --property Tb_K --fold 0 --config configs/finetune.yaml --out ft_Tb_f0.csv

## Input surfaces

The model reads DMol3-style `.cosmo` files. The segment table is introduced by a
line containing `position (X, Y, Z)` and holds nine columns:

| Column | Meaning | Unit |
|--------|---------|------|
| n | segment index | |
| atom | index of the atom the segment belongs to | |
| X, Y, Z | segment centre | atomic units, converted to Angstrom on read |
| charge | segment polarisation charge | e |
| area | segment area | Angstrom squared |
| charge/area | screening-charge density | e per Angstrom squared |
| potential | solute potential on the segment | e per Angstrom |

`cosmo3dformer.cosmo_io.read_cosmo` returns x, y, z in Angstrom, the screening
charge density as charge divided by area, the area, the solute potential and the
atomic number. Segments below 5e-6 Angstrom squared are dropped and the surface
is translated so that its area-weighted centroid sits at the origin. Note that
the solute potential is in e per Angstrom, not kcal per mole per elementary
charge; 1 e per Angstrom equals 14.3996 V.

The surfaces used in this work were produced with PySCF at the BP86 level with
the def2-SVP basis set, in the conductor-screening limit of the COSMO model
(permittivity infinite), on PM6 geometries taken from PubChemQC without further
optimisation. Cavity radii are the Klamt values for H, C, N, O, F, P, S and Cl
and modified Bondi radii elsewhere, with a van der Waals scale of 1.0.

## Preparing surfaces for new molecules

This repository consumes `.cosmo` files; it does not generate them. The
generation pipeline lives with the sigma-profile database. For a molecule that
is not already in PubChemQC it runs

    SMILES
      -> RDKit ETKDGv3 embedding with UFF pre-optimisation
      -> MOPAC PM6 geometry optimisation
      -> PySCF BP86/def2-SVP with COSMO at infinite permittivity,
         Klamt radii, van der Waals scale 1.0
      -> DMol3-style .cosmo

Molecules already in PubChemQC carry the PM6 geometry, so only the PySCF step
runs. The code is at:

    https://github.com/Danny-Taehyun-Kim/PySCF-Sigmaprofile-COSMO-SAC

See `src/generation/` in that repository, in particular `process_chunk.py` for
the PySCF driver and the radii table, and `smiles_to_sigma.py` for the route
from a SMILES string. Match those settings when preparing new surfaces, or the
inputs will not be on the same footing as the pre-training corpus.

## What is not included

No property prediction model trained on KDB labels is distributed here. The
released weights are the pre-trained encoders and the encoder-decoder pair from
the masked-reconstruction objective; none of them has seen a property label.
Fine-tuned per-property models are not released because they are derived from
KDB values, which are subject to the KDB terms of use. Reproduce them with
`scripts/05_finetune.py` once you hold the labels.

## Reproducing the paper

Pre-training was run once per encoder on a single A100.

The fine-tuned numbers in the paper come from a single run per fold. Repeating
a run with a different seed moves the error by a few per cent, so a rerun is
expected to land near the published value rather than on it.

Compute is marked **measured** when it comes from a job log and **estimated**
otherwise.

| Step | Command | Config | Result | Compute |
|------|---------|--------|--------|---------|
| Patch corpus | `scripts/01_build_patches.py` | fixed in `cosmo3dformer.patches` | 362,303 molecules, 192-patch padding | estimated, a few CPU hours on tens of cores |
| Pre-train S | `scripts/02_pretrain.py` | `configs/pretrain_S.yaml` | validation loss 0.4472 at epoch 147 | measured, 233 min on 1 A100 |
| Pre-train M | `scripts/02_pretrain.py` | `configs/pretrain_M.yaml` | validation loss 0.4136 at epoch 146 | measured, 235 min on 1 A100 |
| Pre-train L | `scripts/02_pretrain.py` | `configs/pretrain_L.yaml` | validation loss 0.4109 at epoch 150 | measured, 243 min on 1 A100 |
| Pre-train M without distance bias | `scripts/02_pretrain.py` | `configs/pretrain_M_no_bias.yaml` | validation loss 0.5566 at epoch 146 | measured, 235 min on 1 A100 |
| Figure 2b, surface reconstruction | `scripts/06_reconstruction_eval.py --target-stats configs/pretrain_target_stats.json` then `scripts/07_make_figures.py --recon` | 24 masks at ratio 0.9 | sigma R2 0.790, potential R2 0.926 on 774,179 patches | measured, 37 min on 1 A100 |
| Table 1, full-data errors | `scripts/05_finetune.py` per property and fold | `configs/finetune.yaml` | 45 fold-property runs per model | estimated, tens of GPU hours for all nine properties |
| Figure 3, limited-label curves | `scripts/05_finetune.py --train-cids ... --val-cids ...` over the subsets in `data/splits`, then `scripts/07_make_figures.py --curves` | `configs/finetune.yaml` | 15 runs per property and budget | estimated, hundreds of GPU hours for all models and budgets |
| Figure 4, representation comparison | `scripts/03_embed.py`, `scripts/04_ridge_eval.py`, then `scripts/07_make_figures.py --ridge` | `configs/ridge.yaml` | frozen ridge, no GPU needed after embedding | measured, 3 min on 1 A100 for the embeddings plus 2 min of CPU ridge |

Property labels are not distributed with this repository; see `data/README.md`.

## Data and pretrained weights

**Surfaces.** The COSMO surface corpus is the PubChemQC COSMO sigma-profile
database, Zenodo DOI 10.5281/zenodo.20672192.

**Property labels.** Pure-component values come from the Korea Thermophysical
Properties Data Bank and are subject to the KDB terms of use, so this
repository ships identifiers, folds and flags only.

**Weights.** Pre-trained weights are attached to the GitHub release. Fetch and
verify them with

    python scripts/download_weights.py --out-dir weights

The expected digests are listed in `scripts/weights_manifest.json` and repeated
here.

| File | Size | SHA-256 |
|------|-----:|---------|
| `L_encoder_ep150.pt` | 182.2 MB | `c631c3745d653c690abb51214df35d54ceaa2ea528e7f2b3f7029aadca450580` |
| `M_encoder_ep146.pt` | 88.8 MB | `4daa91e8090d931454e2ae530e650eee80773d8e8fd166b9d1f5101442c1be48` |
| `M_full_ep150.pt` | 90.7 MB | `9933440dbeeb21d39658e1440c5ed168631bee4bf7356211446ee38dd3f46d0d` |
| `M_no_bias_encoder_ep146.pt` | 88.8 MB | `7799f695e926f4fa84fa8c5423a341380d4e586f8a251f4daf9136e86dcad241` |
| `M_untrained_seed1234.pt` | 88.8 MB | `1035aed512bb313855bb67db3dc1acef7c51dfca205a72fc62da68a71e5bbe7e` |
| `M_untrained_seed1235.pt` | 88.8 MB | `175fba4ca481ec8b2488806c7b968e0d59202b2f681dd937687c018d4b53cabf` |
| `M_untrained_seed1236.pt` | 88.8 MB | `7f89945af3da0f42c309e41b10082567bc20947ec85a20bde34ca31bcc32733a` |
| `S_encoder_ep147.pt` | 25.4 MB | `3a3647c29c07ebab2bfccc407fd2cfef2b236116737dbbf3fbf302f286c283ce` |

`M_full_ep150.pt` holds the encoder and the decoder and is the file the
reconstruction example needs. The other files hold an encoder only. Optimiser
state is not included.

## Citation

Kim, T. H., Davis, K., Cha, Y. & Kwon, J. S.-I. Thermodynamically informed
representation learning for data-efficient molecular property prediction.
Manuscript submitted (2026).

Surface database: PubChemQC COSMO sigma-profile database, DOI 10.5281/zenodo.20672192

## License

Released under the MIT License, see `LICENSE`. The surface database and the
property labels carry their own terms; see `data/README.md`.
