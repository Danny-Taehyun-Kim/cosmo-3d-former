# Baselines

Four published models are compared against COSMO-3D-Former. Their code is not
included here. Uni-Mol and Uni-Mol2 are run through the wrapper in this
directory. CrystalFormer and SigmaFormer were trained with their original
training code, with the data splitting and target scaling replaced by the
splits and target transformations of this repository.

| Model | Input | Pretraining data | Code |
|-------|-------|------------------|------|
| Uni-Mol | 3D conformer | 209 million conformations | `unimol_tools`, https://github.com/deepmodeling/Uni-Mol |
| Uni-Mol2 | 3D conformer | 800 million conformations | `unimol_tools` with the 1.1B checkpoint, https://github.com/deepmodeling/Uni-Mol |
| CrystalFormer | SMILES | 1.8 billion SMILES | https://github.com/Danny-Taehyun-Kim/SigmaFormer, directory `CrystalFormer/` (Pahari et al., AIChE J. 71, e18923, 2025) |
| SigmaFormer | SMILES and σ53 | 1.8 billion SMILES | https://github.com/Danny-Taehyun-Kim/SigmaFormer (Kim et al., AIChE J. 72, e70450, 2026) |

All four use the same folds, the same validation split and the same limited-data
subsets as COSMO-3D-Former. Targets go through the per-property transform in
`configs/ridge.yaml`.

## Uni-Mol

Uni-Mol is reached through `unimol_tools`, not through `unicore`. The wrapper
`unimol_finetune.py` mirrors the published protocol:

    python baselines/unimol_finetune.py --conformers conformers.pkl \
        --labels labels.csv --property Tb_K --fold 0 --out um_Tb_f0.csv \
        [--weight-dir DIR]

The pre-trained weights are not bundled. On first use `unimol_tools` downloads
`mol_pre_all_h_220816.pt` (190.5 MB) and `mol.dict.txt` from the Hugging Face
repository `dptech/Uni-Mol-Models` into its weight directory, which defaults to
`<site-packages>/unimol_tools/weights` and is overridden by `--weight-dir` or
the `UNIMOL_WEIGHT_DIR` environment variable. Compute nodes often have no
outbound network access, so populate that directory from a node that does
before submitting. `mol_pre_all_h_220816.pt` is the all-hydrogen model and the
wrapper passes `remove_hs=False` to match; the encoder holds 47,331,139
parameters.

Uni-Mol2 uses the same entry points with `model_name="unimolv2"`,
`model_size="1.1B"` and an explicit `pretrained_model_path`. That checkpoint
(4.52 GB, repository `dptech/Uni-Mol2`, path `modelzoo/1.1B/checkpoint.pt`) is
a separate download and holds 1,126,650,593 parameters, returning a 1536-wide
representation. Its featuriser is `UniMolV2Feature.transform_raw(atoms,
coords)`, which requires Python floats rather than a NumPy array, and its
fine-tuning uses a micro-batch of 8 with 8 accumulation steps, giving the same
effective batch as Uni-Mol v1.

## CrystalFormer and SigmaFormer

Both models were fine-tuned with their original training code. Only the
internal data splitting and target scaling were replaced, by the folds, the
validation split, the limited-data subsets and the per-property target
transforms of this repository. SigmaFormer receives σ53, the 51-bin
sigma-profile with the total surface area and cavity volume, as an additional
input. The learning rate starts at 3e-5 rather than the 3e-4 default of the
public code. The first ten epochs train the regression head only. At epoch 10
the encoder and token embedding are unfrozen and the learning rate is lowered
to 1e-5. The other settings are listed below.

The pre-trained encoder weights are available on request from the corresponding
author of the SigmaFormer paper; they are not part of either repository.

## Conformers

Uni-Mol and Uni-Mol2 take 3D conformers. `cache_conformers.py` builds them once
so that every fold and every budget sees the same geometry:

    python baselines/cache_conformers.py --smiles smiles.csv --out conformers.pkl

The recipe is RDKit ETKDGv3 with random seed 42, ten embedded conformers, MMFF
optimisation, and the lowest-energy conformer retained. If embedding yields no
conformer the attempt is repeated with random coordinates.

## Settings

| Setting | Uni-Mol, Uni-Mol2 | CrystalFormer, SigmaFormer |
|---------|-------------------|----------------------------|
| optimiser | AdamW | AdamW, betas (0.9, 0.99), weight decay 0 |
| learning rate | see `envs/unimol.yaml` | 3e-5, lowered once to 1e-5 at epoch 10 |
| batch size | see `envs/unimol.yaml` | 64 |
| loss | Huber | SmoothL1 |
| epochs | see `envs/unimol.yaml` | up to 500 |
| early stopping | on validation loss | patience 50 on validation loss |
| gradient clipping | 1.0 | 1.0 |
