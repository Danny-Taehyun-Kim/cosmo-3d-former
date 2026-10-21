"""Masked-autoencoder pre-training on COSMO surface patches.

Targets are the raw sigma, the patch area and the solute potential, with the
solid angle as an auxiliary target. Target statistics are computed once over
the valid patches of the training molecules and are never refitted on
validation. Inputs are passed to the encoder unstandardised.

The optimiser is AdamW with betas (0.9, 0.95), a linear warm-up followed by a
cosine decay, gradient-norm clipping, and bfloat16 autocast on CUDA.
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import h5py
import numpy as np
import torch

from .masking import block_mask, gather_masked_batch
from .model import PatchMAE, count_params

TARGET_COLS = [4, 3, 8]
AUX_COL = 9
CHANNELS = ("sigma", "area", "potential")


@dataclass
class PretrainConfig:
    """Pre-training settings. Field names match the keys in configs/*.yaml."""

    d_model: int = 480
    n_heads: int = 8
    layers: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    decoder_width: int = 128
    decoder_layers: int = 2
    decoder_heads: int = 4
    rbf_centres: int = 64
    rbf_cutoff: float = 16.0
    encoder_bias_scale: float = 1.0

    lr: float = 3e-4
    epochs: int = 150
    batch_size: int = 128
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    precision: str = "bfloat16"

    mask_ratio: float = 0.9
    block_radius: float = 2.0
    aux_weight: float = 0.5

    seed: int = 1234
    val_fraction: float = 0.02
    val_min: int = 64
    val_mask_seed: int = 777
    checkpoint_every: int = 10
    extra: dict = field(default_factory=dict)


def molecule_split(n_molecule, seed, val_fraction, val_min):
    """Deterministic molecule-level train and validation split."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_molecule, generator=generator)
    n_val = max(val_min, int(val_fraction * n_molecule))
    return order[n_val:], order[:n_val]


def load_corpus(h5_path, device, limit=0):
    """Load the padded patch corpus onto `device`.

    Returns coordinates, encoder features, targets, auxiliary target, the
    padding mask, the encoder input columns and the molecule identifiers.
    """
    with h5py.File(h5_path, "r") as handle:
        cols = [int(c) for c in handle.attrs["encoder_input_cols"]]
        n_total = handle["cid"].shape[0]
        n = min(n_total, limit) if limit else n_total
        cid = handle["cid"][:n]
        n_patch = torch.from_numpy(handle["n_patch"][:n].astype(np.int64))
        patches = torch.from_numpy(handle["patches"][:n])
    pad_len = patches.shape[1]
    pad = (torch.arange(pad_len)[None, :] < n_patch[:, None])
    xyz = patches[:, :, :3].to(device)
    features = patches[:, :, cols].to(device)
    targets = patches[:, :, TARGET_COLS].to(device)
    aux = patches[:, :, AUX_COL].to(device)
    del patches
    return xyz, features, targets, aux, pad.to(device), cols, cid


def target_statistics(targets, aux, pad, train_index):
    """Mean and standard deviation over valid patches of the training molecules."""
    valid = pad[train_index]
    mu = torch.stack([targets[train_index][..., k][valid].mean() for k in range(3)])
    sd = torch.stack([targets[train_index][..., k][valid].std() for k in range(3)])
    return mu, sd, aux[train_index][valid].mean(), aux[train_index][valid].std()


def build_model(config, n_in, device):
    """Instantiate the masked autoencoder described by `config`."""
    model = PatchMAE(n_in=n_in, d=config.d_model, L=config.layers, h=config.n_heads,
                     ff_mult=config.ff_mult, p=config.dropout, dd=config.decoder_width,
                     dL=config.decoder_layers, dh_=config.decoder_heads,
                     K=config.rbf_centres, cut=config.rbf_cutoff,
                     encoder_bias_scale=config.encoder_bias_scale)
    return model.to(device)


def make_batch(index, xyz, features, targets, aux, pad, config, mu, sd, amu, asd,
               generator):
    """Draw one masked batch and return model inputs and standardised targets."""
    coords = xyz[index]
    dist = torch.cdist(coords, coords)
    hidden, visible = block_mask(dist, pad[index], config.mask_ratio, generator=generator,
                                 block_radius=config.block_radius)
    fv, dvv, pv, dmv, pm, tg, ax, _, _ = gather_masked_batch(
        features[index], coords, pad[index], hidden, visible,
        targets=targets[index], aux=aux[index])
    return fv, dvv, pv, dmv, pm, (tg - mu) / sd, (ax - amu) / asd


def run(h5_path, config, out_dir, device="cuda", limit=0, max_steps=0):
    """Pre-train one encoder. Set `max_steps` for a short smoke run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    xyz, features, targets, aux, pad, cols, cid = load_corpus(h5_path, device, limit)
    train_index, val_index = molecule_split(len(cid), 42, config.val_fraction, config.val_min)
    mu, sd, amu, asd = target_statistics(targets, aux, pad, train_index)

    model = build_model(config, len(cols), device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimiser = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.lr, betas=tuple(config.betas))

    steps_per_epoch = max(1, len(train_index) // config.batch_size)
    total_steps = config.epochs * steps_per_epoch
    warmup = int(config.warmup_fraction * total_steps)

    def lr_at(step):
        if step < warmup:
            return config.lr * step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return config.lr * 0.5 * (1 + math.cos(math.pi * progress))

    use_autocast = device.startswith("cuda") and config.precision == "bfloat16"
    generator = torch.Generator(device=device).manual_seed(config.seed)
    history = []
    step = 0
    started = time.time()

    for epoch in range(config.epochs):
        model.train()
        order = train_index[torch.randperm(len(train_index))]
        for start in range(0, len(order) - config.batch_size + 1, config.batch_size):
            index = order[start:start + config.batch_size].to(device)
            for group in optimiser.param_groups:
                group["lr"] = lr_at(step)
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fv, dvv, pv, dmv, pm, tg, ax = make_batch(
                        index, xyz, features, targets, aux, pad, config, mu, sd, amu, asd,
                        generator)
                    recon, aux_pred, _ = model(fv, dvv, pv, dmv, pm)
                    weight = pm.float()
                    denom = weight.sum().clamp(min=1)
                    per_channel = [(((recon.float()[..., k] - tg[..., k]) ** 2) * weight).sum() / denom
                                   for k in range(3)]
                    loss_aux = (((aux_pred.float() - ax) ** 2) * weight).sum() / denom
                    loss = sum(per_channel) / 3 + config.aux_weight * loss_aux
            else:
                fv, dvv, pv, dmv, pm, tg, ax = make_batch(
                    index, xyz, features, targets, aux, pad, config, mu, sd, amu, asd,
                    generator)
                recon, aux_pred, _ = model(fv, dvv, pv, dmv, pm)
                weight = pm.float()
                denom = weight.sum().clamp(min=1)
                per_channel = [(((recon[..., k] - tg[..., k]) ** 2) * weight).sum() / denom
                               for k in range(3)]
                loss_aux = (((aux_pred - ax) ** 2) * weight).sum() / denom
                loss = sum(per_channel) / 3 + config.aux_weight * loss_aux

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            step += 1
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

        val = validate(model, xyz, features, targets, aux, pad, val_index, config,
                       mu, sd, amu, asd, device)
        history.append({"epoch": epoch, **val})
        if (epoch + 1) % config.checkpoint_every == 0 or epoch + 1 == config.epochs:
            save_encoder(out_dir / f"encoder_ep{epoch + 1}.pt", model, config, cols, epoch + 1, val)

    elapsed = time.time() - started
    save_encoder(out_dir / "encoder_last.pt", model, config, cols, step, history[-1] if history else {})
    save_full(out_dir / "full_last.pt", model, config, cols, step, history[-1] if history else {})
    (out_dir / "history.json").write_text(json.dumps(
        {"config": asdict(config), "cols": cols, "steps": step,
         "seconds": round(elapsed, 1), "history": history}, indent=1))
    return history


@torch.no_grad()
def validate(model, xyz, features, targets, aux, pad, val_index, config, mu, sd,
             amu, asd, device):
    """Masked reconstruction loss on the held-out molecules."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.val_mask_seed)
    channel = np.zeros(3)
    aux_total = 0.0
    n_batch = 0
    use_autocast = device.startswith("cuda") and config.precision == "bfloat16"
    for start in range(0, len(val_index), config.batch_size):
        index = val_index[start:start + config.batch_size].to(device)
        if len(index) < 4:
            continue
        context = (torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast
                   else contextlib.nullcontext())
        with context:
            fv, dvv, pv, dmv, pm, tg, ax = make_batch(
                index, xyz, features, targets, aux, pad, config, mu, sd, amu, asd, generator)
            recon, aux_pred, _ = model(fv, dvv, pv, dmv, pm)
            weight = pm.float()
            denom = weight.sum().clamp(min=1)
            squared = (recon.float() - tg) ** 2
            channel += np.array([float((squared[..., k] * weight).sum() / denom) for k in range(3)])
            aux_total += float((((aux_pred.float() - ax) ** 2) * weight).sum() / denom)
            n_batch += 1
    channel = channel / max(n_batch, 1)
    return {"val_msm": float(channel.mean()), "val_aux": aux_total / max(n_batch, 1),
            "val_ch": {name: float(channel[i]) for i, name in enumerate(CHANNELS)}}


def save_encoder(path, model, config, cols, epoch, val):
    """Store encoder weights and the settings needed to rebuild it."""
    torch.save({"encoder": model.enc.state_dict(), "n_in": len(cols), "cols": cols,
                "d_model": config.d_model, "n_heads": config.n_heads,
                "encoder_bias_scale": config.encoder_bias_scale,
                "epoch": epoch, "val": val, "config": asdict(config)}, path)


def save_full(path, model, config, cols, epoch, val):
    """Store encoder and decoder weights together with the settings."""
    torch.save({"model": model.state_dict(), "n_in": len(cols), "cols": cols,
                "d_model": config.d_model, "n_heads": config.n_heads,
                "encoder_bias_scale": config.encoder_bias_scale,
                "epoch": epoch, "val": val, "config": asdict(config)}, path)


def parameter_counts(config, n_in=12):
    """Encoder and decoder parameter counts for a configuration."""
    model = PatchMAE(n_in=n_in, d=config.d_model, L=config.layers, h=config.n_heads,
                     ff_mult=config.ff_mult, p=config.dropout, dd=config.decoder_width,
                     dL=config.decoder_layers, dh_=config.decoder_heads,
                     K=config.rbf_centres, cut=config.rbf_cutoff,
                     encoder_bias_scale=config.encoder_bias_scale)
    encoder = count_params(model.enc)
    return {"encoder": encoder, "decoder": count_params(model) - encoder,
            "total": count_params(model), "embedding_dim": 2 * config.d_model}
