"""Two-stage fine-tuning of a pre-trained encoder for a scalar property.

Stage one trains the regression head alone on cached embeddings. Stage two
unfreezes the encoder and trains both parts with separate learning rates, a
linear warm-up followed by a cosine decay, Huber loss and gradient-norm
clipping. The head is selected on validation loss.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .embed import mean_sum
from .model import PatchEncoder
from .splits import Target


class PropertyHead(nn.Module):
    """Regression head over the concatenated masked mean and sum.

    The layer norm is applied to the concatenated 2*d vector before the MLP.
    """

    def __init__(self, d, hidden=256, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(2 * d)
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def from_meansum(self, mean, summed):
        return self.net(self.ln(torch.cat([mean, summed], -1))).squeeze(-1)

    def forward(self, tokens, pooled, mask):
        summed = (tokens * mask.unsqueeze(-1).float()).sum(1)
        return self.from_meansum(pooled, summed)


@dataclass
class FinetuneConfig:
    """Fine-tuning settings. Field names match the keys in configs/finetune.yaml."""

    head_hidden: int = 256
    dropout: float = 0.1

    stage1_lr: float = 1e-3
    stage1_weight_decay: float = 1e-4
    stage1_epochs: int = 80
    stage1_patience: int = 10
    stage1_batch_size: int = 256

    seed: int = 0
    encoder_lr: float = 3e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 40
    warmup_epochs: int = 5
    batch_size: int = 96
    grad_clip: float = 1.0
    head_seed: int = 0
    precision: str = "bfloat16"


def train_head(head, mean_cache, sum_cache, z, train_idx, val_idx, config, device,
               shuffle=None):
    """Stage one. Train the head on cached embeddings and keep the best state."""
    optimiser = torch.optim.AdamW(head.parameters(), lr=config.stage1_lr,
                                  weight_decay=config.stage1_weight_decay)
    huber = nn.SmoothL1Loss()
    best, best_state, bad = np.inf, None, 0
    for _ in range(config.stage1_epochs):
        head.train()
        perm = (torch.randperm(len(train_idx), generator=shuffle) if shuffle is not None
                else torch.randperm(len(train_idx)))
        order = train_idx[perm.to(train_idx.device)]
        for start in range(0, len(order), config.stage1_batch_size):
            batch = order[start:start + config.stage1_batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = huber(head.from_meansum(mean_cache[batch], sum_cache[batch]), z[batch])
            loss.backward()
            optimiser.step()
        head.eval()
        with torch.no_grad():
            val = huber(head.from_meansum(mean_cache[val_idx], sum_cache[val_idx]),
                        z[val_idx]).item()
        if val < best - 1e-4:
            best, best_state, bad = val, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= config.stage1_patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, best


def forward_batch(encoder, head, features, xyz, pad, index):
    """Encode a batch and pool inside the graph so gradients reach the encoder."""
    f = features[index]
    x = xyz[index]
    p = pad[index]
    tokens = encoder(f, torch.cdist(x, x), p)
    mask = p.unsqueeze(-1).to(tokens.dtype)
    pooled = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return head(tokens, pooled, p)


def finetune(encoder_state, features, xyz, pad, y_train, y_val, train_idx, val_idx,
             test_idx, transform, config, device="cuda", fold=0):
    """Run both stages for one fold and return test predictions in original units.

    The random state is reset at the start of the fold and batch shuffling uses a
    generator seeded from the fold index, so a run is reproducible and does not
    depend on what ran before it.
    """
    torch.manual_seed(config.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    shuffle = torch.Generator()
    shuffle.manual_seed(config.seed + 1000 * int(fold))
    target = Target(transform).fit(y_train)
    z = torch.zeros(features.shape[0], device=device)
    z[train_idx] = torch.tensor(target.forward(y_train).astype("float32"), device=device)
    z[val_idx] = torch.tensor(target.forward(y_val).astype("float32"), device=device)

    encoder = PatchEncoder(n_in=int(encoder_state["n_in"]),
                           d=int(encoder_state.get("d_model", 256)),
                           h=int(encoder_state.get("n_heads", 8)),
                           encoder_bias_scale=float(encoder_state.get("encoder_bias_scale", 1.0)))
    encoder.load_state_dict(encoder_state["encoder"])
    encoder = encoder.to(device)

    use_autocast = device.startswith("cuda") and config.precision == "bfloat16"
    with torch.no_grad():
        encoder.eval()
        means, sums = [], []
        for start in range(0, features.shape[0], config.batch_size):
            stop = start + config.batch_size
            f, x, p = features[start:stop], xyz[start:stop], pad[start:stop]
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = encoder(f, torch.cdist(x, x), p).float()
            else:
                tokens = encoder(f, torch.cdist(x, x), p).float()
            pooled = mean_sum(tokens, p)
            means.append(pooled[:, :tokens.shape[-1]])
            sums.append(pooled[:, tokens.shape[-1]:])
        mean_cache = torch.cat(means)
        sum_cache = torch.cat(sums)

    torch.manual_seed(config.head_seed)
    head = PropertyHead(encoder.d, hidden=config.head_hidden, dropout=config.dropout).to(device)
    head, _ = train_head(head, mean_cache, sum_cache, z, train_idx, val_idx, config,
                         device, shuffle=shuffle)

    for param in encoder.parameters():
        param.requires_grad_(True)
    optimiser = torch.optim.AdamW(
        [{"params": encoder.parameters(), "lr": config.encoder_lr},
         {"params": head.parameters(), "lr": config.head_lr}],
        weight_decay=config.weight_decay)
    huber = nn.SmoothL1Loss()
    clip = list(encoder.parameters()) + list(head.parameters())

    def lr_factor(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / max(1, config.warmup_epochs)
        progress = min(1.0, (epoch - config.warmup_epochs) /
                       max(1, config.epochs - config.warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_factor)

    def evaluate(index):
        encoder.eval()
        head.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(index), config.batch_size):
                batch = index[start:start + config.batch_size]
                if use_autocast:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out.append(forward_batch(encoder, head, features, xyz, pad, batch).float())
                else:
                    out.append(forward_batch(encoder, head, features, xyz, pad, batch))
        return torch.cat(out)

    best, best_encoder, best_head, bad = np.inf, None, None, 0
    for epoch in range(config.epochs):
        encoder.train()
        head.train()
        order = train_idx[torch.randperm(len(train_idx), generator=shuffle).to(train_idx.device)]
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = huber(forward_batch(encoder, head, features, xyz, pad, batch), z[batch])
            else:
                loss = huber(forward_batch(encoder, head, features, xyz, pad, batch), z[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip, config.grad_clip)
            optimiser.step()
        val = huber(evaluate(val_idx), z[val_idx]).item()
        if val < best - 1e-4:
            best, bad = val, 0
            best_encoder = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            best_head = copy.deepcopy(head.state_dict())
        else:
            bad += 1
            if bad >= config.patience:
                scheduler.step()
                break
        scheduler.step()

    if best_encoder is not None:
        encoder.load_state_dict({k: v.to(device) for k, v in best_encoder.items()})
        head.load_state_dict(best_head)
    return target.inverse(evaluate(test_idx).cpu().numpy()), float(best)
