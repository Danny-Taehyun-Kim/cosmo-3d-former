"""Frozen embedding extraction.

A molecule is represented by the masked mean and the masked sum of the encoder
token outputs, concatenated. For an encoder of width d this gives a vector of
length 2*d: 512 for S, 960 for M and 1376 for L.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import PatchEncoder


def load_encoder(checkpoint_path, device="cpu", strict=True):
    """Load a stored encoder. Returns the module and the checkpoint dictionary."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = PatchEncoder(n_in=int(state["n_in"]),
                           d=int(state.get("d_model", 256)),
                           h=int(state.get("n_heads", 8)),
                           encoder_bias_scale=float(state.get("encoder_bias_scale", 1.0)))
    encoder.load_state_dict(state["encoder"], strict=strict)
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder.to(device).eval(), state


def mean_sum(tokens, pad):
    """Concatenate the masked mean and masked sum of token outputs."""
    mask = pad.unsqueeze(-1).to(tokens.dtype)
    summed = (tokens * mask).sum(1)
    count = mask.sum(1).clamp(min=1.0)
    return torch.cat([summed / count, summed], -1)


@torch.no_grad()
def embed_patches(encoder, features, xyz, pad, batch_size=64, device="cpu",
                  autocast_dtype=torch.bfloat16):
    """Embed a stack of padded patch tensors.

    features (N, P, n_in), xyz (N, P, 3) and pad (N, P) must already live on
    `device`. Autocast is used on CUDA only, matching how the stored embeddings
    were produced.
    """
    use_autocast = device.startswith("cuda") and autocast_dtype is not None
    out = []
    for start in range(0, features.shape[0], batch_size):
        stop = start + batch_size
        f = features[start:stop]
        x = xyz[start:stop]
        p = pad[start:stop]
        if use_autocast:
            with torch.autocast("cuda", dtype=autocast_dtype):
                tokens = encoder(f, torch.cdist(x, x), p)
        else:
            tokens = encoder(f, torch.cdist(x, x), p)
        out.append(mean_sum(tokens.float(), p).cpu().numpy())
    return np.concatenate(out)
