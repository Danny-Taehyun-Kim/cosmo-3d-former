"""Patch encoder and masked autoencoder for COSMO surfaces.

The encoder takes rotation-invariant inputs only: per-patch features, the
patch-centre distance matrix and a padding mask. Coordinates are never passed
in, so rotation invariance holds by construction.

The decoder lets masked queries attend to visible patches only. Masked patches
do not attend to each other, and the query is a copy of a single learned mask
token, so no hidden feature can reach the decoder.
"""
import math

import torch
import torch.nn as nn

class RBF(nn.Module):
    """Distance to K fixed Gaussian radial basis features. No parameters."""
    def __init__(self, K=64, cut=16.0):
        super().__init__()
        self.register_buffer("mu", torch.linspace(0.0, cut, K))
        self.sig = cut / K * 2.0
    def forward(self, d):                       # (...,) -> (...,K)
        return torch.exp(-((d.unsqueeze(-1) - self.mu) / self.sig) ** 2)

class Block(nn.Module):
    """Pre-LN self-attention with feed-forward. 12*d^2 + 13*d parameters."""
    def __init__(self, d, h, ff_mult, p):
        super().__init__()
        self.h, self.dh = h, d // h
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                nn.Dropout(p), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(p)
    def forward(self, x, bias, pad, return_attn=False):   # x (B,P,d), bias (B,h,P,P), pad (B,P) with True = valid
        B, P, D = x.shape
        y = self.n1(x)
        q, k, v = self.qkv(y).chunk(3, -1)
        q = q.view(B, P, self.h, self.dh).transpose(1, 2)
        k = k.view(B, P, self.h, self.dh).transpose(1, 2)
        v = v.view(B, P, self.h, self.dh).transpose(1, 2)
        a = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        content = a if return_attn else None      # q.k/sqrt(d_h) before the bias is added
        if bias is not None: a = a + bias
        a = a.masked_fill(~pad[:, None, None, :], float("-inf"))
        a = self.drop(torch.softmax(a, -1))
        o = (a @ v).transpose(1, 2).reshape(B, P, D)
        x = x + self.drop(self.proj(o))
        out = x + self.ff(self.n2(x))
        if return_attn:
            return out, dict(attn=a, content=content,
                             bias=(bias if bias is not None else torch.zeros_like(content)))
        return out

class CrossBlock(nn.Module):
    """Decoder block: masked queries attend to visible keys. 12*d^2 + 13*d parameters."""
    def __init__(self, d, h, ff_mult, p):
        super().__init__()
        self.h, self.dh = h, d // h
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                nn.Dropout(p), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(p)
    def forward(self, x, ctx, bias, cpad):      # x (B,Pm,d) ctx (B,Pv,d) bias (B,h,Pm,Pv)
        B, Pm, D = x.shape; Pv = ctx.shape[1]
        q = self.q(self.nq(x)).view(B, Pm, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(self.nk(ctx)).chunk(2, -1)
        k = k.view(B, Pv, self.h, self.dh).transpose(1, 2)
        v = v.view(B, Pv, self.h, self.dh).transpose(1, 2)
        a = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if bias is not None: a = a + bias
        a = a.masked_fill(~cpad[:, None, None, :], float("-inf"))
        a = self.drop(torch.softmax(a, -1))
        o = (a @ v).transpose(1, 2).reshape(B, Pm, D)
        x = x + self.drop(self.proj(o))
        return x + self.ff(self.n2(x))

class PatchEncoder(nn.Module):
    """Encoder over visible patches.

    encoder_bias_scale multiplies the distance bias. It is a plain float, not a
    parameter, so it never enters the state dict and parameter counts are
    unchanged. The ablation encoder was trained with the scale at zero and must
    be loaded with the same value.
    """

    def __init__(self, n_in=12, d=256, L=8, h=8, ff_mult=4, p=0.1, K=64, cut=16.0,
                 encoder_bias_scale=1.0):
        super().__init__()
        self.d, self.L, self.h = d, L, h
        self.encoder_bias_scale = float(encoder_bias_scale)
        self.inp = nn.Linear(n_in, d)
        self.rbf = RBF(K, cut)
        self.bias_mlp = nn.Sequential(nn.Linear(K, 128), nn.GELU(), nn.Linear(128, L * h))
        self.blocks = nn.ModuleList([Block(d, h, ff_mult, p) for _ in range(L)])
        self.norm = nn.LayerNorm(d)
    def forward(self, feat, dist, pad, return_attn=False):
        B, P, _ = feat.shape
        bias = self.bias_mlp(self.rbf(dist))                    # (B,P,P,L*h)
        if self.encoder_bias_scale != 1.0:
            bias = bias * self.encoder_bias_scale
        bias = bias.view(B, P, P, self.L, self.h).permute(3, 0, 4, 1, 2)  # (L,B,h,P,P)
        x = self.inp(feat)
        if not return_attn:                                     # default path, unchanged
            for i, blk in enumerate(self.blocks): x = blk(x, bias[i], pad)
            return self.norm(x)                                 # (B,P,d)
        A = []
        for i, blk in enumerate(self.blocks):
            x, a = blk(x, bias[i], pad, return_attn=True); A.append(a)
        return self.norm(x), A                                  # (B,P,d), [8 × dict]

class PatchMAE(nn.Module):
    """Encoder over visible patches plus a distance-conditioned masked decoder."""
    def __init__(self, n_in=12, n_recon=3, d=256, L=8, h=8, ff_mult=4, p=0.1,
                 dd=128, dL=2, dh_=4, K=64, cut=16.0, encoder_bias_scale=1.0):
        super().__init__()
        self.enc = PatchEncoder(n_in, d, L, h, ff_mult, p, K, cut, encoder_bias_scale)
        self.to_dec = nn.Linear(d, dd)
        self.mask_tok = nn.Parameter(torch.zeros(dd)); nn.init.normal_(self.mask_tok, std=0.02)
        self.drbf = RBF(K, cut)
        self.dbias = nn.Sequential(nn.Linear(K, 128), nn.GELU(), nn.Linear(128, dL * dh_))
        self.dblocks = nn.ModuleList([CrossBlock(dd, dh_, ff_mult, p) for _ in range(dL)])
        self.dnorm = nn.LayerNorm(dd)
        self.recon = nn.Linear(dd, n_recon)     # sigma_raw, area, potential
        self.aux = nn.Linear(dd, 1)             # solid angle, one value per masked patch
        self.dL, self.dh_ = dL, dh_
    def forward(self, fv, dvv, pv, dmv, pm):
        """fv (B,Pv,n_in) visible features, dvv (B,Pv,Pv), pv (B,Pv),
           dmv (B,Pm,Pv) masked-to-visible distances, pm (B,Pm) valid masked slots."""
        Hv = self.enc(fv, dvv, pv)                              # (B,Pv,d)
        ctx = self.to_dec(Hv)
        B, Pm, Pv = dmv.shape
        b = self.dbias(self.drbf(dmv)).view(B, Pm, Pv, self.dL, self.dh_).permute(3, 0, 4, 1, 2)
        x = self.mask_tok.view(1, 1, -1).expand(B, Pm, -1)
        for i, blk in enumerate(self.dblocks): x = blk(x, ctx, b[i], pv)
        x = self.dnorm(x)
        return self.recon(x), self.aux(x).squeeze(-1), Hv



def count_params(module):
    """Total number of parameters in a module."""
    return sum(p.numel() for p in module.parameters())
