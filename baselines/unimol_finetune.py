#!/usr/bin/env python
"""Fine-tune Uni-Mol on one property and one fold.

Usage:
    unimol_finetune.py --conformers conformers.pkl --labels labels.csv \
        --property Tb_K --fold 0 --out um_Tb_f0.csv [--max-epochs N]

Stage one trains the head on cached representations, stage two unfreezes the
Uni-Mol encoder. The protocol matches the runs behind the paper: AdamW with
3e-5 on the encoder and 1e-4 on the head, weight decay 1e-4, five warm-up
epochs then a cosine decay, Huber loss, gradient-norm clipping at 1.0,
batch 64, at most 300 epochs with patience 40, bfloat16 autocast on CUDA.

Uni-Mol comes from `unimol_tools`; `unicore` is not used. On first use the
package downloads `mol_pre_all_h_220816.pt` and `mol.dict.txt` from Hugging
Face into its weight directory, which defaults to the installed package and is
overridden by `--weight-dir` (equivalently the `UNIMOL_WEIGHT_DIR` environment
variable). Compute nodes without network access need that directory populated
beforehand. `--max-epochs` bounds stage two for a short check.
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cosmo3dformer.finetune import PropertyHead
from cosmo3dformer.ridge import r2_score, rmse
from cosmo3dformer.splits import CLEAN2_PROPERTIES, TRANSFORM, Target, val_split

ENC_LR, HEAD_LR, EPOCHS, PATIENCE, WARMUP, BATCH, WD = 3e-5, 1e-4, 300, 40, 5, 64, 1e-4
STAGE1_EPOCHS, STAGE1_PATIENCE, STAGE1_LR = 80, 10, 1e-3


def load_model(device):
    """Uni-Mol with its pre-trained weights, plus the token dictionary."""
    from unimol_tools import UniMolRepr
    model = UniMolRepr(data_type="molecule", remove_hs=False).model
    return model.to(device), model.dictionary


def encode(model, cids, conformers, dictionary, device):
    """Representation of a batch of molecules, pooled by Uni-Mol itself."""
    from unimol_tools.data.conformer import coords2unimol
    items = [coords2unimol(conformers[c][0], conformers[c][1], dictionary,
                           remove_hs=False, data_type="molecule") for c in cids]
    batch, _ = model.batch_collate_fn([(d, 0.0) for d in items])
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        out = model(**batch, return_repr=True, return_atomic_reprs=False)
    return out.float()


def predict(model, head, ids, conformers, dictionary, device, batch=BATCH):
    """Predictions for a list of identifiers, one plain loop, nothing lazy."""
    model.eval()
    head.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            cls = encode(model, ids[start:start + batch], conformers, dictionary, device)
            out.append(head.from_meansum(cls[:, :256], cls[:, 256:]).float().cpu())
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conformers", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--property", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-epochs", type=int, default=EPOCHS)
    parser.add_argument("--clean2-column", default="clean2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weight-dir", default=None,
                        help="directory holding the pre-trained Uni-Mol weights; "
                             "sets UNIMOL_WEIGHT_DIR for unimol_tools")
    args = parser.parse_args()
    if args.weight_dir:
        os.environ["UNIMOL_WEIGHT_DIR"] = str(Path(args.weight_dir).resolve())

    device = args.device
    conformers = pickle.load(open(args.conformers, "rb"))
    table = pd.read_csv(args.labels)
    table["cid"] = table.cid.astype(int)
    table = table.dropna(subset=[args.property])
    table = table[table.cid.isin(conformers)]
    unfiltered_pool = table.cid[table.fold != args.fold]
    if args.property in CLEAN2_PROPERTIES and args.clean2_column in table.columns:
        table = table[table[args.clean2_column].astype(bool)]

    test = table[table.fold == args.fold]
    pool = table[table.fold != args.fold]
    val_cids = val_split(unfiltered_pool, args.fold)
    val = pool[pool.cid.isin(val_cids)]
    train = pool[~pool.cid.isin(val_cids)]
    print(f"train {len(train)} val {len(val)} test {len(test)}", flush=True)

    target = Target(TRANSFORM.get(args.property, "standard")).fit(
        train[args.property].to_numpy(float))
    z = {int(c): float(v) for c, v in zip(
        pd.concat([train, val]).cid,
        target.forward(pd.concat([train, val])[args.property].to_numpy(float)))}

    model, dictionary = load_model(device)
    torch.manual_seed(0)
    head = PropertyHead(256).to(device)
    huber = nn.SmoothL1Loss()

    # stage one: cache the representations once and train the head alone
    model.eval()
    cache = {}
    with torch.no_grad():
        for frame in (train, val, test):
            ids = list(frame.cid)
            for start in range(0, len(ids), BATCH):
                chunk = ids[start:start + BATCH]
                vecs = encode(model, chunk, conformers, dictionary, device)
                for cid, vec in zip(chunk, vecs):
                    cache[int(cid)] = vec.detach().clone()
    stack = lambda ids: torch.stack([cache[int(c)] for c in ids])
    opt1 = torch.optim.AdamW(head.parameters(), lr=STAGE1_LR, weight_decay=WD)
    best, best_h, bad = np.inf, None, 0
    zt = torch.tensor([z[int(c)] for c in train.cid], device=device)
    zv = torch.tensor([z[int(c)] for c in val.cid], device=device)
    xt, xv = stack(train.cid), stack(val.cid)
    for _ in range(STAGE1_EPOCHS):
        head.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), 256):
            idx = order[start:start + 256]
            opt1.zero_grad(set_to_none=True)
            huber(head.from_meansum(xt[idx, :256], xt[idx, 256:]), zt[idx]).backward()
            opt1.step()
        head.eval()
        with torch.no_grad():
            v = huber(head.from_meansum(xv[:, :256], xv[:, 256:]), zv).item()
        if v < best - 1e-4:
            best, best_h, bad = v, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= STAGE1_PATIENCE:
                break
    head.load_state_dict(best_h)
    with torch.no_grad():
        frozen = target.inverse(head.from_meansum(
            stack(test.cid)[:, :256], stack(test.cid)[:, 256:]).cpu().numpy())
    y_test = test[args.property].to_numpy(float)
    print(f"stage one done, frozen R2 {r2_score(y_test, frozen):.4f}", flush=True)
    del cache, xt, xv

    # stage two: unfreeze the encoder
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": ENC_LR},
                             {"params": head.parameters(), "lr": HEAD_LR}], weight_decay=WD)
    clip = list(model.parameters()) + list(head.parameters())

    def lr_factor(ep):
        if ep < WARMUP:
            return (ep + 1) / max(1, WARMUP)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (ep - WARMUP) / max(1, EPOCHS - WARMUP))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
    train_ids = list(train.cid)
    val_ids = list(val.cid)
    best, best_m, best_h, bad, best_ep = np.inf, None, None, 0, -1
    started = time.time()
    for ep in range(args.max_epochs):
        model.train()
        head.train()
        perm = list(np.random.default_rng(ep).permutation(len(train_ids)))
        for start in range(0, len(perm), BATCH):
            ids = [train_ids[j] for j in perm[start:start + BATCH]]
            opt.zero_grad(set_to_none=True)
            cls = encode(model, ids, conformers, dictionary, device)
            pred = head.from_meansum(cls[:, :256], cls[:, 256:])
            loss = huber(pred, torch.tensor([z[int(c)] for c in ids],
                                            dtype=torch.float32, device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip, 1.0)
            opt.step()
        model.eval()
        head.eval()
        vp = torch.from_numpy(predict(model, head, val_ids, conformers, dictionary, device))
        v = float(huber(vp, torch.tensor([z[int(c)] for c in val_ids], dtype=torch.float32)))
        if not np.isfinite(v):
            break
        if v < best - 1e-4:
            best, bad, best_ep = v, 0, ep
            best_m = copy.deepcopy(model.state_dict())
            best_h = copy.deepcopy(head.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        sched.step()
        print(f"  epoch {ep} val {v:.5f} best {best:.5f} at {best_ep} "
              f"({time.time() - started:.0f} s)", flush=True)

    if best_m is not None:
        model.load_state_dict(best_m)
        head.load_state_dict(best_h)
    del best_m, best_h
    opt.zero_grad(set_to_none=True)
    for p_ in model.parameters():
        p_.grad = None
        p_.requires_grad_(False)
    torch.cuda.empty_cache() if device == "cuda" else None
    pred = target.inverse(predict(model, head, list(test.cid), conformers, dictionary, device))

    frame = pd.DataFrame([{"model": "Uni-Mol", "property": args.property, "fold": args.fold,
                           "n_train": len(train), "n_val": len(val), "n_test": len(test),
                           "best_epoch": best_ep, "epochs_run": ep + 1,
                           "max_epochs": args.max_epochs,
                           "frozen_R2": r2_score(y_test, frozen),
                           "R2": r2_score(y_test, pred), "RMSE": rmse(y_test, pred),
                           "seconds": round(time.time() - started, 1)}])
    frame.to_csv(args.out, index=False)
    # Importing RDKit patches the pandas string formatter, and that patch can
    # crash inside the XML parser it uses. Print the row directly instead.
    row = frame.iloc[0]
    for key in frame.columns:
        value = row[key]
        print(f"  {key:12s} {value:.6g}" if isinstance(value, float) else f"  {key:12s} {value}",
              flush=True)
    return model, head


if __name__ == "__main__":
    model, head = main()
    # Release the Uni-Mol model before the interpreter tears down. Without this
    # the third-party extension can crash during shutdown, after the result has
    # already been written.
    del model, head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
