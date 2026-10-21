#!/usr/bin/env python
"""Build the figures of the paper from stored results.

Usage:
    07_make_figures.py --recon recon.npz --out-dir figures            # Figure 2b
    07_make_figures.py --curves curves.csv --out-dir figures          # Figure 3
    07_make_figures.py --ridge ridge_summary.csv --out-dir figures    # Figure 4

Any combination may be given. Figure 2b is the reconstruction parity plot and
the per-method statistics; a patch is scored only where it was hidden at least
once, and patches with zero coverage stay unpredicted. Figure 3 is the
limited-label learning curve, one panel per property, from a table with the
columns property, budget, model and RMSE_median. Figure 4 is the frozen
representation comparison, from a table with the columns property,
representation and RMSE_median.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

UNITS = {"sigma": "e/A^2", "area": "A^2", "potential": "e/A", "aux": "solid angle"}


def statistics(reference, prediction):
    """Error statistics in the units of `reference`."""
    error = prediction - reference
    sse = float((error ** 2).sum())
    sst = float(((reference - reference.mean()) ** 2).sum())
    return {"n": int(reference.size), "RMSE": float(np.sqrt(sse / reference.size)),
            "R2": 1 - sse / sst, "sd_reference": float(reference.std()),
            "sd_prediction": float(prediction.std()),
            "sd_ratio": float(prediction.std() / reference.std()),
            "sign_agreement": float((np.sign(reference) == np.sign(prediction)).mean())}


def _plt():
    """Matplotlib with the font of the paper. Falls back to a metric-compatible clone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cosmo3dformer.parity_plot import use_font
    use_font()
    plt.rcParams.update({"font.size": 8, "xtick.direction": "in", "ytick.direction": "in",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


def figure_three(path, out_dir, no_plot=False):
    """Limited-label learning curves, one panel per property."""
    table = pd.read_csv(path)
    table.to_csv(out_dir / "figure3_source.csv", index=False)
    print(f"Figure 3 source: {len(table)} rows, {table.property.nunique()} properties", flush=True)
    if no_plot:
        return
    plt = _plt()
    properties = sorted(table.property.unique())
    rows = int(np.ceil(len(properties) / 3))
    figure, axes = plt.subplots(rows, 3, figsize=(9.5, 3.0 * rows), squeeze=False)
    for i, prop in enumerate(properties):
        ax = axes[i // 3][i % 3]
        sub = table[table.property == prop]
        for model in sorted(sub.model.unique()):
            g = sub[sub.model == model].copy()
            curve = g[g.budget.astype(str) != "Full"].copy()
            full = g[g.budget.astype(str) == "Full"]
            line = None
            if len(curve):
                curve["budget"] = curve.budget.astype(int)
                curve = curve.sort_values("budget")
                line, = ax.plot(curve.budget, curve.RMSE_median, marker="o", ms=3, lw=1.0,
                                label=model)
            if len(full):
                # the full-data endpoint is drawn as an open marker at its own label count
                x = full.n_labels.iloc[0] if "n_labels" in full.columns else (
                    curve.budget.max() * 2 if len(curve) else 1)
                ax.plot([x], [full.RMSE_median.iloc[0]], marker="o", ms=5, ls="none",
                        mfc="none", mec=(line.get_color() if line is not None else "0.3"),
                        mew=1.2)
        ax.set_xscale("log")
        ax.set_title(prop, fontsize=8)
        ax.set_xlabel("labels (open marker: full data)")
        ax.set_ylabel("median RMSE")
    for j in range(len(properties), rows * 3):
        axes[j // 3][j % 3].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7)
    figure.tight_layout(rect=[0, 0.04, 1, 1])
    figure.savefig(out_dir / "figure3_limited_label.png", dpi=300)
    plt.close(figure)
    print(f"wrote {out_dir / 'figure3_limited_label.png'}", flush=True)


REFERENCE_REPRESENTATION = "sigma53"


def figure_four(path, out_dir, matched=None, no_plot=False):
    """Frozen representation comparison expressed against the sigma-profile descriptor.

    Panel a uses the full-data result: median RMSE of each representation divided
    by the median RMSE of `sigma53` on the same property, so a value below one
    means lower error than the descriptor. Panel b repeats the ratio at matched
    label budgets when a matched-N table is supplied.
    """
    table = pd.read_csv(path)
    wide = table.pivot(index="property", columns="representation", values="RMSE_median")
    if REFERENCE_REPRESENTATION not in wide.columns:
        raise SystemExit(f"{REFERENCE_REPRESENTATION} is missing from {path}")
    ratio = wide.div(wide[REFERENCE_REPRESENTATION], axis=0)
    ratio.to_csv(out_dir / "figure4a_source.csv")
    print(f"Figure 4a: ratio to {REFERENCE_REPRESENTATION}, "
          f"{len(ratio)} properties by {ratio.shape[1]} representations", flush=True)
    matched_table = None
    if matched:
        matched_table = pd.read_csv(matched)
        matched_table.to_csv(out_dir / "figure4b_source.csv", index=False)
        print(f"Figure 4b: {len(matched_table)} matched-budget rows", flush=True)
    if no_plot:
        return
    plt = _plt()
    ncol = 2 if matched_table is not None else 1
    figure, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 3.4), squeeze=False)

    ax = axes[0][0]
    names = [c for c in ratio.columns if c != REFERENCE_REPRESENTATION]
    width = 0.8 / max(len(names), 1)
    x = np.arange(len(ratio))
    for i, name in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * width, ratio[name], width, label=name)
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ratio.index, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(f"median RMSE relative to {REFERENCE_REPRESENTATION}")
    ax.set_title("a", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    if matched_table is not None:
        ax = axes[0][1]
        for prop in sorted(matched_table.property.unique()):
            g = matched_table[matched_table.property == prop].copy()
            g = g[g.N.astype(str) != "Full"]
            if not len(g):
                continue
            g["N"] = g.N.astype(int)
            g = g.sort_values("N")
            ax.plot(g.N, g.ratio_to_sigma53, marker="o", ms=3, lw=1.0, label=prop)
        ax.axhline(1.0, color="0.3", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("labels")
        ax.set_ylabel(f"RMSE relative to {REFERENCE_REPRESENTATION}")
        ax.set_title("b", loc="left", fontsize=9)
        ax.legend(fontsize=6, frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(out_dir / "figure4_representations.png", dpi=300)
    plt.close(figure)
    print(f"wrote {out_dir / 'figure4_representations.png'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon", default=None, help="output of 06_reconstruction_eval.py")
    parser.add_argument("--curves", default=None, help="limited-label table for Figure 3")
    parser.add_argument("--ridge", default=None, help="frozen representation table for Figure 4a")
    parser.add_argument("--matched-n", default=None,
                        help="matched-budget table for Figure 4b, with N and ratio_to_sigma53")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if not (args.recon or args.curves or args.ridge):
        raise SystemExit("give at least one of --recon, --curves, --ridge")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.curves:
        figure_three(args.curves, out_dir, args.no_plot)
    if args.ridge:
        figure_four(args.ridge, out_dir, matched=args.matched_n, no_plot=args.no_plot)
    if not args.recon:
        return
    data = np.load(args.recon, allow_pickle=False)
    methods = [str(m) for m in data["methods"]]
    channels = [str(c) for c in data["channels"]]
    sums, coverage = data["sums"], data["coverage"]
    pad, hidden_first = data["pad"].astype(bool), data["hidden_first"].astype(bool)
    mu, sd = data["mu"], data["sd"]
    targets = data["targets_std"]

    scored = hidden_first & pad & (coverage > 0)
    rows = []
    for mi, method in enumerate(methods):
        for ci, channel in enumerate(channels):
            prediction = np.divide(sums[mi, ci], np.maximum(coverage, 1), dtype=np.float64)
            prediction = prediction * sd[ci] + mu[ci]
            reference = targets[:, :, ci].astype(np.float64) * sd[ci] + mu[ci]
            if not np.isfinite(prediction[scored]).all():
                rows.append({"method": method, "channel": channel, "unit": UNITS[channel],
                             "note": "not given to this baseline"})
                continue
            rows.append({"method": method, "channel": channel, "unit": UNITS[channel],
                         **statistics(reference[scored], prediction[scored]), "note": ""})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "reconstruction_statistics.csv", index=False)
    print(frame.to_string(index=False), flush=True)
    print(f"unpredicted patches (coverage zero): {int((pad & (coverage == 0)).sum())}", flush=True)

    if args.no_plot:
        return
    plt = _plt()

    model_index = methods.index("model")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, channel, scale in ((axes[0], "sigma", 1e3), (axes[1], "potential", 1.0)):
        ci = channels.index(channel)
        prediction = np.divide(sums[model_index, ci], np.maximum(coverage, 1), dtype=np.float64)
        prediction = (prediction * sd[ci] + mu[ci])[scored] * scale
        reference = (targets[:, :, ci].astype(np.float64) * sd[ci] + mu[ci])[scored] * scale
        ax.hexbin(reference, prediction, gridsize=60, bins="log", mincnt=1)
        low = float(min(reference.min(), prediction.min()))
        high = float(max(reference.max(), prediction.max()))
        ax.plot([low, high], [low, high], "r-", lw=0.8)
        ax.set_aspect("equal")
        label = f"{channel} ({'1e-3 ' if scale != 1 else ''}{UNITS[channel]})"
        ax.set_xlabel(f"calculated {label}")
        ax.set_ylabel(f"reconstructed {label}")
    figure.tight_layout()
    figure.savefig(out_dir / "figure2b_reconstruction_parity.png", dpi=300)
    plt.close(figure)
    print(f"wrote {out_dir / 'figure2b_reconstruction_parity.png'}", flush=True)


if __name__ == "__main__":
    main()
