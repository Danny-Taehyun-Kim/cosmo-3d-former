"""Parity figures for the masked-patch reconstruction analysis.

This is the plotting code used for the figure in the paper, kept unchanged apart
from the module docstring.

Two functions, both fed with the per-patch arrays that the reconstruction analysis already saved
(one entry per evaluated patch, 774,179 in total, in the STORED units):

    screening-charge density  sigma      e/A^2
    solute potential                     e/A      (PySCF atomic units / 0.529177; NOT kcal/mol/e.
                                                   1 e/A = 14.3996 V = 332.06 kcal/mol/e)

    from fig2c_parity import make_fig2c, make_ed_parity

    # main-text Fig. 2c: 24-mask mean only, sigma | potential
    make_fig2c(sig_ref, sig_avg, pot_ref, pot_avg,
               sig_nn=sig_nn_avg, pot_nn=pot_nn_avg,          # nearest-patch estimate, 24-mask mean
               out="Fig2c_parity")

    # Extended Data: single mask | 24-mask mean, for sigma and potential
    make_ed_parity(sig_ref, sig_single, sig_avg, pot_ref, pot_single, pot_avg,
                   nn={"sig_single": sig_nn_single, "sig_avg": sig_nn_avg,
                       "pot_single": pot_nn_single, "pot_avg": pot_nn_avg},
                   out="ED_parity_single_vs_24mask")

A nearest-patch entry may be an array (statistics are computed here) or a tuple (rmse, r2) taken
from saved metrics, in the stored units. Leave it out and the row is omitted.

Each call writes <out>.pdf, <out>.png and <out>_stats.json and prints the numbers quoted in the text
(SD of the calculated values, RMSE and R2 of the model and of the nearest-patch estimate, RMSE
reduction, SD ratio, patches outside the plotted range).

Font: Times New Roman if installed, otherwise the first available metric-compatible clone; the
font actually used is printed. Change FONT_PREFERENCE to switch (e.g. ["Arial", "Helvetica"]).
"""
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, LogNorm

FONT_PREFERENCE = ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman",
                   "TeX Gyre Termes", "STIXGeneral"]
INK, MUTED = "#1F2933", "#52606D"      # text / secondary text
LINE = "#C46A2B"                       # identity line (orange of Fig. 4)
# one hue, light -> dark (darker = more patches); blue family of Fig. 4
CMAP = LinearSegmentedColormap.from_list("blue_seq", ["#D3E3EC", "#7DADC5", "#156082", "#0A2836"])
E_PER_A_TO_V = 14.3996
FS, FS_TICK, FS_TAB = 7.0, 6.5, 6.5    # points (Nature: 5-7 pt)


def use_font(preference=FONT_PREFERENCE):
    """Select the first installed font of `preference` and set the shared style."""
    chosen = None
    for name in preference:
        try:
            path = fm.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        chosen = (name, path)
        break
    if chosen is None:
        raise RuntimeError(f"none of these fonts is installed: {preference}")
    mpl.rcParams.update({
        "font.family": [chosen[0]], "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "mathtext.fontset": "stix", "mathtext.default": "regular",
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    print(f"[font] {chosen[0]}  ({chosen[1]})")
    return chosen


def _stats(ref, est):
    err = est - ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = float(1.0 - np.sum(err ** 2) / np.sum((ref - ref.mean()) ** 2))
    return rmse, r2


def _arr(a, scale):
    return None if a is None else np.asarray(a, dtype=np.float64).ravel() * scale


def _panel(ax, ref, pred, base, lim, ticks, qlabel, unit, unit_plain, fmt, gridsize, name,
           scale, xlabel=True, ylabel=True):
    """One parity panel. `ref`, `pred` are already in plotting units; `base` is the nearest-patch
    estimate as an array (stored units), a tuple (rmse, r2) in stored units, or None."""
    rmse, r2 = _stats(ref, pred)
    sd_ref, sd_pred = float(ref.std()), float(pred.std())
    inside = (ref >= lim[0]) & (ref <= lim[1]) & (pred >= lim[0]) & (pred <= lim[1])
    n_out = int((~inside).sum())

    hb = ax.hexbin(ref[inside], pred[inside], gridsize=gridsize, extent=(*lim, *lim),
                   mincnt=1, cmap=CMAP, linewidths=0, rasterized=True)   # raw counts; log scale set in _finish
    ax.plot(lim, lim, color=LINE, lw=0.8, solid_capstyle="butt", zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    if xlabel: ax.set_xlabel(f"Calculated {qlabel} ({unit})")
    if ylabel: ax.set_ylabel(f"Reconstructed {qlabel} ({unit})")

    rows = [("Model", rmse, r2)]
    base_stats = None
    if base is not None:
        if isinstance(base, (tuple, list)) and len(base) == 2 and np.isscalar(base[0]):
            base_stats = (float(base[0]) * scale, float(base[1]))
        else:
            base_stats = _stats(ref, _arr(base, scale))
        rows.append(("Nearest patch", *base_stats))
    # small statistics table in the empty upper-left corner (RMSE in axis units)
    # adjusted 2026-10-07: measured text widths (label 0.261, col1 0.128, col2 0.098 of panel
    # width, MEASURED INSIDE THE REAL FIGURE INCLUDING THE COLOURBAR) and the data occupancy of
    # all four panels; the original values overlapped the sigma single-mask cloud (134 patches
    # under the table). Table lifted and tightened; width essentially unchanged.
    # Only these five constants changed.  was: 0.04, 0.52, 0.68, 0.955, 0.072
    x0, xr, x2, y0, dy = 0.018, 0.520, 0.663, 0.988, 0.064
    kw = dict(transform=ax.transAxes, va="top", fontsize=FS_TAB)
    ax.text(xr, y0, "RMSE", ha="right", color=MUTED, **kw)
    ax.text(x2, y0, r"$R^2$", ha="right", color=MUTED, **kw)
    for i, (lab, a, b) in enumerate(rows, start=1):
        y = y0 - i * dy
        ax.text(x0, y, lab, ha="left", **kw)
        ax.text(xr, y, fmt.format(a), ha="right", **kw)
        ax.text(x2, y, f"{b:.2f}", ha="right", **kw)

    out = {"panel": name, "unit": unit_plain, "n_patches": int(ref.size),
           "sd_calculated": sd_ref, "model_rmse": rmse, "model_r2": r2,
           "sd_ratio_pred_over_calc": sd_pred / sd_ref,
           "plot_range": list(lim), "n_outside_plot_range": n_out}
    print(f"[{name}]  n = {ref.size:,}   unit = {unit_plain}")
    print(f"   SD of calculated values      {sd_ref:.5g}")
    print(f"   model          RMSE {rmse:.5g}   R2 {r2:.3f}   SD ratio {sd_pred / sd_ref:.3f}")
    if base_stats is not None:
        red = 100 * (1 - rmse / base_stats[0])
        out.update({"nearest_patch_rmse": base_stats[0], "nearest_patch_r2": base_stats[1],
                    "rmse_reduction_percent": red})
        print(f"   nearest patch  RMSE {base_stats[0]:.5g}   R2 {base_stats[1]:.3f}"
              f"   -> RMSE reduction {red:.1f}%")
    print(f"   patches outside the plotted range {lim}: {n_out:,} ({100 * n_out / ref.size:.4f}%)")
    return hb, out


def _pot_setup(pot_unit, pot_lim):
    if pot_unit == "V":
        return (E_PER_A_TO_V, pot_lim or (-6.0, 6.0), [-4, -2, 0, 2, 4], "V", "V", "{:.2f}")
    return (1.0, pot_lim or (-0.4, 0.4), [-0.4, -0.2, 0.0, 0.2, 0.4], r"e Å$^{-1}$", "e/A", "{:.3f}")


def _finish(fig, axes, hbs, out, stats, watermark):
    # one colour scale for all panels, from 1 to the largest bin count found
    vmax = max(float(h.get_array().max()) for h in hbs)
    norm = LogNorm(vmin=1.0, vmax=max(vmax, 10.0))
    for h in hbs:
        h.set_norm(norm)
    cb = fig.colorbar(hbs[0], ax=axes, shrink=0.86, pad=0.02, aspect=24)
    cb.set_label("Number of patches"); cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2, labelsize=FS_TICK)
    if watermark:
        for ax in np.ravel(axes):
            ax.text(0.97, 0.04, watermark, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.5, color="#9AA5B1")
    stats["max_patches_per_bin"] = vmax
    fig.savefig(out + ".pdf", dpi=600); fig.savefig(out + ".png", dpi=600)
    plt.close(fig)
    with open(out + "_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    print(f"saved {out}.pdf / {out}.png / {out}_stats.json")


SIG_UNIT, SIG_UNIT_PLAIN = r"10$^{-3}$ e Å$^{-2}$", "1e-3 e/A^2"
SIG_TICKS = [-20, -10, 0, 10, 20]


def make_fig2c(sig_ref, sig_pred, pot_ref, pot_pred, sig_nn=None, pot_nn=None,
               sig_lim=(-25.0, 25.0), pot_lim=None, pot_unit="eA", out="Fig2c_parity",
               gridsize=75, width_in=4.72, watermark=None):
    """Main-text panel: 24-mask-mean reconstruction, sigma (left) and potential (right).
    sig_lim is in 10^-3 e/A^2 (default: the range of the COSMO-SAC sigma-profile)."""
    font = use_font()
    k, pot_lim, pot_ticks, pot_u, pot_u_plain, pot_fmt = _pot_setup(pot_unit, pot_lim)
    s_ref, s_pred = _arr(sig_ref, 1e3), _arr(sig_pred, 1e3)
    p_ref, p_pred = _arr(pot_ref, k), _arr(pot_pred, k)
    fig, axes = plt.subplots(1, 2, figsize=(width_in, width_in * 0.50), constrained_layout=True)
    hb1, st1 = _panel(axes[0], s_ref, s_pred, sig_nn, sig_lim, SIG_TICKS, "σ", SIG_UNIT,
                      SIG_UNIT_PLAIN, "{:.2f}", gridsize, "sigma, 24-mask mean", 1e3)
    hb2, st2 = _panel(axes[1], p_ref, p_pred, pot_nn, pot_lim, pot_ticks, "potential", pot_u,
                      pot_u_plain, pot_fmt, gridsize, "potential, 24-mask mean", k)
    _finish(fig, axes, [hb1, hb2], out, {"font": font[0], "panels": [st1, st2]}, watermark)


def make_ed_parity(sig_ref, sig_single, sig_avg, pot_ref, pot_single, pot_avg, nn=None,
                   sig_lim=(-25.0, 25.0), pot_lim=None, pot_unit="eA",
                   out="ED_parity_single_vs_24mask", gridsize=75, width_in=4.72, watermark=None):
    """Extended Data figure: columns = single mask | mean of 24 masks; rows = sigma, potential."""
    font = use_font()
    nn = nn or {}
    k, pot_lim, pot_ticks, pot_u, pot_u_plain, pot_fmt = _pot_setup(pot_unit, pot_lim)
    s_ref, p_ref = _arr(sig_ref, 1e3), _arr(pot_ref, k)
    fig, axes = plt.subplots(2, 2, figsize=(width_in, width_in * 0.93), constrained_layout=True)
    stats, hbs = [], []
    cols = [("single", "Single mask"), ("avg", "Mean of 24 masks")]
    preds = {"sig_single": sig_single, "sig_avg": sig_avg, "pot_single": pot_single, "pot_avg": pot_avg}
    for j, (key, title) in enumerate(cols):
        hb, st = _panel(axes[0, j], s_ref, _arr(preds[f"sig_{key}"], 1e3), nn.get(f"sig_{key}"),
                        sig_lim, SIG_TICKS, "σ", SIG_UNIT, SIG_UNIT_PLAIN, "{:.2f}", gridsize,
                        f"sigma, {title.lower()}", 1e3, ylabel=(j == 0))
        axes[0, j].set_title(title, pad=4)
        stats.append(st); hbs.append(hb)
        hb, st = _panel(axes[1, j], p_ref, _arr(preds[f"pot_{key}"], k), nn.get(f"pot_{key}"),
                        pot_lim, pot_ticks, "potential", pot_u, pot_u_plain, pot_fmt, gridsize,
                        f"potential, {title.lower()}", k, ylabel=(j == 0))
        stats.append(st); hbs.append(hb)
    _finish(fig, axes, hbs, out, {"font": font[0], "panels": stats}, watermark)
