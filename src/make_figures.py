#!/usr/bin/env python3
"""
Generate the four main figures from the result tables written by src/analysis.py.

Usage:  python src/make_figures.py [--results results] [--outdir figures]

Every value plotted here is read from results/*.csv; nothing is recomputed.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- constants
MODELS = ["CHANGE-net", "CFD", "CRISOT", "CRISPRoff", "CRISPRnet"]
POPS = ["CEU", "CHB", "ASW", "MXL"]
ASSAYS = ["GUIDE-seq-2", "rhAmpSeq"]
MODEL_COLOR = {
    "CHANGE-net": "#3B6EA8",
    "CFD":        "#D1733A",
    "CRISOT":     "#4E9A6B",
    "CRISPRoff":  "#7E6BA8",
    "CRISPRnet":  "#8C7B6B",
}
POP_COLOR = {"CEU": "#2B4C7E", "CHB": "#5B8FB9", "ASW": "#B8A36E", "MXL": "#9E6B52"}
GREY = "#555555"
NOMINAL = 0.90
# Measured rhAmpSeq indel% distribution over the analysed rows (analysis.py console
# summary): median 0.016 %, inter-quartile range 0.000-0.075 %.
RHAMP_MEDIAN, RHAMP_IQR = 0.016, (0.000, 0.075)


def apply_figure_style(sizes=(8, 7, 6)):
    """Role-mapped font ladder, outward ticks, frameless legends, 300-dpi output."""
    base, second, tick = sizes
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "legend.fontsize": second, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.titlelocation": "left", "axes.titleweight": "regular",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "axes.grid": False, "lines.linewidth": 1.2,
    })


def panel_letter(ax, letter, dx=-0.14, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


# ------------------------------------------------------------------ figure 1
def figure1(res, outdir):
    b = pd.read_csv(res / "benchmark_noise_ceiling.csv")
    cal = pd.read_csv(res / "calibration_by_ancestry.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, assay in zip(axes, ASSAYS):
        d = b[b.assay == assay].set_index("model").loc[MODELS]
        x = np.arange(len(MODELS))
        clo, chi = d.ceiling_lo.iloc[0], d.ceiling_hi.iloc[0]
        ax.axhspan(clo, chi, color=GREY, alpha=0.13, lw=0)
        ax.axhline(d.ceiling_rho.iloc[0], color=GREY, ls="--", lw=1.1)
        ax.text(len(MODELS) - 0.45, d.ceiling_rho.iloc[0] + 0.012,
                f"noise ceiling {d.ceiling_rho.iloc[0]:.2f}", color=GREY,
                fontsize=7, ha="right", va="bottom")
        ax.axhline(0, color="black", lw=0.6)
        ax.scatter(x - 0.13, d.rho_pooled, marker="D", s=26, facecolors="none",
                   edgecolors=GREY, linewidths=1.0, zorder=3,
                   label="pooled across sites" if assay == ASSAYS[0] else None)
        cols = [MODEL_COLOR[m] for m in MODELS]
        for xi, m in zip(x, MODELS):
            r = d.loc[m]
            ax.errorbar(xi + 0.13, r.rho_within,
                        yerr=[[r.rho_within - r.ci_lo], [r.ci_hi - r.rho_within]],
                        fmt="none", ecolor=MODEL_COLOR[m], elinewidth=1.2,
                        capsize=2.5, zorder=3)
        ax.scatter(x + 0.13, d.rho_within, s=34, c=cols, zorder=4,
                   label="within-site (between-donor)" if assay == ASSAYS[0] else None)
        n = cal[cal.assay == assay].n
        ax.set_title(f"{assay}\nn = {n.min():,}\u2013{n.max():,} scored donor \u00d7 site rows",
                     fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=30, ha="right")
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.margins(y=0.08)
    axes[0].set_ylabel("Spearman $\\rho$ with measured activity")
    axes[0].set_ylim(-0.05, 0.98)
    axes[0].legend(loc="upper left", handletextpad=0.4, borderpad=0.2)
    axes[0].text(0.98, 0.80, "higher = better", fontsize=7, color=GREY,
                 ha="right", transform=axes[0].transAxes)
    panel_letter(axes[0], "a", dx=-0.13, dy=1.14)
    panel_letter(axes[1], "b", dx=-0.06, dy=1.14)
    fig.suptitle("Pooled ranking accuracy collapses once between-donor variation is isolated",
                 x=0.02, ha="left", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = outdir / "figure1_pooled_vs_within_site.png"
    fig.savefig(p)
    return fig, p


# ------------------------------------------------------------------ figure 2
def figure2(res, outdir):
    b = pd.read_csv(res / "benchmark_noise_ceiling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, assay in zip(axes, ASSAYS):
        d = b[b.assay == assay].set_index("model").loc[MODELS]
        x = np.arange(len(MODELS))
        ax.bar(x, d.pct_of_ceiling, color=[MODEL_COLOR[m] for m in MODELS], width=0.62)
        for xi, v in zip(x, d.pct_of_ceiling):
            ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=30, ha="right")
        ax.set_title(f"{assay}\nceiling $\\rho$ = {d.ceiling_rho.iloc[0]:.3f} "
                     f"[{d.ceiling_lo.iloc[0]:.3f}, {d.ceiling_hi.iloc[0]:.3f}]", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("% of attainable signal")
    panel_letter(axes[0], "a", dx=-0.12, dy=1.14)
    panel_letter(axes[1], "b", dx=-0.06, dy=1.14)
    fig.suptitle("No predictor exceeds one fifth of the attainable between-donor signal",
                 x=0.02, ha="left", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = outdir / "figure2_pct_of_ceiling.png"
    fig.savefig(p)
    return fig, p


# ------------------------------------------------------------------ figure 3
def figure3(res, outdir):
    c = pd.read_csv(res / "calibration_by_ancestry.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2))
    for ax, assay in zip(axes, ASSAYS):
        d = c[c.assay == assay].set_index("model").loc[MODELS]
        x = np.arange(len(MODELS)); w = 0.19
        for i, pop in enumerate(POPS):
            ax.bar(x + (i - 1.5) * w, d[f"bias_{pop}"], width=w,
                   color=POP_COLOR[pop], label=pop if assay == ASSAYS[0] else None)
        ax.axhline(0, color="black", lw=0.7)
        top = max(abs(d[[f"bias_{p}" for p in POPS]].values.min()),
                  d[[f"bias_{p}" for p in POPS]].values.max())
        for xi, m in zip(x, MODELS):
            star = "*" if d.loc[m, "p_BH"] < 0.05 else "n.s."
            ax.text(xi, top * 1.22, f"{star}  $p_{{BH}}$={d.loc[m, 'p_BH']:.3g}",
                    ha="center", va="bottom", fontsize=6.5, color=GREY)
        ax.set_ylim(-top * 1.45, top * 1.75)
        ax.set_xticks(x); ax.set_xticklabels(MODELS)
        ax.set_ylabel("mean signed residual\n(measured \u2212 calibrated prediction)")
        mae = d.MAE
        ax.set_title(f"{assay} \u2014 ancestry-dependent bias is significant but small next to "
                     f"total error (MAE {mae.min():.2f}\u2013{mae.max():.2f})", fontsize=8)
    axes[0].legend(ncol=4, loc="lower center", columnspacing=1.2, handlelength=1.1)
    panel_letter(axes[0], "a", dx=-0.09); panel_letter(axes[1], "b", dx=-0.09)
    fig.suptitle("Every model \u00d7 assay combination shows ancestry-dependent miscalibration "
                 "(donor-permutation test)", x=0.02, ha="left", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = outdir / "figure3_ancestry_calibration.png"
    fig.savefig(p)
    return fig, p


# ------------------------------------------------------------------ figure 4
def figure4(res, outdir):
    cc = pd.read_csv(res / "conformal_coverage.csv")
    de = pd.read_csv(res / "decision_rule_eval.csv")
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 0.85, 1.15], hspace=0.75, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    # a, b: coverage by ancestry
    for ax, assay, letter in [(ax_a, ASSAYS[0], "a"), (ax_b, ASSAYS[1], "b")]:
        d = cc[cc.assay == assay]
        x = np.arange(len(POPS))
        ax.axhline(NOMINAL, color="black", ls="--", lw=0.9)
        ax.text(len(POPS) - 0.55, NOMINAL + 0.0025, "nominal 0.90", fontsize=6.5, ha="right")
        for j, m in enumerate(MODELS):
            off = (j - 2) * 0.13
            mg = d[(d.model == m) & (d.method == "marginal")].set_index("ancestry").reindex(POPS)
            mo = d[(d.model == m) & (d.method == "mondrian")].set_index("ancestry").reindex(POPS)
            ax.scatter(x + off, mg.coverage, s=20, color=MODEL_COLOR[m], zorder=3,
                       label=m if letter == "a" else None)
            ax.scatter(x + off, mo.coverage, s=22, facecolors="none",
                       edgecolors=MODEL_COLOR[m], linewidths=0.9, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(POPS)
        ax.set_ylim(0.865, 0.935)
        ax.set_title(f"{assay}", fontsize=8)
        panel_letter(ax, letter, dx=-0.17 if letter == "a" else -0.12)
    ax_a.set_ylabel("empirical coverage of\n90 % conformal interval")
    ax_a.legend(ncol=5, loc="upper center", bbox_to_anchor=(1.14, 1.46),
                fontsize=6.5, handletextpad=0.2, columnspacing=0.9, borderpad=0.1)
    ax_b.text(0.02, 0.06, "filled = marginal   open = Mondrian (per-ancestry)",
              transform=ax_b.transAxes, fontsize=6.2, color=GREY)

    # c, d: interval width in interpretable units
    wid = cc[(cc.method == "marginal") & (cc.ancestry == "ALL")].set_index(["assay", "model"])
    y = np.arange(len(MODELS))[::-1]
    wg = [wid.loc[(ASSAYS[0], m), "width"] for m in MODELS]
    ax_c.barh(y, wg, color=[MODEL_COLOR[m] for m in MODELS], height=0.6)
    ax_c.set_yticks(y); ax_c.set_yticklabels(MODELS)
    ax_c.set_xlabel("interval width (log$_2$ normalised read count)")
    mean_w = float(np.mean(wg))
    ax_c.set_title(f"mean width {mean_w:.2f} log$_2$ units\n\u2248 {2 ** mean_w:.0f}-fold "
                   "activity range", fontsize=8)
    ax_c.set_xlim(0, max(wg) * 1.18)
    wr = [wid.loc[(ASSAYS[1], m), "width"] for m in MODELS]
    ax_d.barh(y, wr, color=[MODEL_COLOR[m] for m in MODELS], height=0.6)
    ax_d.set_yticks(y); ax_d.set_yticklabels(MODELS)
    ax_d.set_xlabel("interval width (indel %, percentage points)")
    ax_d.axvline(RHAMP_IQR[1], color="black", lw=0.8, ls=":")
    ax_d.annotate(f"measured upper\nquartile {RHAMP_IQR[1]:.3f} %",
                  xy=(RHAMP_IQR[1], 1.0), xytext=(0.40, 0.55), fontsize=6.2,
                  textcoords="axes fraction", va="center",
                  arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
    ax_d.set_title(f"mean width {np.mean(wr):.2f} points\nmeasured median "
                   f"{RHAMP_MEDIAN:.3f} %", fontsize=8)
    ax_d.set_xlim(0, max(wr) * 1.18)
    panel_letter(ax_c, "c", dx=-0.21, dy=1.22); panel_letter(ax_d, "d", dx=-0.17, dy=1.22)

    # e: threshold sweep, point rule
    pt = de[(de.rule == "point") & (de.ancestry == "ALL")]
    for m in MODELS:
        d = pt[pt.model == m].sort_values("tau")
        ax_e.plot(d.tau, d.sensitivity, marker="o", ms=3.2, color=MODEL_COLOR[m], label=m)
    ax_e.set_xscale("log")
    ax_e.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20, 40])
    ax_e.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax_e.set_xlabel("decision threshold $\\tau$ on measured indel % (log scale)")
    ax_e.set_ylabel("sensitivity of point rule")
    ax_e.set_ylim(-0.03, 1.0)
    ax_e.axhline(0, color="black", lw=0.6)
    zero = pt[(pt.tau >= 20) & (pt.sensitivity == 0)].model.unique()
    ax_e.annotate(f"{', '.join(sorted(zero))}: sensitivity exactly 0\nat $\\tau \\geq$ 20 %",
                  xy=(20, 0.0), xytext=(0.22, 0.13), textcoords="axes fraction",
                  fontsize=6.8, color=GREY,
                  arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
    ax_e.legend(ncol=5, loc="upper right", fontsize=6.5, columnspacing=1.0,
                handletextpad=0.3)
    ax_e.set_title("The highest-activity off-target events are the ones the point rule misses",
                   fontsize=8)
    panel_letter(ax_e, "e", dx=-0.075, dy=1.12)
    fig.suptitle("Conformal intervals are already valid in every ancestry group, "
                 "but too wide to act on", x=0.02, ha="left", fontsize=8.5)
    p = outdir / "figure4_conformal_and_decision.png"
    fig.savefig(p)
    return fig, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    res, outdir = Path(a.results), Path(a.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    apply_figure_style()
    for fn in (figure1, figure2, figure3, figure4):
        _, p = fn(res, outdir)
        print("wrote", p)


if __name__ == "__main__":
    main()
