"""Figures 5-6: the variant-aware model and the donor propensity.

Reads results/*.csv only, so it runs without the raw data.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from make_figures import apply_figure_style, panel_letter   # noqa: E402

ASSAYS = ["GUIDE-seq-2", "rhAmpSeq"]
ORDER = ["M1 baseline", "M2 variant", "M3 combined", "M4 donor", "M5 full"]
SHORT = {"M1 baseline": "released\nscores", "M2 variant": "variant\nfeatures",
         "M3 combined": "scores +\nvariant", "M4 donor": "donor\npropensity",
         "M5 full": "scores + variant\n+ donor"}
C = {"M1 baseline": "#4878a8", "M2 variant": "#b0b0b0", "M3 combined": "#7fa8c9",
     "M4 donor": "#d1894f", "M5 full": "#c0392b"}


def fig5():
    P = pd.read_csv(RES / "delta_model_performance.csv")
    G = pd.read_csv(RES / "delta_model_per_guide.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5),
                             gridspec_kw={"width_ratios": [1.25, 1.25, 1.0]})
    for ax, assay in zip(axes[:2], ASSAYS):
        sub = P[(P.assay == assay) & (P.estimator == "ridge")].set_index("model")
        rel = P[(P.assay == assay) & (P.estimator == "-")]
        best = rel.assoc_guide_cv.max()
        best_name = rel.loc[rel.assoc_guide_cv.idxmax(), "model"].replace("released ", "")
        x = np.arange(len(ORDER))
        ax.bar(x, [sub.loc[m, "assoc_guide_cv"] for m in ORDER],
               color=[C[m] for m in ORDER], width=0.68)
        ax.axhline(best, color="0.25", ls="--", lw=1.1)
        ax.text(len(ORDER) - 0.55, best * 1.045, f"best released score ({best_name})",
                va="bottom", ha="right", fontsize=6, color="0.25")
        for xi, m in zip(x, ORDER):
            v = sub.loc[m, "assoc_guide_cv"]
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=6)
        ax.set_ylim(0, max(0.22, best * 1.25))
        ax.set_title(assay)
        if ax is axes[0]:
            ax.set_ylabel("between-donor rank association\n(leave-one-guide-out)")
            ax.text(0.02, 0.96, "higher = better", transform=ax.transAxes,
                    fontsize=6, color="0.4", va="top")
        ax.margins(x=0.04)

    ax = axes[2]
    g = G[(G.estimator == "ridge") & (G.model.isin(["M1 baseline", "M5 full"]))]
    piv = g.pivot_table(index=["assay", "gRNA"], columns="model", values="assoc")
    piv["gain"] = piv["M5 full"] - piv["M1 baseline"]
    piv = piv.sort_values("gain")
    y = np.arange(len(piv))
    cols = ["#c0392b" if v > 0 else "#7f8c8d" for v in piv.gain]
    ax.barh(y, piv.gain, color=cols, height=0.68)
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a.split('-')[0][:5]}  {b.replace('_site','')}"
                        for a, b in piv.index], fontsize=6)
    ax.set_xlabel("gain from the donor term\n(M5 − M1, per guide)")
    ax.set_title(f"{int((piv.gain > 0).sum())} of {len(piv)} guide × assay pairs improve")
    for a, L in zip(axes, "abc"):
        panel_letter(a, L)
    fig.tight_layout()
    fig.savefig(FIG / "figure5_variant_aware_model.png", dpi=300)
    return fig


def fig6():
    DE = pd.read_csv(RES / "donor_effect_permutation.csv")
    CR = pd.read_csv(RES / "donor_effect_cross_assay.csv")
    LC = pd.read_csv(RES / "propensity_learning_curve.csv")
    H = pd.read_csv(RES / "hurdle_model.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4))

    ax = axes[0]
    x = np.arange(len(DE))
    ax.bar(x - 0.18, DE.null_mean, yerr=2 * DE.null_sd, width=0.34,
           color="0.75", label="permuted donors", capsize=2)
    ax.bar(x + 0.18, DE.donor_mean_sd, width=0.34, color="#c0392b", label="observed")
    for xi, v in zip(x, DE.donor_mean_sd):
        ax.text(xi + 0.18, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(DE.assay)
    ax.set_ylabel("SD of donor mean percentile")
    ax.set_title("Donor propensity exceeds the\npermutation null in both assays")
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    ax = axes[1]
    for assay, c in zip(ASSAYS, ["#4878a8", "#c0392b"]):
        s = LC[LC.assay_name == assay].groupby("k_sites")[
            ["assoc_propensity", "assoc_best_score_alone"]].mean()
        ax.plot(s.index, s.assoc_propensity, "o-", color=c, ms=3.5, label=assay)
        ax.axhline(float(s.assoc_best_score_alone.iloc[0]), color=c, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 40, 80])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("sites measured per donor")
    ax.set_ylabel("between-donor rank association\nof the donor propensity alone")
    ax.set_title("Donor propensity needs many sites\n(dotted: best released score)")
    ax.legend(frameon=False, fontsize=6, loc="lower right")

    ax = axes[2]
    h = H[(H.lod.isin(["detected read", "0.1% indels"]))]
    lab = {"M1 baseline": "released scores", "M3 combined": "+ variant",
           "M5 full": "+ variant + donor"}
    x = np.arange(len(ASSAYS))
    w = 0.26
    for i, m in enumerate(["M1 baseline", "M3 combined", "M5 full"]):
        vals = [float(h[(h.assay == a) & (h.model == m)].assoc_stageB.iloc[0]) for a in ASSAYS]
        ax.bar(x + (i - 1) * w, vals, width=w, color=C[m], label=lab[m])
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(ASSAYS)
    ax.set_ylabel("rank association among rows\nwith detectable editing")
    ax.set_title("Given detectable editing, the donor term\nhelps on rhAmpSeq but not GUIDE-seq-2")
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    for a, L in zip(axes, "abc"):
        panel_letter(a, L)
    fig.tight_layout()
    fig.savefig(FIG / "figure6_donor_propensity.png", dpi=300)
    return fig


if __name__ == "__main__":
    apply_figure_style()
    fig5(); fig6()
    print("wrote figure5_variant_aware_model.png, figure6_donor_propensity.png")
