"""What the between-donor result means for study design.

Two analyses, both read directly off the data rather than from a parametric
model of risk:

1. Propensity learning curve.  A donor's editing propensity is estimated from
   k measured sites drawn at random from OTHER guides, and evaluated on the
   remaining guides' sites.  This answers the design question the main result
   raises: how many sites must be assayed on a new donor before a donor-level
   term improves between-donor ranking, and where does the improvement
   saturate?  All three columns -- propensity alone, best released score alone,
   and their fitted combination -- are computed on the SAME held-out rows, so
   they are directly comparable.

2. Site-level moderators.  Which properties of an off-target site predict
   whether the released predictors can rank donors there at all?  Guide is
   carried as a fixed effect, because with six guides no guide-level
   generalisation is claimed.

A sample-size statement follows from (1): the number of sites at which a
donor's propensity must be measured for its estimate to separate the top
decile of the population from the median with a stated power, computed from
the observed between-donor propensity variance and the observed sampling
variance of the estimate.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm, pearsonr, rankdata
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analysis            # noqa: E402
import delta_model as dm   # noqa: E402
import variant_features as vf  # noqa: E402

OUT = ROOT / "results"
SEED = 0
KS = [1, 2, 3, 5, 10, 20, 40, 80]
N_DRAW = 30


def learning_curve(var, col, rng):
    """Propensity from k sites of OTHER guides; evaluated on the held-out guide."""
    d = var[var[col].notna()].copy()
    d["_mp"] = dm.wpct(d, col)
    best = max(analysis.PREDS,
               key=lambda p: (dm.assoc(d, p, col) if dm.assoc(d, p, col) == dm.assoc(d, p, col)
                              else -9))
    rows = []
    for g in sorted(d.gRNA.unique()):
        train = d[d.gRNA != g]
        test = d[d.gRNA == g].copy()
        sites_by_donor = {s: sub for s, sub in train.groupby("SampleID")}
        for k in KS:
            vals_p, vals_c, vals_b = [], [], []
            for _ in range(N_DRAW):
                eff = {}
                for s, sub in sites_by_donor.items():
                    sites = sub.OT_name.unique()
                    if len(sites) < k:
                        continue
                    pick = rng.choice(sites, k, replace=False)
                    eff[s] = float(sub[sub.OT_name.isin(pick)]._mp.mean())
                t = test.copy()
                t["_prop"] = t.SampleID.map(eff)
                t = t.dropna(subset=["_prop", best])
                if len(t) < 50:
                    continue
                r_p = dm.assoc(t, "_prop", col)
                # the two terms are combined with weights fitted on the TRAINING
                # guides; an equal-weight average understates the gain, because
                # the sequence and donor terms carry unequal signal
                tr = train.copy()
                tr["_prop"] = tr.SampleID.map(eff)
                tr = tr.dropna(subset=["_prop", best])
                if len(tr) < 100:
                    continue
                fit_cols = [dm.wpct(tr, best), dm.wpct(tr, "_prop")]
                w = LinearRegression().fit(np.column_stack(fit_cols), dm.wpct(tr, col))
                t["_comb"] = w.predict(np.column_stack([dm.wpct(t, best),
                                                        dm.wpct(t, "_prop")]))
                r_c = dm.assoc(t, "_comb", col)
                r_b = dm.assoc(t, best, col)      # same rows as r_p and r_c
                if r_p == r_p:
                    vals_p.append(r_p)
                if r_c == r_c:
                    vals_c.append(r_c)
                if r_b == r_b:
                    vals_b.append(r_b)
            if vals_p:
                rows.append({"assay": col, "gRNA": g, "k_sites": k,
                             "assoc_propensity": round(float(np.mean(vals_p)), 4),
                             "sd_propensity": round(float(np.std(vals_p)), 4),
                             "assoc_best_score_plus_propensity":
                                 round(float(np.mean(vals_c)), 4) if vals_c else np.nan,
                             "assoc_best_score_alone": (round(float(np.mean(vals_b)), 4)
                                                        if vals_b else np.nan),
                             "best_score": best})
    return pd.DataFrame(rows)


def propensity_power(var, col):
    """Sites per donor needed to place a top-decile donor above the median.

    The propensity estimate from k sites has sampling variance s2_within / k;
    the between-donor variance of the true propensity is estimated by removing
    that sampling component from the observed spread of donor means.
    """
    d = var[var[col].notna()].copy()
    d["_mp"] = dm.wpct(d, col)
    per = d.groupby(["SampleID", "OT_name"])._mp.mean().reset_index()
    s2_within = float(per.groupby("SampleID")._mp.var(ddof=1).mean())
    means = per.groupby("SampleID")._mp.mean()
    n_sites = float(per.groupby("SampleID").size().mean())
    s2_obs = float(means.var(ddof=1))
    s2_true = max(s2_obs - s2_within / n_sites, 1e-9)
    delta = norm.ppf(0.9) * np.sqrt(s2_true)       # top-decile donor vs the median
    rows = []
    for power in (0.8, 0.9):
        z = norm.ppf(power) + norm.ppf(0.975)
        k = (z ** 2) * s2_within / delta ** 2
        rows.append({"assay": col, "power": power,
                     "sd_true_propensity": round(float(np.sqrt(s2_true)), 4),
                     "sd_single_site": round(float(np.sqrt(s2_within)), 4),
                     "top_decile_delta": round(float(delta), 4),
                     "sites_needed": int(np.ceil(k))})
    return pd.DataFrame(rows)


def site_moderators(var, col):
    """Which site properties predict within-site ranking success?"""
    d = var[var[col].notna()].copy()
    X = vf.build(d)
    best = max(analysis.PREDS, key=lambda p: dm.assoc(d, p, col))
    rows = []
    for s, g in d.groupby("OT_name"):
        gg = g[g[best].notna()]
        if len(gg) < 20 or gg[best].std() < 1e-9 or gg[col].std() <= 0:
            continue
        r = float(pearsonr(rankdata(gg[best]), rankdata(gg[col])).statistic)
        xi = X.loc[gg.index]
        rows.append({"OT_name": s, "gRNA": gg.gRNA.iat[0], "rho": r,
                     "n_hap_classes": int(pd.Series(analysis.canon_hap(gg)).nunique()),
                     "n_variant_rows": int((xi.n_sub_max > 0).sum()),
                     "frac_seed": float((xi.n_seed_max > 0).mean()),
                     "any_pam_altering": int((xi.pam_disrupted_max > 0).any()),
                     "median_activity": float(gg[col].median()),
                     "frac_detected": float((gg[col] > 0).mean()),
                     "site_var": float(gg[col].var())})
    S = pd.DataFrame(rows)
    if len(S) < 20:
        return S, None
    m = smf.ols("rho ~ np.log2(n_hap_classes) + frac_seed + any_pam_altering "
                "+ frac_detected + C(gRNA)", data=S).fit()
    tab = pd.DataFrame({"term": m.params.index, "coef": m.params.round(4).values,
                        "se": m.bse.round(4).values, "p": m.pvalues.round(4).values})
    tab = tab[~tab.term.str.startswith("C(gRNA)")]
    tab["assay"] = col
    tab["n_sites"] = len(S)
    tab["r2"] = round(float(m.rsquared), 3)
    return S, tab


def main():
    rng = np.random.default_rng(SEED)
    ot = analysis.load()
    var = ot[ot.overlapped_vars.notna()].copy()
    curves, powers, mods, sites = [], [], [], []
    for aname, col in analysis.ASSAYS:
        lc = learning_curve(var, col, rng); lc["assay_name"] = aname
        curves.append(lc)
        pw = propensity_power(var, col); pw["assay_name"] = aname
        powers.append(pw)
        S, tab = site_moderators(var, col)
        S["assay_name"] = aname; sites.append(S)
        if tab is not None:
            tab["assay_name"] = aname; mods.append(tab)
    pd.concat(curves).to_csv(OUT / "propensity_learning_curve.csv", index=False)
    pd.concat(powers).to_csv(OUT / "propensity_power.csv", index=False)
    pd.concat(sites).to_csv(OUT / "site_ranking_success.csv", index=False)
    pd.concat(mods).to_csv(OUT / "site_moderators.csv", index=False)
    print(pd.concat(curves).groupby(["assay_name", "k_sites"])
          [["assoc_propensity", "assoc_best_score_plus_propensity",
            "assoc_best_score_alone"]].mean().round(4).to_string())
    print("\n", pd.concat(powers).to_string(index=False))
    print("\n", pd.concat(mods).to_string(index=False))


if __name__ == "__main__":
    main()
