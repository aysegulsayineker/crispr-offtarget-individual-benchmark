"""Can the uncaptured between-donor signal be recovered?

Five nested predictors of the WITHIN-SITE (between-donor) component are
compared on identical rows and identical folds:

  M1 baseline   the five released predictor scores
  M2 variant    variant features derived from the donor's haplotypes
  M3 combined   M1 + M2
  M4 donor      a donor propensity estimated from that donor's measurements at
                OTHER sites (or other guides); no sequence information at all
  M5 full       M3 + M4

M3 - M1 asks whether variant features add information the released scores do
not already carry.  M5 - M3 asks whether a donor-level term adds information no
sequence model can carry.

Scale.  Features and target are on the within-site percentile scale used
throughout the study, so M1 reproduces the published metric up to the ensemble
weighting.  The target percentile is computed over TRAINING rows only.

Generalisation.  Two schemes, never a random row split:

  guide-held-out (primary)  leave-one-guide-out.  The held-out guide's sites are
      absent from training, so all 95 donors at a test site are scored by the
      same model and the metric is directly comparable to the published values.
  donor-held-out            repeated grouped 2-fold on SampleID.  Here a site's
      donors are split across folds, so the association is evaluated WITHIN each
      fold -- pooling predictions from different fitted models would compare
      donors scored by different functions.  Because the percentile transform
      has fewer ties in a smaller donor set, absolute values under this scheme
      run higher than the published ones; the released scores are therefore
      evaluated on the identical subsets as the reference.

Estimators.  Ridge and histogram gradient boosting (scikit-learn's
gradient-boosted trees; xgboost is not used, to avoid an extra dependency for
an equivalent model family).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analysis                  # noqa: E402
import variant_features as vf    # noqa: E402

OUT = ROOT / "results"
SEED = 0
N_REPEATS = 10          # repeats of the grouped 2-fold donor scheme
B_BOOT = 300
SITE_CTX = ["site_ref_pam_intact", "site_has_bulge", "site_len", "site_n_mm_ref"]


def wpct(d, col):
    return d.groupby("OT_name")[col].transform(lambda x: rankdata(x) / (len(x) + 1))


def assoc(d, pred_col, col):
    d = d[d[pred_col].notna() & d[col].notna()]
    d = d[d.groupby("OT_name")[pred_col].transform("std") > 1e-9]
    d = d[d.groupby("OT_name")[col].transform("std") > 0]
    if d.OT_name.nunique() < 5 or len(d) < 50:
        return np.nan
    return float(pearsonr(wpct(d, pred_col), wpct(d, col)).statistic)


def site_avg_assoc(d, pred_col, col, min_n=5):
    rs = []
    for _, g in d.groupby("OT_name"):
        g = g[g[pred_col].notna() & g[col].notna()]
        if len(g) < min_n or g[pred_col].std() < 1e-9 or g[col].std() < 1e-9:
            continue
        r = float(pearsonr(rankdata(g[pred_col]), rankdata(g[col])).statistic)
        if r == r:
            rs.append(r)
    return (float(np.mean(rs)), len(rs)) if rs else (np.nan, 0)


def estimator(name):
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             Ridge(alpha=10.0))
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(
                             max_depth=4, max_iter=300, learning_rate=0.06,
                             min_samples_leaf=50, l2_regularization=1.0,
                             random_state=SEED))


def donor_propensity(d, col, unit):
    """Leave-one-<unit>-out mean within-site percentile for each donor.

    unit='OT_name' excludes the evaluated site; unit='gRNA' excludes every site
    of the evaluated guide, which is the stricter construction.
    """
    mp = wpct(d, col)
    t = d.assign(_mp=mp)
    tot = t.groupby("SampleID")._mp.agg(["sum", "count"])
    sub = t.groupby(["SampleID", unit])._mp.agg(["sum", "count"])
    a = t[["SampleID", unit]].merge(tot, left_on="SampleID", right_index=True, how="left")
    b = t[["SampleID", unit]].merge(sub, left_on=["SampleID", unit], right_index=True,
                                    how="left", suffixes=("", "_s"))
    num = a["sum"].to_numpy() - b["sum"].to_numpy()
    den = a["count"].to_numpy() - b["count"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series(np.where(den > 0, num / den, np.nan), index=d.index)


def build_frame(var, col):
    """Rows scorable by every released predictor, with all feature blocks."""
    d = var[var[col].notna() & var[analysis.PREDS].notna().all(axis=1)].copy()
    X = vf.build(d)
    for p in analysis.PREDS:
        d["pct_" + p] = wpct(d, p)
    d["donor_site"] = donor_propensity(d, col, "OT_name")
    d["donor_guide"] = donor_propensity(d, col, "gRNA")
    ctx = [c for c in SITE_CTX if c in X.columns]
    varcols = [c for c in X.columns if c not in ctx]
    F = pd.concat([d[["pct_" + p for p in analysis.PREDS] + ["donor_site", "donor_guide"]],
                   X], axis=1)
    sets = {
        "M1 baseline": ["pct_" + p for p in analysis.PREDS] + ctx,
        "M2 variant": varcols + ctx,
        "M3 combined": ["pct_" + p for p in analysis.PREDS] + varcols + ctx,
        "M4 donor": ["donor_site"],
        "M5 full": ["pct_" + p for p in analysis.PREDS] + varcols + ctx + ["donor_site"],
    }
    return d.reset_index(drop=True), F.reset_index(drop=True), sets


def fit_guide_cv(d, F, col, cols, est_name):
    """Leave-one-guide-out; the donor term is rebuilt excluding the test guide."""
    e = estimator(est_name)
    feats = F[cols].astype(float).copy()
    if "donor_site" in cols:                       # stricter, leak-free across guides
        feats["donor_site"] = F["donor_guide"].to_numpy()
    y = d[col].to_numpy(dtype=float)
    pred = np.full(len(d), np.nan)
    for g in sorted(d.gRNA.unique()):
        tr = np.flatnonzero(d.gRNA.values != g)
        te = np.flatnonzero(d.gRNA.values == g)
        ytr = wpct(d.iloc[tr], col).to_numpy()
        keep = np.isfinite(ytr)
        e.fit(feats.iloc[tr].iloc[keep], ytr[keep])
        pred[te] = e.predict(feats.iloc[te])
    return d.assign(_pred=pred)


def donor_cv_scores(d, F, col, cols, est_name, rng, n_repeats=N_REPEATS):
    """Repeated grouped 2-fold on donors, evaluated within each held-out fold."""
    e = estimator(est_name)
    feats = F[cols].astype(float)
    y = d[col].to_numpy(dtype=float)
    vals, ns = [], []
    donors = d.SampleID.unique()
    for rep in range(n_repeats):
        order = rng.permutation(donors)
        half = set(order[: len(order) // 2])
        for test_set in (half, set(order) - half):
            te = np.flatnonzero(d.SampleID.isin(test_set).values)
            tr = np.flatnonzero(~d.SampleID.isin(test_set).values)
            ytr = wpct(d.iloc[tr], col).to_numpy()
            keep = np.isfinite(ytr)
            e.fit(feats.iloc[tr].iloc[keep], ytr[keep])
            dte = d.iloc[te].assign(_pred=e.predict(feats.iloc[te]))
            r = assoc(dte, "_pred", col)
            if r == r:
                vals.append(r); ns.append(len(dte))
    return (float(np.average(vals, weights=ns)), float(np.std(vals))) if vals else (np.nan, np.nan)


def reference_under_donor_cv(d, col, rng, n_repeats=N_REPEATS):
    """Released scores on the identical held-out donor subsets."""
    out = {}
    donors = d.SampleID.unique()
    for p in analysis.PREDS:
        vals, ns = [], []
        r2 = np.random.default_rng(SEED)        # same partitions for every model
        for rep in range(n_repeats):
            order = r2.permutation(donors)
            half = set(order[: len(order) // 2])
            for test_set in (half, set(order) - half):
                dte = d[d.SampleID.isin(test_set)]
                r = assoc(dte, "pct_" + p, col)
                if r == r:
                    vals.append(r); ns.append(len(dte))
        out[p] = float(np.average(vals, weights=ns)) if vals else np.nan
    return out


def paired_boot(Da, Db, col, rng, B=B_BOOT):
    sites = Da.OT_name.unique()
    ga = {s: g for s, g in Da.groupby("OT_name")}
    gb = {s: g for s, g in Db.groupby("OT_name")}
    diffs = []
    for _ in range(B):
        draw = rng.choice(sites, len(sites), replace=True)
        ra = assoc(pd.concat([ga[s].assign(OT_name=f"{s}__{i}") for i, s in enumerate(draw)],
                             ignore_index=True), "_pred", col)
        rb = assoc(pd.concat([gb[s].assign(OT_name=f"{s}__{i}") for i, s in enumerate(draw)],
                             ignore_index=True), "_pred", col)
        if ra == ra and rb == rb:
            diffs.append(rb - ra)
    if len(diffs) < 50:
        return (np.nan,) * 4
    a = np.array(diffs)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(a.mean()), float(lo), float(hi), float(min(2 * min((a <= 0).mean(),
                                                                    (a >= 0).mean()), 1.0))


def donor_effect_checks(var, rng, n_perm=200):
    """Is the donor propensity real, and does it survive a change of assay?"""
    rows = []
    eff = {}
    for aname, col in analysis.ASSAYS:
        d = var[var[col].notna()].copy()
        d["_mp"] = wpct(d, col)
        obs = float(d.groupby("SampleID")._mp.mean().std())
        mp = d._mp.to_numpy(); sid = d.SampleID.to_numpy()
        idx = [np.flatnonzero(d.OT_name.values == s) for s in d.OT_name.unique()]
        null = []
        for _ in range(n_perm):
            perm = sid.copy()
            for ix in idx:
                perm[ix] = rng.permutation(sid[ix])
            null.append(float(pd.Series(mp).groupby(perm).mean().std()))
        null = np.array(null)
        rows.append({"assay": aname, "donor_mean_sd": round(obs, 4),
                     "null_mean": round(float(null.mean()), 4),
                     "null_sd": round(float(null.std()), 4),
                     "p_perm": round(float((null >= obs).mean()), 4),
                     "n_donors": int(d.SampleID.nunique())})
        eff[aname] = d.groupby("SampleID")._mp.mean()

    a, b = analysis.ASSAYS[0][0], analysis.ASSAYS[1][0]
    both = pd.concat([eff[a].rename("a"), eff[b].rename("b")], axis=1).dropna()
    cross = {"assay": "cross-assay",
             "donor_effect_r_between_assays": round(float(pearsonr(both.a, both.b).statistic), 4),
             "n_donors": int(len(both))}
    d2 = var[var[analysis.ASSAYS[1][1]].notna()].copy()
    d2["from_other_assay"] = d2.SampleID.map(eff[a])
    cross["transfer_assoc"] = round(assoc(d2.dropna(subset=["from_other_assay"]),
                                          "from_other_assay", analysis.ASSAYS[1][1]), 4)
    g = var[var[analysis.ASSAYS[0][1]].notna()]
    dep = g.groupby("SampleID").guideseqRC.sum()
    common = eff[a].index.intersection(dep.index)
    cross["r_with_total_readcount"] = round(float(pearsonr(eff[a].loc[common],
                                                           dep.loc[common]).statistic), 3)
    return pd.DataFrame(rows), pd.DataFrame([cross])


def main():
    rng = np.random.default_rng(SEED)
    ot = analysis.load()
    var = ot[ot.overlapped_vars.notna()].copy()

    perf, comps, guides = [], [], []
    for aname, col in analysis.ASSAYS:
        d, F, sets = build_frame(var, col)
        ref_guide = {p: assoc(d, "pct_" + p, col) for p in analysis.PREDS}
        ref_donor = reference_under_donor_cv(d, col, rng)
        for p in analysis.PREDS:
            perf.append({"assay": aname, "estimator": "-", "model": f"released {p}",
                         "assoc_guide_cv": round(ref_guide[p], 4),
                         "assoc_donor_cv": round(ref_donor[p], 4)})
        store = {}
        for est in ("ridge", "gbt"):
            for label, cols in sets.items():
                D = fit_guide_cv(d, F, col, cols, est)
                store[(est, label)] = D[D._pred.notna()]
                a_g = assoc(D, "_pred", col)
                sa, ns = site_avg_assoc(D[D._pred.notna()], "_pred", col)
                a_d, sd_d = donor_cv_scores(d, F, col, cols, est, np.random.default_rng(SEED))
                dd = D[D._pred.notna()].copy()
                dd = dd[dd.groupby("OT_name")["_pred"].transform("std") > 1e-9]
                dd["_mp"] = wpct(dd, col)
                c = analysis.vc_ceiling_on(dd, "_mp")
                perf.append({"assay": aname, "estimator": est, "model": label,
                             "assoc_guide_cv": round(a_g, 4) if a_g == a_g else np.nan,
                             "assoc_donor_cv": round(a_d, 4) if a_d == a_d else np.nan,
                             "assoc_donor_cv_sd": round(sd_d, 4) if sd_d == sd_d else np.nan,
                             "site_avg_assoc": round(sa, 4) if sa == sa else np.nan,
                             "n_sites_site_avg": ns,
                             "ceiling_matched": round(c["ceiling"], 4) if c else np.nan,
                             "pct_of_ceiling": (round(100 * a_g / c["ceiling"], 1)
                                                if c and a_g == a_g else np.nan)})
                for g_, gd in D[D._pred.notna()].groupby("gRNA"):
                    r = assoc(gd, "_pred", col)
                    guides.append({"assay": aname, "estimator": est, "model": label,
                                   "gRNA": g_, "assoc": round(r, 4) if r == r else np.nan})
        for est in ("ridge", "gbt"):
            for a_lab, b_lab in (("M1 baseline", "M3 combined"),
                                 ("M1 baseline", "M5 full"),
                                 ("M3 combined", "M5 full"),
                                 ("M1 baseline", "M2 variant"),
                                 ("M1 baseline", "M4 donor")):
                m, lo, hi, pv = paired_boot(store[(est, a_lab)], store[(est, b_lab)], col, rng)
                comps.append({"assay": aname, "estimator": est,
                              "contrast": f"{b_lab} - {a_lab}",
                              "mean_diff": round(m, 4) if m == m else np.nan,
                              "ci_lo": round(lo, 4) if lo == lo else np.nan,
                              "ci_hi": round(hi, 4) if hi == hi else np.nan,
                              "p_boot": round(pv, 4) if pv == pv else np.nan})

    P = pd.DataFrame(perf)
    C = pd.DataFrame(comps)
    ok = C.p_boot.notna()
    C.loc[ok, "p_BH"] = multipletests(C.loc[ok, "p_boot"], method="fdr_bh")[1].round(4)
    C["significant"] = C.p_BH < 0.05
    G = pd.DataFrame(guides)
    DE, CR = donor_effect_checks(var, np.random.default_rng(SEED))

    P.to_csv(OUT / "delta_model_performance.csv", index=False)
    C.to_csv(OUT / "delta_model_contrasts.csv", index=False)
    G.to_csv(OUT / "delta_model_per_guide.csv", index=False)
    DE.to_csv(OUT / "donor_effect_permutation.csv", index=False)
    CR.to_csv(OUT / "donor_effect_cross_assay.csv", index=False)
    for name, t in (("performance", P), ("contrasts", C),
                    ("donor permutation", DE), ("cross-assay", CR)):
        print(f"\n=== {name} ===\n{t.to_string(index=False)}")


if __name__ == "__main__":
    main()
