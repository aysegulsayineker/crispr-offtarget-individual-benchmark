"""Where the released predictors fail, by variant context.

For every variant context (seed / PAM-altering / multi-variant / heterozygous)
the within-site rank association of each predictor is recomputed on the subset
of rows in that context, with site-level bootstrap confidence intervals and a
paired contrast against the complementary subset.  p-values are
Benjamini-Hochberg corrected across all contrasts.

The purpose is to justify the feature set of the variant-aware model in
src/delta_model.py rather than choosing features arbitrarily.

A "PAM-proximal" context was defined and then dropped: only 350 rows at 19
sites qualify, and at none of them does any released score vary between
donors, so the contrast is not estimable.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analysis            # noqa: E402
import variant_features as vf   # noqa: E402

OUT = ROOT / "results"
B = 300
SEED = 0


def _within_pct(d, col):
    return d.groupby("OT_name")[col].transform(lambda x: rankdata(x) / (len(x) + 1))


def _assoc(d, p, col):
    """Within-site percentile association on rows where the model is informative."""
    d = d[d[p].notna() & d[col].notna()]
    d = d[d.groupby("OT_name")[p].transform("std") > 1e-9]
    if d.OT_name.nunique() < 5 or len(d) < 50:
        return np.nan, 0, 0
    return (float(pearsonr(_within_pct(d, p), _within_pct(d, col)).statistic),
            int(len(d)), int(d.OT_name.nunique()))


def _boot_diff(d_in, d_out, p, col, rng, B=B):
    """Paired site bootstrap of assoc(in) - assoc(out)."""
    si, so = d_in.OT_name.unique(), d_out.OT_name.unique()
    if len(si) < 5 or len(so) < 5:
        return np.nan, np.nan, np.nan
    gi = {s: g for s, g in d_in.groupby("OT_name")}
    go = {s: g for s, g in d_out.groupby("OT_name")}
    diffs = []
    for _ in range(B):
        a = pd.concat([gi[s].assign(OT_name=f"{s}__{i}")
                       for i, s in enumerate(rng.choice(si, len(si), replace=True))],
                      ignore_index=True)
        b = pd.concat([go[s].assign(OT_name=f"{s}__{i}")
                       for i, s in enumerate(rng.choice(so, len(so), replace=True))],
                      ignore_index=True)
        ra, rb = _assoc(a, p, col)[0], _assoc(b, p, col)[0]
        if not (np.isnan(ra) or np.isnan(rb)):
            diffs.append(ra - rb)
    if len(diffs) < 50:
        return np.nan, np.nan, np.nan
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pv = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(lo), float(hi), float(min(pv, 1.0))


def contexts(X):
    """Boolean masks defining the variant contexts, from the feature matrix."""
    return {
        "seed variant": X.n_seed_max > 0,
        "PAM-altering variant": (X.get("pam_created_max", 0) > 0) | (X.pam_disrupted_max > 0),
        "multiple variants": X.n_sub_max >= 2,
        "heterozygous": X.heterozygous > 0,
        "distal variant only": (X.n_distal_max > 0) & (X.n_seed_max == 0) & (X.n_pam_max == 0),
    }


def main():
    rng = np.random.default_rng(SEED)
    ot = analysis.load()
    var = ot[ot.overlapped_vars.notna()].copy()
    X = vf.build(var)
    masks = contexts(X)

    rows = []
    for aname, col in analysis.ASSAYS:
        base = var[var[col].notna()]
        for name, m in masks.items():
            mm = m.reindex(base.index).fillna(False)
            d_in, d_out = base[mm], base[~mm]
            for p in analysis.PREDS:
                r_in, n_in, s_in = _assoc(d_in, p, col)
                r_out, n_out, s_out = _assoc(d_out, p, col)
                lo, hi, pv = _boot_diff(d_in, d_out, p, col, rng)
                rows.append({"assay": aname, "context": name, "model": p,
                             "n_in": n_in, "sites_in": s_in,
                             "n_out": n_out, "sites_out": s_out,
                             "assoc_in": round(r_in, 3) if r_in == r_in else np.nan,
                             "assoc_out": round(r_out, 3) if r_out == r_out else np.nan,
                             "diff": round(r_in - r_out, 3) if r_in == r_in and r_out == r_out else np.nan,
                             "ci_lo": round(lo, 3) if lo == lo else np.nan,
                             "ci_hi": round(hi, 3) if hi == hi else np.nan,
                             "p_boot": round(pv, 4) if pv == pv else np.nan})
    out = pd.DataFrame(rows)
    ok = out.p_boot.notna()
    out.loc[ok, "p_BH"] = multipletests(out.loc[ok, "p_boot"], method="fdr_bh")[1].round(4)
    out["significant"] = out.p_BH < 0.05
    out.to_csv(OUT / "error_profile_by_context.csv", index=False)
    print(out.to_string(index=False))
    print("\nsignificant contrasts:", int(out.significant.sum()), "of", int(ok.sum()))


if __name__ == "__main__":
    main()
