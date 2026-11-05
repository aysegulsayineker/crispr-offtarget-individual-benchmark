"""Does the donor-aware model inherit the ancestry-dependent miscalibration?

Every one of the ten released model x assay combinations shows statistically
significant ancestry-dependent calibration bias.  A model that adds a
donor-level term could plausibly make that worse -- the donor propensity
differs between the four cohort populations -- so the same audit is applied to
the new predictors: cross-fitted isotonic recalibration, then a donor-level
permutation test on the spread of population mean residuals.

Discrimination is reported alongside, because the two can move independently.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analysis            # noqa: E402
import delta_model as dm   # noqa: E402

OUT = ROOT / "results"
SEED = 0
MODELS = ["M1 baseline", "M3 combined", "M5 full"]
N_PERM = 2000


def audit(d, col, rng):
    cd = analysis.crossfit_calibrate(d, "_pred", col)
    if cd is None:
        return None
    rec = {"n": int(len(cd)), "MAE": round(float(cd.resid.abs().mean()), 3)}
    for pop in analysis.POPS:
        rec[f"bias_{pop}"] = round(float(cd[cd.Population == pop].resid.mean()), 3)
    bl = [rec[f"bias_{q}"] for q in analysis.POPS]
    rec["bias_spread"] = round(max(bl) - min(bl), 3)
    uniq, inv = np.unique(cd.SampleID.values, return_inverse=True)
    dpop = cd.groupby("SampleID").Population.first().reindex(uniq).values
    resid = cd.resid.values

    def spread(pv):
        m = [resid[pv[inv] == q].mean() for q in analysis.POPS]
        return max(m) - min(m)

    null = np.array([spread(rng.permutation(dpop)) for _ in range(N_PERM)])
    rec["p_perm"] = round(float((null >= rec["bias_spread"]).mean() + 1 / (N_PERM + 1)), 4)
    rhos = {}
    for pop in analysis.POPS:
        g = d[(d.Population == pop) & d._pred.notna() & d[col].notna()]
        rhos[pop] = (float(pearsonr(g._pred.rank(), g[col].rank()).statistic)
                     if len(g) > 100 and g._pred.std() > 0 and g[col].std() > 0 else np.nan)
    rec.update({f"rho_{k}": round(v, 3) if v == v else np.nan for k, v in rhos.items()})
    vals = [v for v in rhos.values() if v == v]
    rec["rho_spread"] = round(max(vals) - min(vals), 3) if vals else np.nan
    return rec


def main():
    rng = np.random.default_rng(SEED)
    ot = analysis.load()
    var = ot[ot.overlapped_vars.notna()].copy()
    rows = []
    for aname, col in analysis.ASSAYS:
        d, F, sets = dm.build_frame(var, col)
        for m in MODELS:
            D = dm.fit_guide_cv(d, F, col, sets[m], "ridge")
            r = audit(D[D._pred.notna()].copy(), col, rng)
            if r:
                r.update(assay=aname, model=m)
                rows.append(r)
    T = pd.DataFrame(rows)
    T["p_BH"] = multipletests(T.p_perm, method="fdr_bh")[1].round(4)
    T["significant"] = T.p_BH < 0.05
    cols = ["assay", "model", "n", "MAE"] + [f"bias_{p}" for p in analysis.POPS] + \
           ["bias_spread", "p_perm", "p_BH", "significant", "rho_spread"]
    T = T[cols + [c for c in T.columns if c not in cols]]
    T.to_csv(OUT / "new_model_ancestry.csv", index=False)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
