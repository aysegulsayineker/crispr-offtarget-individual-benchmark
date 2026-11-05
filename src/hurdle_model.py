"""Two-part (hurdle) formulation of individual-level off-target activity.

Both outcomes are heavily zero-inflated: 77% of variant-overlapping GUIDE-seq-2
rows carry no detected tag-integration read, and the median measured rhAmpSeq
indel fraction is 0.016%.  A single-stage regression therefore spends its
capacity on a point mass.  The two stages ask different questions:

  Stage A   P(Y > LOD | X)          does measurable editing occur at all?
  Stage B   E[Y | Y > LOD, X]       given that it does, how much?
  combined  product of the two

Limit of detection.  Zero and "below the limit of detection" are not the same
measurement, so the hurdle is placed at an assay-defined limit rather than at
zero:

  GUIDE-seq-2   detection of any tag-integration read.  The readout is an
      integer count whose smallest non-zero value in this table is 2, so the
      threshold is a property of the assay pipeline, not a choice made here.
  rhAmpSeq      0.1% indels, the analytical threshold already used as the
      primary decision threshold in the screening-rule analysis, with 0.01%
      and 0.5% as a sensitivity analysis.  Non-zero values in this table reach
      1e-5%, i.e. one read in ten million, which is sequencing error rather
      than editing.

Evaluation of stage A is restricted to sites at which detection status varies
between donors; at a site where every donor is negative there is no
between-donor question to answer.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analysis            # noqa: E402
import delta_model as dm   # noqa: E402

OUT = ROOT / "results"
SEED = 0
LODS = {"GUIDE-seq-2": [("detected read", 0.0)],
        "rhAmpSeq": [("0.1% indels", 0.1), ("0.01% indels", 0.01), ("0.5% indels", 0.5)]}
MODELS = ["M1 baseline", "M3 combined", "M5 full"]


def _clf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingClassifier(
                             max_depth=4, max_iter=300, learning_rate=0.06,
                             min_samples_leaf=50, l2_regularization=1.0,
                             random_state=SEED))


def _reg():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(
                             max_depth=4, max_iter=300, learning_rate=0.06,
                             min_samples_leaf=50, l2_regularization=1.0,
                             random_state=SEED))


def guide_cv_two_part(d, F, col, cols, lod):
    """Leave-one-guide-out predictions from both stages and their product."""
    feats = F[cols].astype(float).copy()
    if "donor_site" in cols:
        feats["donor_site"] = F["donor_guide"].to_numpy()
    y = d[col].to_numpy(dtype=float)
    above = (y > lod).astype(int)
    pa = np.full(len(d), np.nan)
    pb = np.full(len(d), np.nan)
    for g in sorted(d.gRNA.unique()):
        tr = np.flatnonzero(d.gRNA.values != g)
        te = np.flatnonzero(d.gRNA.values == g)
        if above[tr].sum() < 50 or above[tr].sum() == len(tr):
            continue
        pa[te] = _clf().fit(feats.iloc[tr], above[tr]).predict_proba(feats.iloc[te])[:, 1]
        pos = tr[above[tr] == 1]
        if len(pos) >= 50:
            pb[te] = _reg().fit(feats.iloc[pos], np.log1p(y[pos])).predict(feats.iloc[te])
    return d.assign(_pa=pa, _pb=pb, _above=above,
                    _combined=pa * np.expm1(np.clip(pb, -20, 20)))


def evaluate(D, col, lod):
    """Stage A on sites where detection varies; stage B among above-LOD rows."""
    res = {}
    v = D.groupby("OT_name")._above.transform("nunique") > 1
    da = D[v & D._pa.notna()]
    if len(da) > 50 and da._above.nunique() > 1:
        res["n_rows_stageA"] = int(len(da))
        res["n_sites_stageA"] = int(da.OT_name.nunique())
        res["detection_rate"] = round(float(da._above.mean()), 4)
        res["auprc"] = round(float(average_precision_score(da._above, da._pa)), 4)
        res["auroc"] = round(float(roc_auc_score(da._above, da._pa)), 4)
        res["assoc_stageA"] = round(dm.assoc(da.assign(_p=da._pa), "_p", col), 4)
    db = D[(D._above == 1) & D._pb.notna()]
    if len(db) > 50:
        r = dm.assoc(db.assign(_p=db._pb), "_p", col)
        res["n_rows_stageB"] = int(len(db))
        res["assoc_stageB"] = round(r, 4) if r == r else np.nan
    dc = D[D._combined.notna()]
    r = dm.assoc(dc.assign(_p=dc._combined), "_p", col)
    res["assoc_combined"] = round(r, 4) if r == r else np.nan
    return res


def main():
    ot = analysis.load()
    var = ot[ot.overlapped_vars.notna()].copy()
    rows = []
    for aname, col in analysis.ASSAYS:
        d, F, sets = dm.build_frame(var, col)
        single = {m: dm.assoc(dm.fit_guide_cv(d, F, col, sets[m], "gbt"), "_pred", col)
                  for m in MODELS}
        for lod_name, lod in LODS[aname]:
            for m in MODELS:
                D = guide_cv_two_part(d, F, col, sets[m], lod)
                rec = {"assay": aname, "lod": lod_name, "model": m,
                       "assoc_single_stage": round(single[m], 4) if single[m] == single[m] else np.nan}
                rec.update(evaluate(D, col, lod))
                rows.append(rec)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "hurdle_model.csv", index=False)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
