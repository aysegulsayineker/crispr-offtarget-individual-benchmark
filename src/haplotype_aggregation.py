"""How should two haplotype scores become one donor score?

The released table reduces a donor's two haplotype scores to a single value by
taking their mean, i.e. an additive dosage model.  The alternatives are the
maximum (a dominant model: the more active allele sets the risk) and the
minimum.  The choice is invisible for homozygous donors and can only matter for
the 13.5% of rows that are heterozygous, so the effect on the pooled statistic
is diluted; the comparison below is therefore reported both among heterozygous
rows and over all rows.

Point estimates and bootstrap contrasts are computed on the SAME set of sites --
those at which all three rules are informative -- so the columns are mutually
consistent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analysis   # noqa: E402
import fastboot   # noqa: E402

OUT = ROOT / "results"
B = 300
SEED = 0
RULES = ("mean", "max", "min")


def _wp(d, c):
    return d.groupby("OT_name")[c].transform(lambda x: rankdata(x) / (len(x) + 1))


def _assoc(d, c, col):
    d = d[d[c].notna() & d[col].notna()]
    d = d[d.groupby("OT_name")[c].transform("std") > 1e-9]
    if d.OT_name.nunique() < 5 or len(d) < 50:
        return np.nan
    return float(pearsonr(_wp(d, c), _wp(d, col)).statistic)


def main():
    rng = np.random.default_rng(SEED)
    ot = analysis.load()
    var = ot[ot.overlapped_vars.notna()].copy()
    rows = []
    for aname, col in analysis.ASSAYS:
        base = var[var[col].notna()].copy()
        for scope, d0 in (("heterozygous rows", base[base.h1 != base.h2]),
                          ("all rows", base)):
            for p in analysis.PREDS:
                d = d0.copy()
                for r in RULES:
                    d["_" + r] = d[[f"h1_{p}", f"h2_{p}"]].agg(r, axis=1)
                rec = {"assay": aname, "scope": scope, "model": p}
                sites_keep = sorted(set.intersection(*[
                    {s for s, g in d.groupby("OT_name")
                     if g["_" + r].notna().sum() >= 2 and g["_" + r].std() > 1e-9
                     and g[col].std() > 0} for r in RULES]))
                dk = d[d.OT_name.isin(sites_keep)]
                vecs = {r: fastboot.site_vectors(dk, "_" + r, col) for r in RULES}
                rec["n_rows"] = int(len(dk))
                rec["n_sites"] = int(len(sites_keep))
                for r in RULES:                    # point estimates on the SAME sites
                    a = fastboot.pooled_r(vecs[r])
                    rec[r] = round(a, 4) if a == a else np.nan
                for r in ("max", "min"):
                    m, lo, hi, pv = fastboot.boot_diff(vecs["mean"], vecs[r], rng, B=B)
                    rec[f"{r}_minus_mean"] = round(m, 4) if m == m else np.nan
                    rec[f"{r}_ci"] = (f"{lo:.3f} to {hi:.3f}" if lo == lo else "")
                    rec[f"{r}_p"] = round(pv, 4) if pv == pv else np.nan
                best = max(RULES, key=lambda r: (rec[r] if rec[r] == rec[r] else -9))
                rec["best_rule"] = best
                rows.append(rec)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "haplotype_aggregation.csv", index=False)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
