"""
Individual-level CRISPR off-target prediction: noise-ceiling-aware benchmark,
ancestry-stratified calibration, and conformal risk control.

Reproduces every number reported in the manuscript from public data.

Data (downloaded by src/download_data.sh):
  GEO GSE346736 : GSM10039759_All_1115sites_95_donor.csv.gz
  GEO GSE286299 : GSE286299_rhamp_seq_merged_table.csv.gz
  1000 Genomes  : integrated_call_samples_v3.20130502.ALL.panel

Source study (data origin): Population-scale cellular GUIDE-seq-2 and biochemical
CHANGE-seq-R profiles reveal human genetic variation frequently affects Cas9
off-target activity. bioRxiv 2025. PMID 39990392, doi:10.1101/2025.02.10.637517
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from statsmodels.stats.multitest import multipletests

DATA = Path("data"); OUT = Path("results"); OUT.mkdir(exist_ok=True)
PREDS = ["CHANGE-net", "CFD", "CRISOT", "CRISPRoff", "CRISPRnet"]
POPS = ["CEU", "CHB", "ASW", "MXL"]
ASSAYS = [("GUIDE-seq-2", "log2_normRC"), ("rhAmpSeq", "indel%")]
ALPHA = 0.10
SEED = 0


def load():
    ot = pd.read_csv(DATA / "GSM10039759_All_1115sites_95_donor.csv.gz")
    ot.columns = [c.lstrip("\ufeff#") for c in ot.columns]
    return ot


# ---------------------------------------------------------------- integrity
def integrity(ot):
    panel = pd.read_csv(DATA / "panel.txt", sep="\t", usecols=[0, 1, 2],
                        names=["sample", "pop", "super_pop"], header=0)
    j = (ot[["SampleID", "Population"]].drop_duplicates()
         .merge(panel, left_on="SampleID", right_on="sample", how="left"))
    has_indel = ((ot.h1.astype(str).str.len() != 23) | (ot.h2.astype(str).str.len() != 23)
                 | ot.h1.astype(str).str.contains("-") | ot.h2.astype(str).str.contains("-"))
    rows = [("n_rows", len(ot)), ("n_donors", ot.SampleID.nunique()),
            ("n_sites", ot.OT_name.nunique()), ("n_gRNAs", ot.gRNA.nunique()),
            ("donors_matched_1000G", int(j["pop"].notna().sum())),
            ("population_label_conflicts", int((j["pop"].notna() & (j["pop"] != j.Population)).sum())),
            ("variant_overlap_rows", int(ot.overlapped_vars.notna().sum())),
            ("indel_bearing_rows", int(has_indel.sum()))]
    for p in PREDS:
        rows.append((f"missing_{p}", int(ot[p].isna().sum())))
        rows.append((f"missing_{p}_all_indel_bearing",
                     bool((ot[p].isna() & ~has_indel).sum() == 0)))
    pd.DataFrame(rows, columns=["check", "value"]).to_csv(OUT / "integrity_checks.csv", index=False)
    return j


# -------------------------------------------------- within-site decomposition
def within_rho(d, p, col):
    d = d[d[p].notna() & d[col].notna()]
    if len(d) < 30:
        return np.nan
    pc = d[p] - d.groupby("OT_name")[p].transform("mean")
    mc = d[col] - d.groupby("OT_name")[col].transform("mean")
    keep = d.groupby("OT_name")[p].transform("std") > 1e-9
    if keep.sum() < 30:
        return np.nan
    return float(spearmanr(pc[keep], mc[keep]).statistic)


def boot_ci(d, p, col, rng, B=300):
    sites = d.OT_name.unique(); idx = {s: g for s, g in d.groupby("OT_name")}
    vals = []
    for _ in range(B):
        samp = rng.choice(sites, size=len(sites), replace=True)
        dd = pd.concat([idx[s].assign(OT_name=f"{s}__{i}") for i, s in enumerate(samp)],
                       ignore_index=True)
        v = within_rho(dd, p, col)
        if v == v:
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


# ------------------------------------------------- noise ceiling (ANOVA VC)
def vc_ceiling(d, col):
    """Unbiased ANOVA variance components WITHIN sites.
    Donors sharing a haplotype pair receive identical predictions, so their
    measured spread estimates irreducible noise."""
    d = d[d[col].notna()].copy()
    d["hap"] = d.h1.astype(str) + "|" + d.h2.astype(str)
    SSW = dfW = SSB = 0.0; n0_num = n0_den = 0.0; k_tot = 0
    for _, g in d.groupby("OT_name"):
        cls = g.groupby("hap")[col]; k = cls.ngroups; n = len(g)
        if k < 2 or n - k < 1:
            continue
        SSW += float(((g[col] - cls.transform("mean")) ** 2).sum()); dfW += n - k
        SSB += float((cls.size() * (cls.mean() - g[col].mean()) ** 2).sum())
        n0_num += n - (cls.size() ** 2).sum() / n; n0_den += k - 1; k_tot += k - 1
    if dfW < 1 or k_tot < 1:
        return None
    MSW, MSB, n0 = SSW / dfW, SSB / k_tot, n0_num / n0_den
    s2_hap = max(0.0, (MSB - MSW) / n0)
    frac = s2_hap / (s2_hap + MSW)
    return dict(sigma2_hap=s2_hap, sigma2_err=MSW, explainable=frac,
                ceiling_rho=float(np.sqrt(frac)))


def benchmark(var, rng):
    rows = []
    for aname, col in ASSAYS:
        d0 = var[var[col].notna()]
        ceil = vc_ceiling(var, col)
        sites = var.OT_name.unique(); idx = {s: g for s, g in var.groupby("OT_name")}
        bs = []
        for _ in range(300):
            samp = rng.choice(sites, size=len(sites), replace=True)
            dd = pd.concat([idx[s].assign(OT_name=f"{s}__{i}") for i, s in enumerate(samp)],
                           ignore_index=True)
            r = vc_ceiling(dd, col)
            if r:
                bs.append(r["ceiling_rho"])
        for p in PREDS:
            r = within_rho(d0, p, col); lo, hi = boot_ci(d0, p, col, rng)
            d = d0[d0[p].notna()]
            pc = d[p] - d.groupby("OT_name")[p].transform("mean")
            mc = d[col] - d.groupby("OT_name")[col].transform("mean")
            keep = d.groupby("OT_name")[p].transform("std") > 1e-9
            pear = float(pearsonr(pc[keep], mc[keep]).statistic)
            rows.append({"assay": aname, "model": p,
                         "rho_pooled": round(float(spearmanr(d[p], d[col]).statistic), 3),
                         "rho_within": round(r, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                         "pearson_within": round(pear, 3),
                         "partial_R2_pct": round(100 * pear ** 2, 2),
                         "explainable_var": round(ceil["explainable"], 3),
                         "ceiling_rho": round(ceil["ceiling_rho"], 3),
                         "ceiling_lo": round(float(np.percentile(bs, 2.5)), 3),
                         "ceiling_hi": round(float(np.percentile(bs, 97.5)), 3),
                         "pct_of_ceiling": round(100 * r / ceil["ceiling_rho"], 1)})
    out = pd.DataFrame(rows); out.to_csv(OUT / "benchmark_noise_ceiling.csv", index=False)
    return out


# ------------------------------------------------ ancestry calibration + test
def crossfit_calibrate(d, p, col, n_splits=5):
    d = d[d[p].notna() & d[col].notna()].copy()
    if d.OT_name.nunique() < n_splits:
        return None
    d["oof"] = np.nan
    for tr, te in GroupKFold(n_splits=n_splits).split(d, groups=d.OT_name):
        iso = IsotonicRegression(out_of_bounds="clip").fit(d.iloc[tr][p].values, d.iloc[tr][col].values)
        d.iloc[te, d.columns.get_loc("oof")] = iso.predict(d.iloc[te][p].values)
    d["resid"] = d[col] - d.oof
    return d


def calibration(var, rng):
    rows = []
    for aname, col in ASSAYS:
        base = var[var[col].notna()]
        for p in PREDS:
            cd = crossfit_calibrate(base, p, col)
            if cd is None:
                continue
            rec = {"assay": aname, "model": p, "n": len(cd), "MAE": round(float(cd.resid.abs().mean()), 3)}
            for pop in POPS:
                rec[f"bias_{pop}"] = round(float(cd[cd.Population == pop].resid.mean()), 3)
            bl = [rec[f"bias_{q}"] for q in POPS]; rec["bias_spread"] = round(max(bl) - min(bl), 3)
            # permutation test: donors (whole) reassigned to ancestry labels
            uniq, inv = np.unique(cd.SampleID.values, return_inverse=True)
            dpop = cd.groupby("SampleID").Population.first().reindex(uniq).values
            resid = cd.resid.values
            spread = lambda pv: (lambda m: max(m) - min(m))([resid[pv[inv] == q].mean() for q in POPS])
            null = np.array([spread(rng.permutation(dpop)) for _ in range(2000)])
            rec["p_perm"] = round(float((null >= rec["bias_spread"]).mean() + 1 / 2001), 4)
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["p_BH"] = multipletests(out.p_perm, method="fdr_bh")[1].round(4)
    out.to_csv(OUT / "calibration_by_ancestry.csv", index=False)
    return out


# ------------------------------------------------------- conformal prediction
def conformal(var, n_rep=20):
    rows = []
    for aname, col in ASSAYS:
        d_all = var[var[col].notna()]
        for p in PREDS:
            d = d_all[d_all[p].notna()].copy()
            sites = d.OT_name.unique(); rs = np.random.default_rng(SEED)
            for rep in range(n_rep):
                sh = rs.permutation(sites); n = len(sh)
                tr = d[d.OT_name.isin(set(sh[:n // 3]))]
                cal = d[d.OT_name.isin(set(sh[n // 3:2 * n // 3]))]
                te = d[d.OT_name.isin(set(sh[2 * n // 3:]))]
                if min(len(tr), len(cal), len(te)) < 200:
                    continue
                iso = IsotonicRegression(out_of_bounds="clip").fit(tr[p].values, tr[col].values)
                cs = np.abs(cal[col].values - iso.predict(cal[p].values))
                err = np.abs(te[col].values - iso.predict(te[p].values))
                q = np.quantile(cs, min(1.0, np.ceil((len(cs) + 1) * (1 - ALPHA)) / len(cs)), method="higher")
                rows.append({"assay": aname, "model": p, "rep": rep, "method": "marginal",
                             "ancestry": "ALL", "coverage": float((err <= q).mean()), "width": float(2 * q)})
                for pop in POPS:
                    mt = te.Population.values == pop
                    if mt.sum() >= 20:
                        rows.append({"assay": aname, "model": p, "rep": rep, "method": "marginal",
                                     "ancestry": pop, "coverage": float((err[mt] <= q).mean()),
                                     "width": float(2 * q)})
                    mc = cal.Population.values == pop
                    if mc.sum() >= 50 and mt.sum() >= 20:
                        csp = cs[mc]
                        qp = np.quantile(csp, min(1.0, np.ceil((len(csp) + 1) * (1 - ALPHA)) / len(csp)),
                                         method="higher")
                        rows.append({"assay": aname, "model": p, "rep": rep, "method": "mondrian",
                                     "ancestry": pop, "coverage": float((err[mt] <= qp).mean()),
                                     "width": float(2 * qp)})
    cf = pd.DataFrame(rows)
    summ = (cf.groupby(["assay", "model", "method", "ancestry"])
            .agg(coverage=("coverage", "mean"), cov_sd=("coverage", "std"),
                 width=("width", "mean")).reset_index().round(3))
    summ.to_csv(OUT / "conformal_coverage.csv", index=False)
    return summ


# ------------------------------------------------------------ decision rules
def decision_rules(var, col="indel%", taus=(0.1, 0.5, 1, 2, 5, 10, 20, 40), n_rep=20):
    base = var[var[col].notna()]; rows = []
    for p in PREDS:
        d = base[base[p].notna()].copy(); sites = d.OT_name.unique()
        rs = np.random.default_rng(1)
        for rep in range(n_rep):
            sh = rs.permutation(sites); n = len(sh)
            tr = d[d.OT_name.isin(set(sh[:n // 3]))]
            cal = d[d.OT_name.isin(set(sh[n // 3:2 * n // 3]))]
            te = d[d.OT_name.isin(set(sh[2 * n // 3:]))]
            if min(len(tr), len(cal), len(te)) < 200:
                continue
            iso = IsotonicRegression(out_of_bounds="clip").fit(tr[p].values, tr[col].values)
            cs = np.abs(cal[col].values - iso.predict(cal[p].values))
            q = np.quantile(cs, min(1.0, np.ceil((len(cs) + 1) * (1 - ALPHA)) / len(cs)), method="higher")
            pt = iso.predict(te[p].values)
            for tau in taus:
                truth = te[col].values > tau
                for rule, flag in [("point", pt > tau), ("conformal_upper", pt + q > tau)]:
                    for pop in POPS + ["ALL"]:
                        m = np.ones(len(te), bool) if pop == "ALL" else (te.Population.values == pop)
                        if m.sum() < 20 or truth[m].sum() < 1:
                            continue
                        rows.append({"model": p, "tau": tau, "rule": rule, "ancestry": pop, "rep": rep,
                                     "sensitivity": float(flag[m][truth[m]].mean()),
                                     "flag_rate": float(flag[m].mean())})
    de = pd.DataFrame(rows)
    (de.groupby(["model", "tau", "rule", "ancestry"])[["sensitivity", "flag_rate"]]
       .mean().round(3).reset_index().to_csv(OUT / "decision_rule_eval.csv", index=False))
    return de


def ancestry_discrimination(var):
    """Within-ancestry DISCRIMINATION (pooled Spearman per population).
    Complements the calibration analysis: tests whether ranking ability differs by ancestry."""
    rows = []
    for aname, col in ASSAYS:
        d0 = var[var[col].notna()]
        for p in PREDS:
            rec = {"assay": aname, "model": p}
            vals = []
            for pop in POPS:
                s = d0[(d0.Population == pop) & d0[p].notna()]
                r = float(spearmanr(s[p], s[col]).statistic) if len(s) > 30 else np.nan
                rec[pop] = round(r, 3); rec[f"n_{pop}"] = len(s)
                if r == r:
                    vals.append(r)
            rec["rho_spread"] = round(max(vals) - min(vals), 3) if vals else np.nan
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "discrimination_by_ancestry.csv", index=False)
    return out


def measured_descriptives(var):
    """Distribution of the measured outcomes — needed to interpret conformal interval widths."""
    rows = []
    for aname, col in ASSAYS:
        s = var[col].dropna()
        rows.append({"assay": aname, "column": col, "n": len(s),
                     "median": round(float(s.median()), 4),
                     "q25": round(float(s.quantile(.25)), 4),
                     "q75": round(float(s.quantile(.75)), 4),
                     "min": round(float(s.min()), 4), "max": round(float(s.max()), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "measured_descriptives.csv", index=False)
    return out


def main():
    rng = np.random.default_rng(SEED)
    ot = load(); integrity(ot)
    var = ot[ot.overlapped_vars.notna()].copy()     # variant-overlapping rows
    print("benchmark ...");   print(benchmark(var, rng).to_string(index=False))
    print("ancestry discrimination ...")
    print(ancestry_discrimination(var)[["assay","model"]+POPS+["rho_spread"]].to_string(index=False))
    print("measured descriptives ..."); print(measured_descriptives(var).to_string(index=False))
    print("calibration ..."); print(calibration(var, rng).to_string(index=False))
    print("conformal ...");   conformal(var)
    print("decision rules ..."); decision_rules(var)
    print(f"\nAll result tables written to {OUT}/")


if __name__ == "__main__":
    main()
