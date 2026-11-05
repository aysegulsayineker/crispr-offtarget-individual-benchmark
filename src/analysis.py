"""
Individual-level CRISPR off-target prediction: noise-ceiling-aware benchmark,
ancestry-stratified calibration, and conformal risk control.

Reproduces every number reported in the manuscript from public data.

Data (downloaded by src/download_data.sh):
  GEO GSE346736 : GSM10039759_All_1115sites_95_donor.csv.gz
  GEO GSE286299 : GSE286299_rhamp_seq_merged_table.csv.gz
  1000 Genomes  : integrated_call_samples_v3.20130502.ALL.panel

Source study (data origin): Flory AR, Lazzarotto CR, Li Y, et al. Frequent,
context-dependent effects of human genetic variation on Cas9 activity revealed by
population-scale GUIDE-seq-2 and deep combinatorial CHANCE-seq profiling. bioRxiv
(preprint) 2026. PMID 39990392, doi:10.1101/2025.02.10.637517
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, rankdata
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
    d["hap"] = canon_hap(d)
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


# ------------------------------------- metric-consistent attainment (primary)
def canon_hap(d):
    """Haplotype-pair label that is invariant to the order of the two haplotypes.

    A diploid genotype has no intrinsic ordering of its two chromosomes, so h1|h2 and
    h2|h1 are the same class. Grouping on the stored order splits 126 genuine classes
    in two in this dataset (603 ordered vs 477 canonical) and biases the ceiling.
    """
    a = d["h1"].fillna("NA").map(str).tolist()
    b = d["h2"].fillna("NA").map(str).tolist()
    return ["|".join(sorted([x, y])) for x, y in zip(a, b)]


def _within_pct(d, col):
    """Within-site percentile transform: rank/(n+1) inside each off-target site."""
    return d.groupby("OT_name")[col].transform(lambda x: rankdata(x) / (len(x) + 1))


def vc_ceiling_on(d, value_col):
    """ANOVA variance components within site, computed on an arbitrary value column."""
    d = d[d[value_col].notna()].copy()
    d["hap"] = canon_hap(d)
    SSW = dfW = SSB = 0.0; n0_num = n0_den = 0.0; k_tot = 0
    for _, g in d.groupby("OT_name"):
        cls = g.groupby("hap")[value_col]; k = cls.ngroups; n = len(g)
        if k < 2 or n - k < 1:
            continue
        SSW += float(((g[value_col] - cls.transform("mean")) ** 2).sum()); dfW += n - k
        SSB += float((cls.size() * (cls.mean() - g[value_col].mean()) ** 2).sum())
        n0_num += n - (cls.size() ** 2).sum() / n; n0_den += k - 1; k_tot += k - 1
    if dfW < 1 or k_tot < 1:
        return None
    MSW, MSB, n0 = SSW / dfW, SSB / k_tot, n0_num / n0_den
    s2 = max(0.0, (MSB - MSW) / n0)
    frac = s2 / (s2 + MSW)
    return dict(explainable=frac, ceiling=float(np.sqrt(frac)))


def attainment(var, rng, B=300):
    """Primary benchmark, on a single consistent scale and a single consistent sample.

    For each model the association and its ceiling are computed on EXACTLY the rows that
    model can score and that are informative for it (sites where its prediction varies
    across donors). Models drop very different numbers of sites -- CRISOT loses 24% of
    sites on GUIDE-seq-2, CHANGE-net 1.2% -- so a ceiling computed on the union would
    flatter the models that drop most.

    Both quantities are on the within-site percentile scale. The ratio's confidence
    interval comes from a joint bootstrap: association and ceiling are recomputed on the
    SAME resampled sites, so the interval reflects the uncertainty of the ratio itself
    rather than of its parts.

    A site-averaged association (one Spearman per site, averaged with equal weight) is
    reported alongside, because the pooled statistic weights sites by their row count.
    """
    rows = []
    for aname, col in ASSAYS:
        base = var[var[col].notna()].copy()
        base["hap"] = canon_hap(base)
        for p in PREDS:
            d = base[base[p].notna()].copy()
            n_sites_scored = d.OT_name.nunique()
            d = d[d.groupby("OT_name")[p].transform("std") > 1e-9]
            if len(d) < 30:
                continue
            n_sites_used = d.OT_name.nunique()

            def _stats(dd):
                dd = dd.copy()
                dd["_mp"] = _within_pct(dd, col); dd["_pp"] = _within_pct(dd, p)
                r = float(pearsonr(dd["_pp"], dd["_mp"]).statistic)
                c = vc_ceiling_on(dd, "_mp")
                return r, (c["ceiling"] if c else np.nan)

            r_obs, c_obs = _stats(d)

            # site-averaged (equal weight per site)
            per = []
            for _, g in d.groupby("OT_name"):
                if len(g) >= 5 and g[p].std() > 1e-9 and g[col].std() > 1e-9:
                    rr = spearmanr(g[p], g[col]).statistic
                    if rr == rr:
                        per.append(rr)

            # joint bootstrap over sites: ratio recomputed on each resample
            sv = d.OT_name.unique(); ix = {s: g for s, g in d.groupby("OT_name")}
            ratios, assocs = [], []
            for _ in range(B):
                samp = rng.choice(sv, size=len(sv), replace=True)
                dd = pd.concat([ix[s].assign(OT_name=f"{s}__{i}") for i, s in enumerate(samp)],
                               ignore_index=True)
                rb, cb = _stats(dd)
                assocs.append(rb)
                if cb == cb and cb > 0:
                    ratios.append(100 * rb / cb)

            rows.append({
                "assay": aname, "model": p,
                "n_rows": len(d), "n_sites_scored": n_sites_scored, "n_sites_used": n_sites_used,
                "n_sites_dropped_constant": n_sites_scored - n_sites_used,
                "assoc": round(r_obs, 3),
                "assoc_lo": round(float(np.percentile(assocs, 2.5)), 3),
                "assoc_hi": round(float(np.percentile(assocs, 97.5)), 3),
                "ceiling_matched": round(c_obs, 3),
                "pct_of_ceiling": round(100 * r_obs / c_obs, 1),
                "pct_lo": round(float(np.percentile(ratios, 2.5)), 1),
                "pct_hi": round(float(np.percentile(ratios, 97.5)), 1),
                "site_avg_rho": round(float(np.mean(per)), 3) if per else np.nan,
                "site_avg_se": round(float(np.std(per, ddof=1) / np.sqrt(len(per))), 3) if len(per) > 1 else np.nan,
                "n_sites_site_avg": len(per)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "attainment_metric_consistent.csv", index=False)
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
def conformal(var, n_rep=20, unit="site"):
    """Split conformal with the exchangeable unit made explicit.

    The data are a donor x site grid, so rows are not exchangeable: a donor recurs across
    sites and a site recurs across donors. Split conformal's finite-sample guarantee needs
    the calibration and test units to be exchangeable, which holds for whichever margin the
    split is taken on -- not for rows.

      unit="site"  : target is a NEW SITE in the same donors. Whole sites are held out.
      unit="donor" : target is a NEW DONOR at the same sites. Whole donors are held out.

    Both are reported; they answer different clinical questions and neither licenses a
    row-level claim.
    """
    rows = []
    for aname, col in ASSAYS:
        d_all = var[var[col].notna()]
        for p in PREDS:
            d = d_all[d_all[p].notna()].copy()
            key = "OT_name" if unit == "site" else "SampleID"
            sites = d[key].unique(); rs = np.random.default_rng(SEED)
            for rep in range(n_rep):
                sh = rs.permutation(sites); n = len(sh)
                tr = d[d[key].isin(set(sh[:n // 3]))]
                cal = d[d[key].isin(set(sh[n // 3:2 * n // 3]))]
                te = d[d[key].isin(set(sh[2 * n // 3:]))]
                if min(len(tr), len(cal), len(te)) < 200:
                    continue
                iso = IsotonicRegression(out_of_bounds="clip").fit(tr[p].values, tr[col].values)
                cs = np.abs(cal[col].values - iso.predict(cal[p].values))
                err = np.abs(te[col].values - iso.predict(te[p].values))
                q = np.quantile(cs, min(1.0, np.ceil((len(cs) + 1) * (1 - ALPHA)) / len(cs)), method="higher")
                rows.append({"assay": aname, "model": p, "rep": rep, "unit": unit, "method": "marginal",
                             "ancestry": "ALL", "coverage": float((err <= q).mean()), "width": float(2 * q)})
                for pop in POPS:
                    mt = te.Population.values == pop
                    if mt.sum() >= 20:
                        rows.append({"assay": aname, "model": p, "rep": rep, "unit": unit, "method": "marginal",
                                     "ancestry": pop, "coverage": float((err[mt] <= q).mean()),
                                     "width": float(2 * q)})
                    mc = cal.Population.values == pop
                    if mc.sum() >= 50 and mt.sum() >= 20:
                        csp = cs[mc]
                        qp = np.quantile(csp, min(1.0, np.ceil((len(csp) + 1) * (1 - ALPHA)) / len(csp)),
                                         method="higher")
                        rows.append({"assay": aname, "model": p, "rep": rep, "unit": unit, "method": "mondrian",
                                     "ancestry": pop, "coverage": float((err[mt] <= qp).mean()),
                                     "width": float(2 * qp)})
    cf = pd.DataFrame(rows)
    summ = (cf.groupby(["assay", "model", "unit", "method", "ancestry"])
            .agg(coverage=("coverage", "mean"), cov_sd=("coverage", "std"),
                 width=("width", "mean")).reset_index().round(3))
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



def paired_bootstrap(var, B=400, seed=11):
    """Paired site-level bootstrap of the DIFFERENCE between two models.

    Both models are evaluated on the SAME resampled sites, so the comparison is
    paired and the interval reflects the uncertainty of the difference rather
    than of each estimate separately.
    """
    import itertools
    rng = np.random.default_rng(seed)
    out = []
    for aname, col in ASSAYS:
        base = var[var[col].notna()].copy()
        sites = base.OT_name.unique()
        ix = {s: g for s, g in base.groupby("OT_name")}
        draws = {p: [] for p in PREDS}
        for _ in range(B):
            samp = rng.choice(sites, size=len(sites), replace=True)
            dd = pd.concat([ix[s].assign(OT_name=f"{s}__{i}") for i, s in enumerate(samp)],
                           ignore_index=True)
            for p in PREDS:
                d = dd[dd[p].notna()]
                d = d[d.groupby("OT_name")[p].transform("std") > 1e-9]
                draws[p].append(float(pearsonr(_within_pct(d, p), _within_pct(d, col)).statistic)
                                if len(d) > 30 else np.nan)
        for a, b in itertools.combinations(PREDS, 2):
            diff = np.array(draws[a]) - np.array(draws[b])
            diff = diff[~np.isnan(diff)]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            pv = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
            out.append({"assay": aname, "pair": f"{a} - {b}",
                        "mean_diff": round(float(diff.mean()), 4),
                        "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                        "p_boot": round(float(min(pv, 1.0)), 4)})
    pdiff = pd.DataFrame(out)
    pdiff["p_BH"] = multipletests(pdiff.p_boot, method="fdr_bh")[1].round(4)
    pdiff["significant"] = pdiff.p_BH < 0.05
    pdiff.to_csv(OUT / "paired_model_differences.csv", index=False)
    return pdiff


def _group_conformal_one(var, col, p, unit="SampleID", n_rep=20, seed=3):
    """One conformity score per independent unit (max |residual| over its rows).

    This is the version that carries a genuine finite-sample guarantee: it
    covers ALL rows of a new unit simultaneously with probability 1 - ALPHA.
    """
    d = var[var[col].notna() & var[p].notna()].copy()
    units = d[unit].unique()
    rs = np.random.default_rng(seed)
    rows = []
    for rep in range(n_rep):
        sh = rs.permutation(units); n = len(sh)
        tr = d[d[unit].isin(set(sh[:n // 3]))]
        cal = d[d[unit].isin(set(sh[n // 3:2 * n // 3]))]
        te = d[d[unit].isin(set(sh[2 * n // 3:]))]
        if min(len(tr), len(cal), len(te)) < 200:
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(tr[p].values, tr[col].values)
        cal = cal.assign(res=np.abs(cal[col].values - iso.predict(cal[p].values)))
        te = te.assign(res=np.abs(te[col].values - iso.predict(te[p].values)))
        s_cal = cal.groupby(unit).res.max().values
        m = len(s_cal)
        q = np.quantile(s_cal, min(1.0, np.ceil((m + 1) * (1 - ALPHA)) / m), method="higher")
        per_unit = te.groupby(unit).res.max()
        rows.append({"rep": rep, "n_cal_units": m,
                     "unit_coverage": float((per_unit <= q).mean()),
                     "row_coverage": float((te.res <= q).mean()),
                     "width": float(2 * q)})
    return pd.DataFrame(rows)


def group_conformal(var):
    res = []
    for aname, col in ASSAYS:
        for p in PREDS:
            for unit, lbl in [("SampleID", "donor"), ("OT_name", "site")]:
                r = _group_conformal_one(var, col, p, unit=unit)
                if len(r):
                    res.append({"assay": aname, "model": p, "unit": lbl,
                                "n_cal_units": int(r.n_cal_units.iloc[0]),
                                "unit_coverage": round(r.unit_coverage.mean(), 3),
                                "row_coverage": round(r.row_coverage.mean(), 3),
                                "width": round(r.width.mean(), 2)})
    g = pd.DataFrame(res)
    g.to_csv(OUT / "group_conformal.csv", index=False)
    return g


def per_guide(var):
    """Within-site association per guide RNA.

    Reported on the primary within-site percentile scale, and -- as a scale
    sensitivity check -- on the mean-centred raw scale used by benchmark().
    The two can differ in sign at a guide, because mean-centring lets sites
    with large prediction variance dominate while the percentile transform
    weights every site equally.
    """
    rows = []
    for aname, col in ASSAYS:
        for g, gd in var[var[col].notna()].groupby("gRNA"):
            rec = {"assay": aname, "gRNA": g,
                   "n_sites": int(gd.OT_name.nunique()), "n_rows": int(len(gd))}
            for p in PREDS:
                d = gd[gd[p].notna()]
                dc = d[d.groupby("OT_name")[p].transform("std") > 1e-9]
                rec[p] = (round(float(pearsonr(_within_pct(dc, p),
                                               _within_pct(dc, col)).statistic), 3)
                          if len(dc) > 30 else np.nan)
                rec[p + "_rawscale"] = (round(float(within_rho(d, p, col)), 3)
                                        if len(d) > 30 else np.nan)
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "per_guide_within_rho.csv", index=False)
    return out


def main():
    rng = np.random.default_rng(SEED)
    ot = load(); integrity(ot)
    var = ot[ot.overlapped_vars.notna()].copy()     # variant-overlapping rows
    print("benchmark ...");   print(benchmark(var, rng).to_string(index=False))
    print("attainment (metric-consistent) ...")
    print(attainment(var, rng)[["assay","model","n_sites_used","n_sites_dropped_constant","assoc","assoc_lo","assoc_hi","ceiling_matched","pct_of_ceiling","pct_lo","pct_hi","site_avg_rho"]].to_string(index=False))
    print("ancestry discrimination ...")
    print(ancestry_discrimination(var)[["assay","model"]+POPS+["rho_spread"]].to_string(index=False))
    print("measured descriptives ..."); print(measured_descriptives(var).to_string(index=False))
    print("calibration ..."); print(calibration(var, rng).to_string(index=False))
    print("conformal ...")
    cf = pd.concat([conformal(var, unit="site"), conformal(var, unit="donor")],
                   ignore_index=True)
    cf.to_csv(OUT / "conformal_coverage.csv", index=False)
    print(cf[(cf.ancestry == "ALL") & (cf.method == "marginal")]
          [["assay","model","unit","coverage","width"]].to_string(index=False))
    print("decision rules ..."); decision_rules(var)
    print("paired model differences ...")
    print(paired_bootstrap(var).to_string(index=False))
    print("group conformal ...");  print(group_conformal(var).to_string(index=False))
    print("per-guide within-site association ...")
    print(per_guide(var).to_string(index=False))
    print(f"\nAll result tables written to {OUT}/")


if __name__ == "__main__":
    main()
