"""Build the two manuscript tables from results/*.csv.

Emits, for each table, a CSV (results/), a markdown block and a LaTeX booktabs
block (results/tables.md, results/tables.tex). No number is typed by hand.
"""
import pandas as pd
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
ASSAYS = ["GUIDE-seq-2", "rhAmpSeq"]
MODELS = ["CHANGE-net", "CFD", "CRISOT", "CRISPRoff", "CRISPRnet"]


def _order(df):
    df = df.copy()
    df["assay"] = pd.Categorical(df["assay"], ASSAYS, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODELS, ordered=True)
    return df.sort_values(["assay", "model"]).reset_index(drop=True)


def table1():
    at = _order(pd.read_csv(RES / "attainment_metric_consistent.csv"))
    bm = pd.read_csv(RES / "benchmark_noise_ceiling.csv")[["assay", "model", "rho_pooled"]]
    d = at.merge(bm, on=["assay", "model"], how="left")
    out = pd.DataFrame({
        "Assay": d["assay"],
        "Model": d["model"],
        "Rows": d["n_rows"],
        "Sites used": d["n_sites_used"].astype(str) + "/" + d["n_sites_scored"].astype(str),
        "Pooled $\\rho$": d["rho_pooled"].map("{:.2f}".format),
        "Within-site $\\rho$": d.apply(lambda r: f"{r.assoc:.3f} ({r.assoc_lo:.3f}-{r.assoc_hi:.3f})", axis=1),
        "Ceiling": d["ceiling_matched"].map("{:.3f}".format),
        "% of ceiling": d.apply(lambda r: f"{r.pct_of_ceiling:.1f} ({r.pct_lo:.1f}-{r.pct_hi:.1f})", axis=1),
        "Site-avg $\\rho$": d.apply(lambda r: f"{r.site_avg_rho:.3f} ({r.site_avg_se:.3f})", axis=1),
    })
    return out


def table2():
    di = _order(pd.read_csv(RES / "discrimination_by_ancestry.csv"))[["assay", "model", "rho_spread"]]
    ca = pd.read_csv(RES / "calibration_by_ancestry.csv")[["assay", "model", "MAE", "bias_spread", "p_BH"]]
    cc = pd.read_csv(RES / "conformal_coverage.csv").query("ancestry=='ALL' and method=='marginal'")
    site = cc.query("unit=='site'")[["assay", "model", "coverage", "width"]].rename(
        columns={"coverage": "cov_site", "width": "w_site"})
    donor = cc.query("unit=='donor'")[["assay", "model", "coverage", "width"]].rename(
        columns={"coverage": "cov_donor", "width": "w_donor"})
    d = di.merge(ca, on=["assay", "model"]).merge(site, on=["assay", "model"]).merge(
        donor, on=["assay", "model"])
    d = _order(d)
    out = pd.DataFrame({
        "Assay": d["assay"],
        "Model": d["model"],
        "$\\Delta\\rho$": d["rho_spread"].map("{:.3f}".format),
        "MAE": d["MAE"].map("{:.2f}".format),
        "$\\Delta$bias": d["bias_spread"].map("{:.3f}".format),
        "$p_{\\mathrm{BH}}$": d["p_BH"].map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"),
        "Cov.~site": d["cov_site"].map("{:.3f}".format),
        "Cov.~donor": d["cov_donor"].map("{:.3f}".format),
        "W~site": d["w_site"].map("{:.2f}".format),
        "W~donor": d["w_donor"].map("{:.2f}".format),
    })
    return out


def table3():
    """Variant-aware and donor-aware models against the released scores."""
    P = pd.read_csv(RES / "delta_model_performance.csv")
    C = pd.read_csv(RES / "delta_model_contrasts.csv")
    lab = {"M1 baseline": "released scores (ensemble)",
           "M2 variant": "variant features",
           "M3 combined": "scores + variant",
           "M4 donor": "donor propensity",
           "M5 full": "scores + variant + donor"}
    rows = []
    for assay in ASSAYS:
        rel = P[(P.assay == assay) & (P.estimator == "-")]
        b = rel.loc[rel.assoc_guide_cv.idxmax()]
        rows.append({"Assay": assay,
                     "Predictor": "best released (" + b.model.replace("released ", "") + ")",
                     "Guide-held-out": f"{b.assoc_guide_cv:.3f}",
                     "Donor-held-out": f"{b.assoc_donor_cv:.3f}",
                     "% of ceiling": "", "vs released ensemble": ""})
        for m in ["M1 baseline", "M2 variant", "M3 combined", "M4 donor", "M5 full"]:
            r = P[(P.assay == assay) & (P.estimator == "ridge") & (P.model == m)]
            if not len(r):
                continue
            r = r.iloc[0]
            c = C[(C.assay == assay) & (C.estimator == "ridge")
                  & (C.contrast == f"{m} - M1 baseline")]
            if len(c):
                c = c.iloc[0]
                star = "*" if bool(c.significant) else ""
                contrast = f"{c.mean_diff:+.3f} ({c.ci_lo:+.3f}, {c.ci_hi:+.3f}){star}"
            else:
                contrast = "reference"
            rows.append({"Assay": "", "Predictor": lab[m],
                         "Guide-held-out": f"{r.assoc_guide_cv:.3f}",
                         "Donor-held-out": f"{r.assoc_donor_cv:.3f}",
                         "% of ceiling": f"{r.pct_of_ceiling:.1f}",
                         "vs released ensemble": contrast})
    return pd.DataFrame(rows)


CAP3 = (r"\textbf{A donor-level term, not better sequence features, recovers additional "
        r"between-donor signal.} Within-site rank association of each predictor of the "
        r"between-donor component, ridge estimator. Guide-held-out: leave-one-guide-out, "
        r"all 95 donors at a test site scored by one model, directly comparable to the "
        r"released scores in Table~\ref{tab1}. Donor-held-out: repeated grouped two-fold "
        r"on donors, evaluated within each held-out fold; the percentile transform has "
        r"fewer ties in a smaller donor set, so values run higher and are comparable only "
        r"within the column. \% of ceiling uses the model-matched haplotype-conditioned "
        r"ceiling. The last column is a paired site-level bootstrap difference against the "
        r"released-score ensemble with a 95\% interval; * marks Benjamini--Hochberg "
        r"significance at 0.05 over the 20 contrasts. The donor propensity is estimated "
        r"from the donor's measurements at other guides and uses no sequence information.")


def to_md(df):
    hdr = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([hdr, sep] + rows)


def to_tex(df, caption, label, colspec=None, small=True, group_header=None,
           tabcolsep=None, header_override=None):
    """LaTeX booktabs table.

    group_header: optional (spanning_row, cmidrule_row) placed above the column
    names, used to keep wide tables inside the text block.
    """
    def esc(s):
        s = str(s)
        if "$" in s or "~" in s:          # already LaTeX
            return s
        return s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
    cols = colspec or "@{}ll" + "r" * (len(df.columns) - 2) + "@{}"
    body = [r"\begin{table}[htbp]", r"\centering",
            r"\caption{" + caption + "}" + r"\label{" + label + "}"]
    if small:
        body.append(r"\footnotesize")
    if tabcolsep:
        body.append(r"\setlength{\tabcolsep}{" + tabcolsep + "}")
    body += [r"\begin{tabular}{" + cols + "}", r"\toprule"]
    if group_header:
        body += list(group_header)
    hdr = header_override or list(df.columns)
    body += [" & ".join(esc(c) for c in hdr) + r" \\", r"\midrule"]
    prev = None
    for r in df.itertuples(index=False):
        vals = list(r)
        if vals[0] == prev:
            vals[0] = ""
        else:
            if prev is not None:
                body.append(r"\midrule")
            prev = vals[0]
        body.append(" & ".join(esc(v) for v in vals) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body)


CAP1 = (r"\textbf{Pooled, within-site and ceiling-relative performance of five off-target "
        r"predictors.} Rows, scored donor $\times$ site observations on variant-overlapping "
        r"sites. Sites used / scored: sites retained after dropping those on which the model "
        r"returns a constant prediction across donors. Pooled $\rho$, Spearman correlation "
        r"across all scored rows. Within-site $\rho$, association after percentile-transforming "
        r"prediction and measurement inside each site (95\% CI from 300 site-level bootstrap "
        r"replicates). Ceiling, haplotype-conditioned ceiling computed on the same scale and the "
        r"same rows as that model's own association. \% of ceiling, their ratio with a 95\% "
        r"joint-bootstrap interval. Site-avg $\rho$, the site-averaged sensitivity analysis "
        r"(standard error in parentheses).")

CAP2 = (r"\textbf{Ancestry audit, conformal coverage and interval width.} $\rho$ spread, range "
        r"of pooled Spearman correlations across the four 1000 Genomes populations "
        r"(discrimination). MAE and bias spread, mean absolute error and range of mean signed "
        r"residuals across populations after cross-fitted isotonic recalibration; $p$ (BH), "
        r"Benjamini--Hochberg adjusted $p$-value of a 2{,}000-permutation donor-level test on "
        r"the bias spread. Cov., empirical coverage of nominal 90\% marginal split-conformal "
        r"intervals, pooled over populations, with the exchangeable unit taken to be the site or "
        r"the donor. Width, mean interval width in log$_2$ normalised read count (GUIDE-seq-2) "
        r"or indel percentage points (rhAmpSeq); the measured median indel fraction is 0.016\%.")


HEAD2 = ["Assay", "Model", r"$\Delta\rho$", "MAE", r"$\Delta$bias",
         r"$p_{\mathrm{BH}}$", "site", "donor", "site", "donor"]

GROUP2 = (
    r" &  & \multicolumn{4}{c}{Ancestry audit} & \multicolumn{2}{c}{Coverage}"
    r" & \multicolumn{2}{c}{Width} \\",
    r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
)


def main():
    t1, t2, t3 = table1(), table2(), table3()
    t1.to_csv(RES / "table1_performance.csv", index=False)
    t2.to_csv(RES / "table2_ancestry_conformal.csv", index=False)
    t3.to_csv(RES / "table3_variant_aware_models.csv", index=False)
    (RES / "tables.md").write_text(
        "**Table 1.** Pooled, within-site and ceiling-relative performance.\n\n" + to_md(t1) +
        "\n\n**Table 2.** Ancestry audit, conformal coverage and interval width.\n\n" + to_md(t2) +
        "\n\n**Table 3.** Variant-aware and donor-aware models.\n\n" + to_md(t3) + "\n")
    (RES / "tables.tex").write_text(
        to_tex(t1, CAP1, "tab1") + "\n\n" +
        to_tex(t2, CAP2, "tab2", group_header=GROUP2, tabcolsep="4pt",
               header_override=HEAD2) + "\n\n" +
        to_tex(t3, CAP3, "tab3", colspec="@{}llrrrl@{}", tabcolsep="5pt") + "\n")
    print(to_md(t1)); print(); print(to_md(t2)); print(); print(to_md(t3))


if __name__ == "__main__":
    main()
