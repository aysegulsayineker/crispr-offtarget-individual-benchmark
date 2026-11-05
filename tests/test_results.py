"""Smoke test: the analysis outputs exist and the headline numbers are unchanged.

Run after `python src/analysis.py` (or against the committed result tables):
    pytest -q tests
"""
from pathlib import Path

import pandas as pd
import pytest

RES = Path(__file__).resolve().parents[1] / "results"
EXPECTED_FILES = [
    "benchmark_noise_ceiling.csv", "attainment_metric_consistent.csv",
    "calibration_by_ancestry.csv",
    "conformal_coverage.csv", "decision_rule_eval.csv", "integrity_checks.csv",
]
MODELS = {"CHANGE-net", "CFD", "CRISOT", "CRISPRoff", "CRISPRnet"}
POPS = {"CEU", "CHB", "ASW", "MXL"}


@pytest.mark.parametrize("fname", EXPECTED_FILES)
def test_output_exists_and_is_nonempty(fname):
    p = RES / fname
    assert p.exists(), f"missing analysis output: {p}"
    assert len(pd.read_csv(p)) > 0


def test_cohort_integrity():
    chk = pd.read_csv(RES / "integrity_checks.csv").set_index("check").value
    assert int(chk["n_donors"]) == 95
    assert int(chk["n_sites"]) == 1115
    assert int(chk["n_gRNAs"]) == 6
    assert int(chk["population_label_conflicts"]) == 0
    # every unscored row carries an indel-bearing haplotype
    for m in MODELS:
        assert str(chk[f"missing_{m}_all_indel_bearing"]) == "True"
    assert int(chk["missing_CFD"]) == int(chk["indel_bearing_rows"]) == 11117


def test_raw_scale_sensitivity_analysis():
    b = pd.read_csv(RES / "benchmark_noise_ceiling.csv")
    assert set(b.model) == MODELS and set(b.assay) == {"GUIDE-seq-2", "rhAmpSeq"}
    g = b[b.assay == "GUIDE-seq-2"]
    r = b[b.assay == "rhAmpSeq"]
    # pooled correlations are moderate, within-site correlations are not
    assert b.rho_pooled.between(0.19, 0.31).all()
    assert g.rho_within.between(0.050, 0.095).all()
    assert r.rho_within.between(0.155, 0.177).all()
    # noise ceiling
    # raw-scale ceiling, computed with order-invariant haplotype classes
    assert g.explainable_var.eq(0.348).all() and g.ceiling_rho.eq(0.59).all()
    assert r.explainable_var.eq(0.810).all() and r.ceiling_rho.eq(0.90).all()
    # this raw-scale ratio mixes metrics and is reported only as a sensitivity analysis
    assert b.pct_of_ceiling.between(8.5, 19.7).all()
    # CHANGE-net is not separated from CFD: bootstrap CIs overlap in both assays
    for a in ("GUIDE-seq-2", "rhAmpSeq"):
        d = b[b.assay == a].set_index("model")
        assert d.loc["CHANGE-net", "ci_lo"] < d.loc["CFD", "ci_hi"]
        assert d.loc["CFD", "ci_lo"] < d.loc["CHANGE-net", "ci_hi"]


def test_ancestry_calibration_is_significant_everywhere():
    c = pd.read_csv(RES / "calibration_by_ancestry.csv")
    assert len(c) == 10
    assert (c.p_BH < 0.05).all(), "all 10 model x assay combinations should be significant"
    assert c.bias_spread.max() < c.MAE.min(), "bias spread must stay below total error"


def test_conformal_coverage_is_valid_in_every_ancestry_group():
    cc = pd.read_csv(RES / "conformal_coverage.csv")
    assert set(cc.method) == {"marginal", "mondrian"}
    assert POPS <= set(cc.ancestry)
    assert (cc.coverage - 0.90).abs().max() < 0.05
    # primary split is by site; the donor split is checked separately
    marg = cc[(cc.method == "marginal") & (cc.ancestry == "ALL") & (cc.unit == "site")]
    # intervals are valid but wide: ~8 log2 units and ~4.4 indel-% points on average
    assert 8.0 < marg[marg.assay == "GUIDE-seq-2"].width.mean() < 8.5
    assert 4.2 < marg[marg.assay == "rhAmpSeq"].width.mean() < 4.7


def test_decision_rules_fail_at_high_thresholds():
    de = pd.read_csv(RES / "decision_rule_eval.csv")
    pt = de[(de.rule == "point") & (de.ancestry == "ALL")]
    assert pt[pt.tau == 0.1].sensitivity.between(0.55, 0.71).all()
    # the conformal-upper rule attains perfect sensitivity only by flagging everything
    cu = de[(de.rule == "conformal_upper") & (de.ancestry == "ALL") & (de.tau == 0.1)]
    assert cu.sensitivity.eq(1.0).all() and cu.flag_rate.eq(1.0).all()
    # at the highest thresholds the point rule stops flagging true high-activity events
    assert pt[pt.tau >= 20].sensitivity.max() < 0.25
    assert (pt[(pt.tau >= 20) & (pt.model.isin(["CHANGE-net", "CRISPRoff"]))]
            .sensitivity.eq(0.0).all())


def test_primary_attainment_is_metric_and_sample_consistent():
    """Primary result: association and ceiling share a scale AND a row set."""
    a = pd.read_csv(RES / "attainment_metric_consistent.csv")
    g = a[a.assay == "GUIDE-seq-2"]; r = a[a.assay == "rhAmpSeq"]
    # model-matched ceilings vary between models by construction
    assert g.ceiling_matched.between(0.516, 0.549).all()
    assert r.ceiling_matched.between(0.433, 0.471).all()
    assert a.pct_of_ceiling.between(22.4, 36.8).all()
    assert (a.pct_of_ceiling < 50).all()
    # joint-bootstrap ratio intervals exist and overlap across all models
    assert (a.pct_lo < a.pct_of_ceiling).all() and (a.pct_hi > a.pct_of_ceiling).all()
    for assay in ("GUIDE-seq-2", "rhAmpSeq"):   # overlap is checked within an assay
        s = a[a.assay == assay]
        assert s.pct_lo.max() < s.pct_hi.min(), f"ratio intervals should overlap in {assay}"
    # CHANGE-net not separated from CFD in either assay
    for assay in ("GUIDE-seq-2", "rhAmpSeq"):
        s = a[a.assay == assay].set_index("model")
        cn, cfd = s.loc["CHANGE-net"], s.loc["CFD"]
        assert cn.assoc_lo <= cfd.assoc_hi and cfd.assoc_lo <= cn.assoc_hi
    assert (a.assoc_lo > 0).all()
    # models inform very different numbers of sites - this is why ceilings are matched
    assert a.n_sites_dropped_constant.max() >= 20


def test_conformal_valid_under_both_exchangeability_units():
    c = pd.read_csv(RES / "conformal_coverage.csv")
    assert set(c.unit) == {"site", "donor"}
    pooled = c[(c.ancestry == "ALL") & (c.method == "marginal")]
    assert pooled.coverage.between(0.85, 0.95).all()
    gaps = (c[c.ancestry != "ALL"].groupby(["assay", "model", "unit", "method"])
            .coverage.agg(lambda x: x.max() - x.min()))
    assert gaps.max() < 0.05, "no large ancestry coverage gap under either unit"


def test_manuscript_tables_match_the_primary_result_table():
    """Tables 1-2 are generated from results/, not typed by hand."""
    at = pd.read_csv(RES / "attainment_metric_consistent.csv")
    t1 = pd.read_csv(RES / "table1_performance.csv")
    t2 = pd.read_csv(RES / "table2_ancestry_conformal.csv")
    assert len(t1) == len(at) == 10 and len(t2) == 10
    for _, r in at.iterrows():
        row = t1[(t1.Assay == r.assay) & (t1.Model == r.model)]
        assert len(row) == 1
        assert row["% of ceiling"].iloc[0].startswith(f"{r.pct_of_ceiling:.1f}")
        assert float(row["Ceiling"].iloc[0]) == pytest.approx(r.ceiling_matched, abs=5e-4)


def test_per_guide_reports_both_scales_and_primary_is_never_negative():
    pg = pd.read_csv(RES / "per_guide_within_rho.csv")
    models = ["CHANGE-net", "CFD", "CRISOT", "CRISPRoff", "CRISPRnet"]
    assert all(m in pg.columns and m + "_rawscale" in pg.columns for m in models)
    assert (pg[models] >= 0).all().all(), "primary percentile scale should be non-negative"
    # the raw scale does invert at CTLA4_site_9 on GUIDE-seq-2; that is the point of the check
    ctla = pg[(pg.assay == "GUIDE-seq-2") & (pg.gRNA == "CTLA4_site_9")]
    assert (ctla[[m + "_rawscale" for m in models]] < 0).all().all()


def test_donor_term_beats_the_released_ensemble_on_rhampseq():
    """The central new result, as a regression guard."""
    C = pd.read_csv(RES / "delta_model_contrasts.csv")
    r = C[(C.assay == "rhAmpSeq") & (C.estimator == "ridge")
          & (C.contrast == "M5 full - M1 baseline")].iloc[0]
    assert r.mean_diff > 0 and bool(r.significant) and r.ci_lo > 0


def test_variant_features_never_significantly_help():
    C = pd.read_csv(RES / "delta_model_contrasts.csv")
    v = C[C.contrast.isin(["M2 variant - M1 baseline", "M3 combined - M1 baseline"])]
    assert len(v) == 8
    assert not ((v.significant) & (v.mean_diff > 0)).any()
    assert int(((v.significant) & (v.mean_diff < 0)).sum()) == 3
