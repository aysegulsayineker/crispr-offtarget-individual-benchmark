"""Smoke test: the analysis outputs exist and the headline numbers are unchanged.

Run after `python src/analysis.py` (or against the committed result tables):
    pytest -q tests
"""
from pathlib import Path

import pandas as pd
import pytest

RES = Path(__file__).resolve().parents[1] / "results"
EXPECTED_FILES = [
    "benchmark_noise_ceiling.csv", "calibration_by_ancestry.csv",
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


def test_headline_benchmark_numbers():
    b = pd.read_csv(RES / "benchmark_noise_ceiling.csv")
    assert set(b.model) == MODELS and set(b.assay) == {"GUIDE-seq-2", "rhAmpSeq"}
    g = b[b.assay == "GUIDE-seq-2"]
    r = b[b.assay == "rhAmpSeq"]
    # pooled correlations are moderate, within-site correlations are not
    assert b.rho_pooled.between(0.19, 0.31).all()
    assert g.rho_within.between(0.050, 0.095).all()
    assert r.rho_within.between(0.155, 0.177).all()
    # noise ceiling
    assert g.explainable_var.eq(0.330).all() and g.ceiling_rho.eq(0.575).all()
    assert r.explainable_var.eq(0.793).all() and r.ceiling_rho.eq(0.891).all()
    # no model reaches a fifth of the attainable between-donor signal
    assert b.pct_of_ceiling.between(8.7, 19.9).all()
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
    assert (cc.coverage - 0.90).abs().max() < 0.03
    marg = cc[(cc.method == "marginal") & (cc.ancestry == "ALL")]
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
