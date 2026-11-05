"""Donor-level variant features for the individual-level off-target benchmark.

Features are derived from the reference protospacer at the off-target site
(column ``seq``), the donor's two haplotypes (``h1``, ``h2``) and the guide
(``on_target``, an NGG-terminated 23-mer).  Sequences are 3'-anchored: the PAM
occupies the last three characters and the seed the ten characters immediately
5' of it, so positions are counted as distance from the 3' end.  That keeps the
coordinates meaningful for bulge-aligned sites, whose ``seq`` carries a literal
gap character '-' and may be 24 characters long.

gnomAD allele frequency is deliberately NOT a feature: it encodes population
membership, and a model consuming it would be ancestry dependent by
construction, which would undermine the ancestry calibration audit reported in
the same study.

Caveat: when a haplotype differs in length from the reference, the indel is
localised by the longest common prefix and suffix and substitutions inside the
ambiguous region are not counted.  Such rows are 1.3% of the table.
"""
import numpy as np
import pandas as pd

PAM_SPAN = 3        # distance from the 3' end 1..3
SEED_SPAN = 13      # distance from the 3' end 4..13 (protospacer positions 11-20)
BASES = "ACGT"
CHANGES = [a + b for a in BASES for b in BASES if a != b]
PURINES = set("AG")


def _subs(ref, hap):
    """Substitution distances from the 3' end, and the length difference."""
    if not isinstance(ref, str) or not isinstance(hap, str) or not ref or not hap:
        return [], 0
    d = len(hap) - len(ref)
    if d != 0:
        return [], d          # indel: flanks match by construction
    L = len(ref)
    return [L - i for i in range(L) if ref[i] != hap[i] and "-" not in (ref[i], hap[i])], 0


def _hap_features(ref, hap, guide):
    f = {}
    dists, indel = _subs(ref, hap)
    f["n_sub"] = len(dists)
    f["indel_size"] = indel
    f["n_pam"] = sum(1 for x in dists if x <= PAM_SPAN)
    f["n_seed"] = sum(1 for x in dists if PAM_SPAN < x <= SEED_SPAN)
    f["n_distal"] = sum(1 for x in dists if x > SEED_SPAN)
    f["min_dist_pam"] = min(dists) if dists else 99
    f["max_dist_pam"] = max(dists) if dists else 0

    hap_pam = hap[-2:] if isinstance(hap, str) and len(hap) >= 2 else ""
    ref_pam = ref[-2:] if isinstance(ref, str) and len(ref) >= 2 else ""
    f["pam_intact"] = int(hap_pam == "GG")
    f["pam_created"] = int(ref_pam != "GG" and hap_pam == "GG")
    f["pam_disrupted"] = int(ref_pam == "GG" and hap_pam != "GG")

    for c in CHANGES:
        f["sub_" + c] = 0
    f["n_transition"] = f["n_transversion"] = 0
    if indel == 0 and isinstance(ref, str) and isinstance(hap, str):
        for a, b in zip(ref, hap):
            if a == b or a not in BASES or b not in BASES:
                continue
            f["sub_" + a + b] += 1
            key = "n_transition" if (a in PURINES) == (b in PURINES) else "n_transversion"
            f[key] += 1

    nm = nm_seed = np.nan
    if isinstance(guide, str) and isinstance(hap, str) and len(hap) == len(guide):
        nm = nm_seed = 0
        L = len(guide)
        for i, (gc, hc) in enumerate(zip(guide, hap)):
            x = L - i
            if x == PAM_SPAN or gc == hc or "-" in (gc, hc):
                continue
            nm += 1
            if PAM_SPAN < x <= SEED_SPAN:
                nm_seed += 1
    f["n_mm_guide"] = nm
    f["n_mm_guide_seed"] = nm_seed
    return f


def _agg(x, y, how):
    vals = [z for z in (x, y) if z is not None and not (isinstance(z, float) and np.isnan(z))]
    if not vals:
        return np.nan
    return float(np.mean(vals)) if how == "mean" else float(np.max(vals))


def build(df, drop_constant=True):
    """Donor x site feature table, index-aligned with ``df``."""
    rows = []
    for r, h1, h2, gd in zip(df["seq"], df["h1"], df["h2"], df["on_target"]):
        a, b = _hap_features(r, h1, gd), _hap_features(r, h2, gd)
        rec = {}
        for k in a:
            rec[k + "_mean"] = _agg(a[k], b[k], "mean")
            rec[k + "_max"] = _agg(a[k], b[k], "max")
        rec["heterozygous"] = int(h1 != h2)
        # site context: constant within a site, informative only via interactions
        ref_pam = r[-2:] if isinstance(r, str) and len(r) >= 2 else ""
        rec["site_ref_pam_intact"] = int(ref_pam == "GG")
        rec["site_has_bulge"] = int(isinstance(r, str) and "-" in r)
        rec["site_len"] = len(r) if isinstance(r, str) else 0
        rec["site_n_mm_ref"] = _hap_features(r, r, gd)["n_mm_guide"]
        rows.append(rec)
    out = pd.DataFrame(rows, index=df.index)
    if drop_constant:
        keep = out.columns[(out.nunique(dropna=False) > 1)]
        out = out[keep]
    return out


def scoring_coverage(df, preds):
    """Where the released predictor columns are missing, and at what level."""
    uns = df[preds].isna().any(axis=1)
    g = df.assign(_u=uns).groupby("OT_name")._u.agg(["sum", "size"])
    mixed = g[(g["sum"] > 0) & (g["sum"] < g["size"])]
    gap = df["seq"].str.contains("-", regex=False)
    return {
        "rows": int(len(df)),
        "rows_unscored": int(uns.sum()),
        "frac_rows_unscored": round(float(uns.mean()), 4),
        "sites": int(df.OT_name.nunique()),
        "sites_affected": int((g["sum"] > 0).sum()),
        "sites_fully_unscored": int((g["sum"] == g["size"]).sum()),
        "sites_mixed": int(len(mixed)),
        "rows_unscored_in_mixed_sites": int(mixed["sum"].sum()),
        "rows_at_bulge_sites": int(gap.sum()),
        "bulge_sites": int(df.loc[gap, "OT_name"].nunique()),
        "bulge_rows_scored_by_changenet": int(df.loc[gap, "CHANGE-net"].notna().sum()),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analysis
    ot = analysis.load()
    cov = scoring_coverage(ot, analysis.PREDS)
    for k, val in cov.items():
        print(f"{k:34s} {val}")
    X = build(ot[ot.overlapped_vars.notna()])
    print(f"\nfeature matrix: {X.shape[0]} rows x {X.shape[1]} features")
