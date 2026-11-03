"""Extend cohort ancestry analysis to gnomAD v4 nine genetic-ancestry groups.

Reads cached gnomAD v4 region-query JSON (one file per off-target site) and produces
per-site x ancestry and per-guide x ancestry tables.
"""
import os, json, glob
import numpy as np, pandas as pd

SITES_CSV = ("/Users/ayseguleker/.claude-science/orgs/9c3d227d-3360-4a74-8bae-9a995577857e/"
             "artifacts/proj_31fbac01f839/ccfe368e-0d11-479a-8404-f28417a5afd8/"
             "va7179931_offtarget_sites_1115.csv")
CACHE = "gnomad_cache"
POPS = ["afr", "ami", "amr", "asj", "eas", "fin", "mid", "nfe", "sas"]
# 1000 Genomes cohort populations -> closest gnomAD v4 genetic-ancestry group
COHORT_PROXY = {"CEU": "nfe", "CHB": "eas", "ASW": "afr", "MXL": "amr"}
COHORT_GROUPS = sorted(set(COHORT_PROXY.values()))
ABSENT_GROUPS = [p for p in POPS if p not in COHORT_GROUPS]

COMP = str.maketrans("ACGTN", "TGCAN")


def rc(s):
    return s.translate(COMP)[::-1]


# ---------------------------------------------------------------- variant QC
def pick_block(v):
    """Return (block, source) giving the frequency block to use for a variant.

    A variant is kept only if at least one of its genome/exome call sets is PASS
    (empty `filters`). When both are PASS the joint (exome+genome) block is used.
    """
    g, e, j = v.get("genome"), v.get("exome"), v.get("joint")
    g_ok = g is not None and not (g.get("filters") or [])
    e_ok = e is not None and not (e.get("filters") or [])
    if g_ok and e_ok and j is not None:
        return j, "joint"
    if g_ok and e_ok:
        return g, "genome"          # fallback, should not happen
    if g_ok:
        return g, "genome"
    if e_ok:
        return e, "exome"
    return None, None


def pop_af(block):
    """dict pop -> (ac, an) for the nine genetic-ancestry groups."""
    out = {}
    for p in block.get("populations") or []:
        pid = p.get("id")
        if pid in POPS:          # excludes "remaining", "" and sex-stratified ids
            out[pid] = (p.get("ac") or 0, p.get("an") or 0)
    return out


# ------------------------------------------------------- protospacer mapping
def site_base_map(row):
    """pos (1-based) -> expected reference base, from the site's protospacer+PAM.

    Only defined when the interval length equals the sequence length.
    Returns (dict, seq_index_by_pos) or (None, None).
    """
    L = int(row.end) - int(row.start)
    seq = str(row.seq).upper()
    if L != len(seq):
        return None, None
    bases, idx = {}, {}
    for i, b in enumerate(seq):
        if row.strand == "+":
            pos = int(row.start) + 1 + i
            ref_b = b
        else:
            pos = int(row.end) - i
            ref_b = b.translate(COMP)
        bases[pos] = ref_b
        idx[pos] = i
    return bases, idx


def pam_status(seq):
    """True if the last three bases of a 23-mer protospacer+PAM are NGG."""
    return len(seq) >= 3 and seq[-2] == "G" and seq[-1] == "G"


def main():
    sites = pd.read_csv(SITES_CSV)
    rows, qc = [], {"snv_ref_checked": 0, "snv_ref_match": 0}
    site_meta = []
    missing = []

    for row in sites.itertuples():
        f = os.path.join(CACHE, f"{row.OT_name}.json")
        if not os.path.exists(f):
            missing.append(row.OT_name)
            continue
        rec = json.load(open(f))
        start, stop = rec["start"], rec["stop"]
        bases, idx = site_base_map(row)
        seq = str(row.seq).upper()
        pam_assessable = bases is not None and len(seq) == 23

        kept = []        # (variant_id, pop->(ac,an), pam_effect)
        n_raw = len(rec["variants"])
        n_fail_qc = 0
        for v in rec["variants"]:
            if not (start <= v["pos"] <= stop):
                continue
            blk, src = pick_block(v)
            if blk is None:
                n_fail_qc += 1
                continue
            afs = pop_af(blk)
            pam_eff = "not_assessed"
            ref, alt = v["ref"].upper(), v["alt"].upper()
            if pam_assessable:
                if len(ref) == 1 and len(alt) == 1:
                    exp = bases.get(v["pos"])
                    if exp is not None:
                        qc["snv_ref_checked"] += 1
                        if exp == ref:
                            qc["snv_ref_match"] += 1
                        i = idx[v["pos"]]
                        if i >= 20:          # PAM positions 21-23 (0-based 20,21,22)
                            alt_b = alt if row.strand == "+" else alt.translate(COMP)
                            mut = seq[:i] + alt_b + seq[i + 1:]
                            before, after = pam_status(seq), pam_status(mut)
                            if before and not after:
                                pam_eff = "disrupting"
                            elif after and not before:
                                pam_eff = "creating"
                            else:
                                pam_eff = "neutral"
                        else:
                            pam_eff = "outside_pam"
                    else:
                        pam_eff = "outside_site"
                else:
                    pam_eff = "indel_not_assessed"
            kept.append((v["variant_id"], afs, pam_eff))

        site_meta.append({"OT_name": row.OT_name, "n_variants_raw": n_raw,
                          "n_variants_pass": len(kept), "n_variants_failed_qc": n_fail_qc,
                          "pam_assessable": bool(pam_assessable)})

        for pop in POPS:
            afs, nv, pc, pd_, afc, afd = [], 0, 0, 0, 0.0, 0.0
            for vid, popdict, eff in kept:
                ac, an = popdict.get(pop, (0, 0))
                if an == 0:
                    continue
                af = ac / an
                if ac > 0:
                    nv += 1
                    afs.append(af)
                    if eff == "creating":
                        pc += 1; afc = max(afc, af)
                    elif eff == "disrupting":
                        pd_ += 1; afd = max(afd, af)
            # carrier probability: HWE within group, independence across variants
            carrier = 1.0 - float(np.prod([(1.0 - a) ** 2 for a in afs])) if afs else 0.0
            rows.append({
                "OT_name": row.OT_name, "chr": row.chr, "start": row.start, "end": row.end,
                "strand": row.strand, "gRNA": row.gRNA, "on_target": row.on_target,
                "has_variant_in_cohort": row.has_variant_in_cohort,
                "ancestry": pop,
                "n_variants_site": len(kept),
                "n_variants_polymorphic": nv,
                "max_af": max(afs) if afs else 0.0,
                "carrier_prob": carrier,
                "pam_assessable": bool(pam_assessable),
                "n_pam_creating": pc, "max_af_pam_creating": afc,
                "n_pam_disrupting": pd_, "max_af_pam_disrupting": afd,
            })

    site_df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(site_meta)
    site_df.to_csv("gnomad_site_ancestry.csv", index=False)

    # ------------------------------------------------------- per-guide summary
    g = []
    for (grna, pop), sub in site_df.groupby(["gRNA", "ancestry"]):
        g.append({
            "gRNA": grna, "ancestry": pop,
            "in_cohort": pop in COHORT_GROUPS,
            "n_sites": len(sub),
            "n_sites_with_variant": int((sub.n_variants_polymorphic > 0).sum()),
            "n_sites_carrier_ge_1pct": int((sub.carrier_prob >= 0.01).sum()),
            "n_sites_carrier_ge_5pct": int((sub.carrier_prob >= 0.05).sum()),
            "n_sites_maxaf_ge_1pct": int((sub.max_af >= 0.01).sum()),
            "n_sites_maxaf_ge_5pct": int((sub.max_af >= 0.05).sum()),
            "mean_carrier_prob": float(sub.carrier_prob.mean()),
            "median_carrier_prob": float(sub.carrier_prob.median()),
            "n_sites_pam_creating": int((sub.n_pam_creating > 0).sum()),
            "n_sites_pam_creating_af_ge_1pct": int((sub.max_af_pam_creating >= 0.01).sum()),
            "n_sites_pam_disrupting": int((sub.n_pam_disrupting > 0).sum()),
            "n_sites_pam_disrupting_af_ge_1pct": int((sub.max_af_pam_disrupting >= 0.01).sum()),
            "n_sites_pam_assessable": int(sub.pam_assessable.sum()),
        })
    guide_df = pd.DataFrame(g).sort_values(["gRNA", "ancestry"])
    guide_df.to_csv("gnomad_guide_summary.csv", index=False)

    # -------------------------------------------------------- cohort-gap stats
    piv = site_df.pivot(index="OT_name", columns="ancestry", values="carrier_prob")
    cohort_max = piv[COHORT_GROUPS].max(axis=1)
    gap = []
    for p in POPS:
        for thr in (0.01, 0.05):
            blind = int(((piv[p] >= thr) & (cohort_max < thr)).sum())
            gap.append({"ancestry": p, "in_cohort": p in COHORT_GROUPS,
                        "threshold": thr,
                        "n_sites_altered": int((piv[p] >= thr).sum()),
                        "n_sites_invisible_to_cohort": blind})
    gap_df = pd.DataFrame(gap)

    summary = {
        "n_sites_input": len(sites),
        "n_sites_with_cached_json": len(meta_df),
        "missing_sites": missing,
        "n_variants_raw_total": int(meta_df.n_variants_raw.sum()),
        "n_variants_pass_total": int(meta_df.n_variants_pass.sum()),
        "n_variants_failed_qc_total": int(meta_df.n_variants_failed_qc.sum()),
        "n_sites_pam_assessable": int(meta_df.pam_assessable.sum()),
        "snv_ref_checked": qc["snv_ref_checked"],
        "snv_ref_match": qc["snv_ref_match"],
        "gap_table": gap_df.to_dict("records"),
    }
    json.dump(summary, open("gnomad_analysis_summary.json", "w"), indent=1)
    gap_df.to_csv("gnomad_cohort_gap.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "gap_table"}, indent=1)[:1200])
    print(gap_df.to_string(index=False))


if __name__ == "__main__":
    main()
