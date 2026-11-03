# Extension of the ancestry analysis to the nine gnomAD v4 genetic-ancestry groups

## 1. Data source and access

| Item | Value |
|---|---|
| Resource | gnomAD GraphQL API, `https://gnomad.broadinstitute.org/api` |
| Dataset | `gnomad_r4` (gnomAD v4), reference genome GRCh38 |
| Date of access | 2026-11-03 (UTC) |
| Authentication | none required (no API key, no contact e-mail) |
| Intervals queried | 1,115 on-/off-target sites of six gRNAs (AAVS1_site_14, CCR5_site_8, CTLA4_site_9, CXCR4_site_8, LAG3_site_9, TRAC_site_1) |
| Intervals returning data | **1,115 / 1,115 (100 %); 0 failures** |
| Rate-limit retries (HTTP 429) | 149, all resolved by exponential back-off |
| Wall-clock time | 3,385 s |

The nine genetic-ancestry groups are `afr`, `ami`, `amr`, `asj`, `eas`, `fin`, `mid`, `nfe`, `sas`.
The `remaining` group and the unlabelled aggregate returned by the API were discarded.

## 2. Exact query

Chromosome names were passed without the `chr` prefix. Two regions were aliased per HTTP request
(the API enforces a query-cost ceiling of 25, and a single `region` query costs 11), with a 2.4 s
pause between requests and exponential back-off on HTTP 429. Each region's raw JSON was cached to
disk (`gnomad_cache/<OT_name>.json`), so the run is resumable and the analysis is reproducible
without re-querying.

```graphql
{
  r0: region(chrom: "1", start: 2919886, stop: 2919908, reference_genome: GRCh38) {
    variants(dataset: gnomad_r4) {
      variant_id pos ref alt flags
      joint  { ac an filters populations { id ac an } }
      genome { ac an filters populations { id ac an } }
      exome  { ac an filters populations { id ac an } }
    }
  }
  r1: region( ... )   # second site, same selection set
}
```

## 3. Coordinates and their validation

The site table is 0-based half-open (BED-like); the 1-based inclusive interval passed to the API is
therefore `start + 1 .. end`. This was confirmed against the GRCh38 reference through the Ensembl
REST `sequence/region` endpoint for a minus-strand site, whose reverse complement reproduced the
tabulated 23-mer exactly.

As an independent, data-driven check, every single-nucleotide variant returned inside a site was
compared with the reference base predicted from the site's own protospacer+PAM sequence and strand:
**5,862 / 5,862 reference alleles matched (100 %)**. The genomic-position-to-protospacer-position
mapping used for the PAM analysis is therefore verified, not assumed.

## 4. Variant quality control

A variant was retained only if at least one of its call sets carried an empty `filters` array
(i.e. PASS). Frequencies were then taken, in order of preference, from the `joint` (exome+genome)
block when both call sets were PASS, otherwise from whichever single call set was PASS.

| | count |
|---|---|
| Variants returned inside the 1,115 intervals | 10,785 |
| Retained (PASS in genome and/or exome) | **7,027** (6,529 SNVs, 498 indels/MNVs) |
| Excluded (AC0 and/or AS_VQSR in every call set) | 3,758 |
| Frequency source: genome / exome / joint | 5,040 / 1,468 / 519 |

## 5. Allele frequency and carrier probability

For ancestry group *g* and variant *v*, AF<sub>v,g</sub> = AC<sub>v,g</sub> / AN<sub>v,g</sub>
(variants with AN<sub>v,g</sub> = 0 are skipped for that group). For each site *s*:

- `n_variants_site` — PASS variants overlapping the interval (identical across groups);
- `n_variants_polymorphic` — those with AC ≥ 1 in group *g*;
- `max_af` — max<sub>v</sub> AF<sub>v,g</sub>;
- `carrier_prob` — probability that a randomly drawn individual of group *g* carries at least one
  alternate allele somewhere in the site:

  **P(carrier) = 1 − Π<sub>v</sub> (1 − AF<sub>v,g</sub>)²**

**Assumptions, stated explicitly.** (i) Hardy–Weinberg equilibrium within each ancestry group, so
the two haplotypes of an individual are independent draws; (ii) statistical independence across the
variants overlapping a site, i.e. **linkage disequilibrium is ignored**. Because off-target sites
are only 22–24 bp long, variants inside one site are often in strong LD, so the product form
**over-estimates** the carrier probability whenever a site carries several correlated common
variants; it is an upper bound under positive LD. (iii) Alternate alleles of a multi-allelic site
are returned as separate records and are likewise treated as independent, which has the same
upward-bias direction. (iv) Any individual carrying one alternate allele is counted as "altered" —
no claim is made that the alteration changes cleavage.

## 6. PAM assessment

PAM status was assessed only where the genomic interval length equals the sequence length and the
sequence is exactly 23 nt (protospacer 20 nt + PAM 3 nt): **998 / 1,115 sites**. The remaining
**117 sites** have an interval/sequence length of 22 or 24 nt (bulge-type alignments) and were
**not assessed**; they are flagged `pam_assessable = False` in the per-site table.

For each PASS SNV falling on protospacer+PAM positions 21–23, the alternate base was placed into the
site sequence (complemented for minus-strand sites) and the PAM was re-evaluated:

- **PAM-creating** — reference PAM is not NGG, alternate PAM is NGG;
- **PAM-disrupting** — reference PAM is NGG, alternate PAM is not NGG.

Indels overlapping the PAM were not interpreted (`indel_not_assessed`). Of the 998 assessable
sites, 782 already carry an NGG PAM in the reference and 216 do not; only the latter 216 can in
principle gain a PAM.

**Result.** 15 of those 216 sites (6.9 %) carry at least one gnomAD PASS SNV that converts the PAM
to NGG; 369 sites carry at least one PAM-disrupting SNV. Only two PAM-creating variants are common:
`chr14_35389324_35389347` (CXCR4_site_8, reference PAM ATG) reaches AF 0.070 (eas) to 0.318 (amr),
and `chr11_87771024_87771047` (LAG3_site_9, reference PAM AAG) reaches AF 0.0033 (eas) to 0.069
(sas). The other 13 PAM-creating variants are rare (max AF < 0.01 in every group).

## 7. Cohort coverage versus the nine-group extrapolation

The experimental cohort contains four 1000 Genomes populations, mapped to their closest gnomAD v4
genetic-ancestry group as CEU→`nfe`, CHB→`eas`, ASW→`afr`, MXL→`amr`. Five groups have **no**
representation in the cohort: `asj`, `fin`, `mid`, `sas`, `ami`.

A site is called *altered* in a group at threshold τ when its carrier probability ≥ τ.

| | τ = 1 % | τ = 5 % |
|---|---|---|
| Sites altered in ≥ 1 of the nine groups | 256 | 146 |
| Sites altered in ≥ 1 of the four cohort-proxy groups | 209 | 132 |
| **Sites altered only in groups absent from the cohort** | **47 (4.2 % of 1,115)** | **14 (1.3 %)** |

Per absent group, sites altered at τ = 1 % that no cohort-proxy group reaches: `sas` 18, `mid` 12,
`asj` 8, `fin` 8, `ami` 6. The 47 cohort-invisible sites are spread over all six guides
(LAG3_site_9 15, CXCR4_site_8 12, AAVS1_site_14 11, TRAC_site_1 5, CTLA4_site_9 3, CCR5_site_8 1).

Magnitude: taking for each site the highest carrier probability among the five absent groups minus
the highest among the four cohort-proxy groups, the median difference is −0.0002 (the cohort proxies
usually suffice), but the 90th percentile is +0.0061 and the maximum +0.236; at **28 sites** an
absent group exceeds every cohort-proxy group by more than 5 percentage points. The gap is therefore
not uniform — it is concentrated in a small set of sites that the four-population design cannot see
at all.

## 8. Limitations

1. **Region queries are anchored on variant start position.** An indel beginning upstream of a site
   but extending into it is not returned and is therefore not counted.
2. **LD is ignored** (Section 5); carrier probabilities are upper bounds under positive LD.
3. **Small reference panels for two groups.** Median AN at these loci is 316 for `mid`
   (≈158 individuals) and 912 for `ami` (≈456), versus 68,048 for `nfe` and 41,550 for `afr`.
   Frequencies — and hence the counts above — are correspondingly imprecise for `mid` and `ami`,
   and a 1 % threshold is near the resolution limit for `mid`.
4. **gnomAD genetic-ancestry groups are not 1000 Genomes populations.** The CEU→`nfe`,
   CHB→`eas`, ASW→`afr`, MXL→`amr` mapping is an approximation; ASW and MXL are admixed, so
   "cohort-visible" is an optimistic label.
5. **No activity claim.** Carrier probability quantifies how often a site is *genetically altered*
   in a group, not how often cleavage is altered. Linking the two requires the measured
   GUIDE-seq-2 / rhAmpSeq data.
6. No simulated or imputed frequencies were substituted anywhere; every number traces to a cached
   gnomAD v4 API response.

## 9. Files

- `gnomad_site_ancestry.csv` — 10,035 rows = 1,115 sites × 9 ancestry groups.
- `gnomad_guide_summary.csv` — 54 rows = 6 gRNAs × 9 ancestry groups.
- `gnomad_cohort_gap.csv` — 18 rows = 9 groups × 2 thresholds (Section 7).
- `gnomad_ancestry_analysis.py` — the analysis script.
- `gnomad_fetch_log.json` — fetch provenance (access date, per-site success, retry counts).
