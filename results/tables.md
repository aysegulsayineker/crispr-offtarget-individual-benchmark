**Table 1.** Pooled, within-site and ceiling-relative performance.

| Assay | Model | Rows | Sites used | Pooled $\rho$ | Within-site $\rho$ | Ceiling | % of ceiling | Site-avg $\rho$ |
|---|---|---|---|---|---|---|---|---|
| GUIDE-seq-2 | CHANGE-net | 15652 | 165/167 | 0.29 | 0.152 (0.107-0.202) | 0.516 | 29.5 (25.2-35.0) | 0.123 (0.019) |
| GUIDE-seq-2 | CFD | 13755 | 145/167 | 0.26 | 0.155 (0.102-0.213) | 0.533 | 29.1 (22.6-34.8) | 0.130 (0.020) |
| GUIDE-seq-2 | CRISOT | 12043 | 127/167 | 0.31 | 0.161 (0.107-0.230) | 0.549 | 29.4 (22.1-36.7) | 0.150 (0.023) |
| GUIDE-seq-2 | CRISPRoff | 13372 | 141/167 | 0.24 | 0.152 (0.101-0.209) | 0.539 | 28.2 (21.5-34.8) | 0.134 (0.021) |
| GUIDE-seq-2 | CRISPRnet | 15272 | 161/167 | 0.29 | 0.117 (0.064-0.175) | 0.521 | 22.4 (14.6-29.0) | 0.114 (0.019) |
| rhAmpSeq | CHANGE-net | 12175 | 132/136 | 0.28 | 0.146 (0.103-0.196) | 0.433 | 33.7 (27.7-40.5) | 0.124 (0.018) |
| rhAmpSeq | CFD | 10701 | 116/136 | 0.25 | 0.162 (0.117-0.209) | 0.444 | 36.5 (29.8-41.8) | 0.135 (0.019) |
| rhAmpSeq | CRISOT | 9099 | 99/136 | 0.30 | 0.174 (0.116-0.226) | 0.471 | 36.8 (28.8-44.0) | 0.150 (0.022) |
| rhAmpSeq | CRISPRoff | 10215 | 111/136 | 0.19 | 0.162 (0.105-0.216) | 0.452 | 35.9 (27.8-42.9) | 0.137 (0.020) |
| rhAmpSeq | CRISPRnet | 11899 | 129/136 | 0.27 | 0.138 (0.091-0.193) | 0.434 | 31.7 (24.6-39.9) | 0.119 (0.018) |

**Table 2.** Ancestry audit, conformal coverage and interval width.

| Assay | Model | $\Delta\rho$ | MAE | $\Delta$bias | $p_{\mathrm{BH}}$ | Cov.~site | Cov.~donor | W~site | W~donor |
|---|---|---|---|---|---|---|---|---|---|
| GUIDE-seq-2 | CHANGE-net | 0.020 | 1.66 | 0.168 | 0.001 | 0.905 | 0.908 | 8.80 | 7.02 |
| GUIDE-seq-2 | CFD | 0.033 | 1.73 | 0.163 | 0.002 | 0.898 | 0.904 | 8.39 | 7.54 |
| GUIDE-seq-2 | CRISOT | 0.043 | 1.52 | 0.164 | 0.001 | 0.899 | 0.900 | 8.18 | 7.35 |
| GUIDE-seq-2 | CRISPRoff | 0.029 | 1.56 | 0.162 | 0.002 | 0.887 | 0.900 | 7.38 | 6.36 |
| GUIDE-seq-2 | CRISPRnet | 0.032 | 1.60 | 0.198 | 0.001 | 0.893 | 0.901 | 8.45 | 7.30 |
| rhAmpSeq | CHANGE-net | 0.036 | 1.50 | 0.386 | 0.001 | 0.888 | 0.901 | 4.53 | 5.25 |
| rhAmpSeq | CFD | 0.029 | 1.87 | 0.380 | 0.001 | 0.918 | 0.900 | 5.41 | 7.65 |
| rhAmpSeq | CRISOT | 0.048 | 1.29 | 0.282 | 0.038 | 0.897 | 0.905 | 4.05 | 0.87 |
| rhAmpSeq | CRISPRoff | 0.043 | 2.36 | 0.391 | 0.001 | 0.903 | 0.900 | 4.51 | 5.56 |
| rhAmpSeq | CRISPRnet | 0.023 | 1.79 | 0.493 | 0.001 | 0.895 | 0.914 | 3.65 | 3.06 |

**Table 3.** Variant-aware and donor-aware models.

| Assay | Predictor | Guide-held-out | Donor-held-out | % of ceiling | vs released ensemble |
|---|---|---|---|---|---|
| GUIDE-seq-2 | best released (CRISOT) | 0.161 | 0.189 |  |  |
|  | released scores (ensemble) | 0.138 | 0.169 | 26.6 | reference |
|  | variant features | 0.108 | 0.132 | 20.8 | -0.031 (-0.061, +0.001) |
|  | scores + variant | 0.117 | 0.166 | 22.6 | -0.021 (-0.037, -0.007)* |
|  | donor propensity | 0.083 | 0.109 | 16.1 | -0.054 (-0.117, +0.009) |
|  | scores + variant + donor | 0.140 | 0.179 | 27.1 | +0.003 (-0.030, +0.037) |
| rhAmpSeq | best released (CRISOT) | 0.174 | 0.192 |  |  |
|  | released scores (ensemble) | 0.125 | 0.161 | 28.9 | reference |
|  | variant features | 0.091 | 0.135 | 21.0 | -0.036 (-0.093, +0.018) |
|  | scores + variant | 0.138 | 0.170 | 31.8 | +0.012 (-0.002, +0.029) |
|  | donor propensity | 0.133 | 0.152 | 30.8 | +0.006 (-0.055, +0.059) |
|  | scores + variant + donor | 0.193 | 0.215 | 44.5 | +0.068 (+0.024, +0.110)* |
