"""Fast site-level bootstrap for within-site percentile associations.

Resampling sites with replacement leaves each site's internal percentile
transform unchanged, so the per-site percentile vectors can be computed once
and the bootstrap reduces to concatenating pre-computed arrays.  This is
numerically identical to rebuilding the frame on every draw and roughly two
orders of magnitude faster.
"""
import numpy as np
from scipy.stats import rankdata


def site_vectors(d, pred_col, col, site_col="OT_name", min_rows=2):
    """Per-site (predictor percentile, measurement percentile) arrays."""
    out = []
    sub = d[d[pred_col].notna() & d[col].notna()]
    for _, g in sub.groupby(site_col):
        p = g[pred_col].to_numpy(dtype=float)
        y = g[col].to_numpy(dtype=float)
        if len(g) < min_rows or np.nanstd(p) <= 1e-9 or np.nanstd(y) <= 0:
            continue
        n = len(p)
        out.append((rankdata(p) / (n + 1), rankdata(y) / (n + 1)))
    return out


def pooled_r(vecs, idx=None):
    if not vecs:
        return np.nan
    sel = vecs if idx is None else [vecs[i] for i in idx]
    if not sel:
        return np.nan
    a = np.concatenate([v[0] for v in sel])
    b = np.concatenate([v[1] for v in sel])
    if a.size < 50 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def boot_diff(vecs_a, vecs_b, rng, B=300, paired=True):
    """Bootstrap of pooled_r(b) - pooled_r(a) over sites.

    With paired=True the two vector lists must be site-aligned, so the same
    resampled sites enter both estimates.
    """
    n = len(vecs_a) if paired else None
    diffs = []
    for _ in range(B):
        if paired:
            idx = rng.integers(0, n, n)
            ra, rb = pooled_r(vecs_a, idx), pooled_r(vecs_b, idx)
        else:
            ra = pooled_r(vecs_a, rng.integers(0, len(vecs_a), len(vecs_a)))
            rb = pooled_r(vecs_b, rng.integers(0, len(vecs_b), len(vecs_b)))
        if ra == ra and rb == rb:
            diffs.append(rb - ra)
    if len(diffs) < 50:
        return (np.nan,) * 4
    arr = np.array(diffs)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    p = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
    return float(arr.mean()), float(lo), float(hi), float(min(p, 1.0))
