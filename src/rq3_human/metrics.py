"""Agreement estimators for the specified RQ3 human-review protocol.

All estimators operate on the final observation axis, retain leading batch
dimensions, and exclude a pair if either entry is nonfinite. Quadratic kappa
uses the original equally spaced 1--5 scores, not compressed observed labels.
ICC(A,1) uses two fixed named raters, absolute agreement and single measures;
its numerical formula also equals the two-way random absolute-agreement ICC.
It must not be labelled ICC(3,1), which conventionally denotes consistency.

Method references:
https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/ (Table 3)
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

Undefined estimates remain NaN; negative agreement estimates are not clipped.
Bootstrap sampling belongs to the caller and must preserve example clusters.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _paired(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.ndim == 0:
        raise ValueError("An observation axis is required")
    valid = np.isfinite(x) & np.isfinite(y)
    return np.where(valid, x, 0.0), np.where(valid, y, 0.0), valid


def _divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def quadratic_kappa(x, y):
    """Quadratically weighted Cohen kappa using fixed 1--5 score spacing."""
    x, y, valid = _paired(x, y)
    n = valid.sum(axis=-1)
    mean_x = _divide(x.sum(axis=-1), n)
    mean_y = _divide(y.sum(axis=-1), n)
    observed = _divide(((x - y) ** 2).sum(axis=-1), n)
    expected = _divide((x * x + y * y).sum(axis=-1), n) - 2 * mean_x * mean_y
    return 1.0 - _divide(observed, expected)


def spearman(x, y):
    """Pearson correlation of average ranks, with pairwise missingness."""
    x, y, valid = _paired(x, y)
    # Pair masking precedes ranking so a one-sided missing value cannot change
    # the ranks of its counterpart or the denominator of the correlation.
    rx = rankdata(np.where(valid, x, np.nan), axis=-1, method="average", nan_policy="omit")
    ry = rankdata(np.where(valid, y, np.nan), axis=-1, method="average", nan_policy="omit")
    n = valid.sum(axis=-1)
    mx = _divide(np.where(valid, rx, 0.0).sum(axis=-1), n)
    my = _divide(np.where(valid, ry, 0.0).sum(axis=-1), n)
    cx = np.where(valid, rx - mx[..., None], 0.0)
    cy = np.where(valid, ry - my[..., None], 0.0)
    numerator = (cx * cy).sum(axis=-1)
    denominator = np.sqrt((cx * cx).sum(axis=-1) * (cy * cy).sum(axis=-1))
    result = _divide(numerator, denominator)
    return np.where(n >= 2, result, np.nan)


def icc_a1(x, y):
    """Two-way mixed absolute-agreement single-measure ICC, k=2 raters.

    n counts the finite paired output targets, not the number of independently
    sampled example clusters. Cluster dependence is handled by the bootstrap.
    """
    x, y, valid = _paired(x, y)
    n = valid.sum(axis=-1)
    mx = _divide(x.sum(axis=-1), n)
    my = _divide(y.sum(axis=-1), n)
    grand = (mx + my) / 2.0
    row_means = (x + y) / 2.0
    row_centered = np.where(valid, row_means - grand[..., None], 0.0)
    ms_subject = _divide(2.0 * (row_centered ** 2).sum(axis=-1), n - 1)
    ms_rater = n * ((mx - grand) ** 2 + (my - grand) ** 2)
    residual_x = np.where(valid, x - row_means - mx[..., None] + grand[..., None], 0.0)
    residual_y = np.where(valid, y - row_means - my[..., None] + grand[..., None], 0.0)
    ms_error = _divide((residual_x ** 2 + residual_y ** 2).sum(axis=-1), n - 1)
    denominator = ms_subject + ms_error + _divide(2.0 * (ms_rater - ms_error), n)
    result = _divide(ms_subject - ms_error, denominator)
    # Identical constant decimal inputs can otherwise yield tiny roundoff
    # mean squares and a spurious finite coefficient.
    if x.shape[-1]:
        low = np.minimum(np.min(np.where(valid, x, np.inf), axis=-1),
                         np.min(np.where(valid, y, np.inf), axis=-1))
        high = np.maximum(np.max(np.where(valid, x, -np.inf), axis=-1),
                          np.max(np.where(valid, y, -np.inf), axis=-1))
        defined = (n >= 2) & (high != low)
    else:
        defined = n >= 2
    return np.where(defined, result, np.nan)


def summary(point, draws):
    """Point estimate and finite-replicate 95% percentile interval.

    Invalid replicate counts must accompany the interval. None represents an
    undefined point or an interval with no finite bootstrap replicates; invalid
    draws are never replaced by zero or perfect agreement.
    """
    draws = np.asarray(draws, dtype=float).ravel()
    finite = draws[np.isfinite(draws)]
    point = float(point)
    low, high = np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
    return {
        "estimate": point if np.isfinite(point) else None,
        "ci_low": float(low) if np.isfinite(low) else None,
        "ci_high": float(high) if np.isfinite(high) else None,
        "valid_bootstrap_replicates": int(finite.size),
        "invalid_bootstrap_replicates": int(draws.size - finite.size),
    }
