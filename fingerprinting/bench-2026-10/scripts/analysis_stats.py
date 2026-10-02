"""Interval estimators shared by the bench metrics (same definitions as the paper's analysis.py)."""

from __future__ import annotations

from scipy import stats


def cp_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval for a binomial proportion k/n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)
