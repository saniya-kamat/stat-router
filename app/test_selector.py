import numpy as np
from scipy.stats import shapiro


# ── REAL STATISTICAL CHECKS ───────────────────────────────────────────────────
# Replaces the two placeholder functions that were returning hard-coded values
# and causing incorrect test routing.

def is_normal(series):
    """
    Shapiro–Wilk test for normality.

    Returns True (treat as normal / use parametric test) when:
      - n < 3  : too few observations to test meaningfully
      - n > 5000: Shapiro–Wilk is overpowered at this scale and rejects
                 trivial deviations; rely on the Central Limit Theorem instead
      - p > 0.05: fail to reject normality at α = 0.05
    """
    s = np.array(series, dtype=float)
    n = len(s)
    if n < 3 or n > 5000:
        return True
    _, p = shapiro(s)
    return p > 0.05


def has_autocorrelation(series):
    """
    Lag-1 autocorrelation test using the Bartlett 95% confidence bound.

    The standard result for white noise is that the sample autocorrelation
    at any lag is approximately N(0, 1/n), so |r₁| > 2/√n flags significant
    autocorrelation at roughly α = 0.05.

    No extra dependencies — uses only numpy (scipy not required).
    Returns False for series shorter than 10 observations.
    """
    s = np.array(series, dtype=float)
    n = len(s)
    if n < 10:
        return False
    mu  = np.mean(s)
    den = np.sum((s - mu) ** 2)
    if den == 0:
        return False                        # constant series — no autocorrelation
    r1 = np.sum((s[1:] - mu) * (s[:-1] - mu)) / den
    return abs(r1) > 2 / np.sqrt(n)


def is_categorical(data):
    return data.get("type") == "categorical"


# ── MAIN SELECTOR ─────────────────────────────────────────────────────────────

def select_test(hypothesis, data):
    """
    Routes to the correct statistical test using:
      - claim_type from the parsed hypothesis
      - Shapiro–Wilk normality on the actual data
      - Lag-1 autocorrelation on time series

    Returns:
        {"test_name": "...", ...optional metadata}
    """
    claim_type = hypothesis.get("claim_type")

    # ── DIRECTIONAL ──────────────────────────────────────────────────────────
    if claim_type == "directional":
        # data is a plain list for directional tests
        series = data if isinstance(data, (list, np.ndarray)) else []

        # User intent comes first. "slope" means the user wants to *quantify*
        # the trend (a slope value / forecast) → linear regression. This must be
        # decided before the autocorrelation check, otherwise every trending
        # series (which is autocorrelated by nature) would be forced onto the
        # Mann–Kendall path and the slope request would be silently ignored.
        if hypothesis.get("mode") == "slope":
            return {"test_name": "linear_regression_test"}

        if has_autocorrelation(series):
            # Trend *detection* on an autocorrelated series: Mann–Kendall is
            # robust, and we prewhiten first to remove the autocorrelation that
            # would otherwise inflate the significance (see run_test).
            return {"test_name": "mann_kendall_test", "preprocess": "prewhiten"}

        return {"test_name": "mann_kendall_test"}

    # ── COMPARATIVE ──────────────────────────────────────────────────────────
    if claim_type == "comparative":
        groups = data.get("groups", [])
        k = len(groups)

        if k < 2:
            raise ValueError("Comparative tests require at least 2 groups.")

        if k == 2:
            if is_normal(groups[0]) and is_normal(groups[1]):
                return {"test_name": "t_test"}
            return {"test_name": "mann_whitney_test"}

        # k >= 3 — ANOVA or Kruskal–Wallis (handles any number of groups)
        if all(is_normal(g) for g in groups):
            return {"test_name": "anova_test"}
        return {"test_name": "kruskal_wallis_test"}

    # ── CAUSAL ───────────────────────────────────────────────────────────────
    if claim_type == "causal_adjacent":
        if data.get("mode") == "change_point":
            return {"test_name": "detect_change_point"}
        if "treatment_before" in data and "control_before" in data:
            return {"test_name": "difference_in_differences"}
        return {"test_name": "interrupted_time_series"}

    # ── DISTRIBUTIONAL ───────────────────────────────────────────────────────
    if claim_type == "distributional":
        if is_categorical(data):
            return {"test_name": "chi_square_test"}
        if data.get("mode") == "distribution":
            return {"test_name": "ks_test"}
        return {"test_name": "z_test"}

    raise ValueError(f"Unknown claim_type: {claim_type!r}")