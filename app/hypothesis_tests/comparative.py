from itertools import combinations

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal, rankdata
from scipy.stats import norm as _norm

# tukey_hsd was added in scipy 1.8.0
try:
    from scipy.stats import tukey_hsd as _scipy_tukey_hsd
    _TUKEY_AVAILABLE = True
except ImportError:
    _TUKEY_AVAILABLE = False


def _clean_p(p):
    """
    A NaN p-value comes from a degenerate input — most commonly two groups that
    are each constant with identical means (zero variance, zero difference).
    There is genuinely no evidence of a difference there, so we report p = 1.0
    (fail to reject) rather than letting NaN leak into the verdict and UI.
    """
    return 1.0 if np.isnan(p) else float(p)


def t_test(group_a, group_b):
    """
    Two-sample t-test.
    Assumes approximately normal distributions.
    """
    statistic, p_value = ttest_ind(group_a, group_b, equal_var=False)
    p_value = _clean_p(p_value)
    return {"test": "t_test",
            "statistic": float(statistic),
            "p_value": p_value,
            "significant": bool(p_value < 0.05)}


def mann_whitney_test(group_a, group_b):
    """
    Non-parametric comparison between two groups.
    """
    statistic, p_value = mannwhitneyu(group_a, group_b, alternative="two-sided")
    p_value = _clean_p(p_value)
    return {"test": "mann_whitney_test",
            "statistic": float(statistic),
            "p_value": p_value,
            "significant": bool(p_value < 0.05)}


def anova_test(*groups):
    """
    Compare means across 3+ groups.
    """
    statistic, p_value = f_oneway(*groups)
    p_value = _clean_p(p_value)
    return {"test": "anova_test",
            "f_statistic": float(statistic),
            "p_value": p_value,
            "significant": bool(p_value < 0.05)}


def kruskal_wallis_test(*groups):
    """
    Non-parametric ANOVA alternative.
    """
    # scipy's kruskal raises "All numbers are identical" when every observation
    # across every group is the same. That is a valid (if degenerate) input —
    # there is no difference between the groups, so report a non-significant
    # result instead of crashing.
    all_values = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    if all_values.size and np.all(all_values == all_values[0]):
        return {"test": "kruskal_wallis_test",
                "h_statistic": 0.0,
                "p_value": 1.0,
                "significant": False}

    statistic, p_value = kruskal(*groups)
    p_value = _clean_p(p_value)
    return {"test": "kruskal_wallis_test",
            "h_statistic": float(statistic),
            "p_value": p_value,
            "significant": bool(p_value < 0.05)}


def tukey_hsd(*groups):
    """
    Tukey's Honestly Significant Difference post-hoc test.
    Run after a significant ANOVA to identify which specific pairs differ.
    Controls the family-wise error rate across all pairwise comparisons.
    Requires scipy >= 1.8.0  (pip install --upgrade scipy if needed).
    """
    if not _TUKEY_AVAILABLE:
        raise ImportError(
            "tukey_hsd requires scipy >= 1.8.0. "
            "Upgrade with: pip install --upgrade scipy"
        )

    result = _scipy_tukey_hsd(*groups)
    k = len(groups)

    pairs = []
    for i, j in combinations(range(k), 2):
        p_adj = float(result.pvalue[i, j])
        pairs.append({
            "pair":   f"Group {chr(65 + i)} vs Group {chr(65 + j)}",
            "p_adj":  p_adj,
            "reject": bool(p_adj < 0.05),
        })

    return {
        "test":            "tukey_hsd",
        "pairs":           pairs,
        "any_significant": any(p["reject"] for p in pairs),
    }


def dunns_test(*groups):
    """
    Dunn's post-hoc test for Kruskal-Wallis (Dunn 1964).
    Pairwise rank-sum Z statistics with Bonferroni correction.
    No extra dependencies — only numpy and scipy.
    """
    arrays   = [np.array(g, dtype=float) for g in groups]
    all_data = np.concatenate(arrays)
    n_total  = len(all_data)
    ranks    = rankdata(all_data)

    rank_means, ns = [], []
    pos = 0
    for g in arrays:
        n_g = len(g)
        rank_means.append(float(np.mean(ranks[pos:pos + n_g])))
        ns.append(n_g)
        pos += n_g

    k             = len(groups)
    n_comparisons = k * (k - 1) // 2

    pairs = []
    for i, j in combinations(range(k), 2):
        z_num = abs(rank_means[i] - rank_means[j])
        z_den = np.sqrt(
            (n_total * (n_total + 1) / 12.0) * (1.0 / ns[i] + 1.0 / ns[j])
        )
        z     = float(z_num / z_den) if z_den > 0 else 0.0
        p_raw = float(2 * (1 - _norm.cdf(z)))
        p_adj = float(min(p_raw * n_comparisons, 1.0))

        pairs.append({
            "pair":   f"Group {chr(65 + i)} vs Group {chr(65 + j)}",
            "z_stat": round(z, 4),
            "p_raw":  round(p_raw, 6),
            "p_adj":  round(p_adj, 6),
            "reject": bool(p_adj < 0.05),
        })

    return {
        "test":            "dunns_test",
        "pairs":           pairs,
        "correction":      "bonferroni",
        "any_significant": any(p["reject"] for p in pairs),
    }