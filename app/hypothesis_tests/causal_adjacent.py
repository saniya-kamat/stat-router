from scipy.stats import ttest_ind
import numpy as np
import ruptures as rpt

def difference_in_differences(treatment_before, treatment_after, control_before, control_after):
    """
    Difference-in-Differences estimator.

    The DiD estimate (treatment_change - control_change) captures the causal
    effect of the intervention.  Significance is assessed with a Welch t-test
    on the baseline-adjusted post-period values: each post-period observation
    is expressed relative to the mean of its pre-period group, then the two
    adjusted post-groups are compared.  This tests whether the treatment group
    changed significantly more (or less) than the control group.
    """
    treatment_change = np.mean(treatment_after) - np.mean(treatment_before)
    control_change   = np.mean(control_after)   - np.mean(control_before)
    did_estimate     = treatment_change - control_change

    # Baseline-adjusted post-period values
    treatment_adj = np.array(treatment_after, dtype=float) - np.mean(treatment_before)
    control_adj   = np.array(control_after,   dtype=float) - np.mean(control_before)
    t_stat, p_value = ttest_ind(treatment_adj, control_adj, equal_var=False)
    p_value = 1.0 if np.isnan(p_value) else float(p_value)

    return {
        "test":              "difference_in_differences",
        "treatment_change":  float(treatment_change),
        "control_change":    float(control_change),
        "did_estimate":      float(did_estimate),
        "t_statistic":       float(t_stat),
        "p_value":           float(p_value),
        "significant":       bool(p_value < 0.05),
    }

def interrupted_time_series(series, intervention_index):
    """
    Simple prototype ITS implementation.
    Splits data before and after intervention
    and compares means using a t-test.
    """
    pre = series[:intervention_index]
    post = series[intervention_index:]
    statistic, p_value = ttest_ind(post, pre, equal_var=False)
    # A flat series on both sides of the break yields a NaN p-value; there is no
    # detectable level shift, so report p = 1.0 (fail to reject) rather than NaN.
    p_value = 1.0 if np.isnan(p_value) else float(p_value)
    return {
        "test": "interrupted_time_series",
        "intervention_index": intervention_index,
        "pre_mean": float(np.mean(pre)),
        "post_mean": float(np.mean(post)),
        "difference": float(np.mean(post) - np.mean(pre)),
        "t_statistic": float(statistic),
        "p_value": p_value,
        "significant": bool(p_value < 0.05)
    }

def detect_change_point(series):
    """
    Uses PELT algorithm to identify structural breaks.
    """
    signal = np.array(series)
    scale  = np.std(signal) or 1
    normed = (signal - np.mean(signal)) / scale
    algo   = rpt.Pelt(model="l2").fit(normed)
    breakpoints = algo.predict(pen=3)
    detected_point = breakpoints[0] if len(breakpoints) > 1 else None

    # Sentinel-stripped breakpoints (PELT always appends len(series) as the last entry)
    real_breakpoints = breakpoints[:-1] if breakpoints else []
    detected = len(real_breakpoints) > 0

    # Compute a pseudo p-value: how far the biggest level-shift sits from the
    # pre-change distribution, expressed as a one-sided normal tail probability.
    # This gives the frontend a continuous signal to display on the p-value bar.
    pseudo_p = 1.0
    if detected:
        split = real_breakpoints[0]
        pre  = signal[:split]
        post = signal[split:]
        if len(pre) >= 2 and len(post) >= 1:
            pre_mean = np.mean(pre)
            pre_std  = np.std(pre, ddof=1) or (scale * 0.01)
            z        = abs(np.mean(post) - pre_mean) / (pre_std / np.sqrt(len(post)))
            from scipy import stats
            pseudo_p = float(2 * stats.norm.sf(z))   # two-tailed

    return {
        "test":            "detect_change_point",
        "change_point":    detected_point,
        "all_breakpoints": breakpoints,
        "significant":     detected,
        "p_value":         pseudo_p,
    }