import numpy as np
from scipy.stats import norm, chisquare, ks_2samp

def z_test(current_value, historical_mean, historical_std):
    """
    Tests whether a value is significantly different
    from a historical baseline.
    """
    z_score = (current_value - historical_mean) / historical_std
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return {"test": "z_test", 
            "z_score": float(z_score), 
            "p_value": float(p_value), 
            "significant": bool(p_value < 0.05)}

def chi_square_test(observed, expected=None):
    """
    Tests whether observed category frequencies
    differ from expected frequencies.

    scipy's ``chisquare`` requires the expected counts to sum to the same total
    as the observed counts. Real users routinely supply expected values as
    proportions or as a differently-scaled baseline, so we rescale the expected
    vector to the observed total before testing (the test statistic is
    invariant to which interpretation you intend, only the totals must match).
    When no expected vector is given, a uniform distribution is assumed.
    """
    obs = np.asarray(observed, dtype=float)

    exp = None
    if expected is not None and len(expected) > 0:
        exp = np.asarray(expected, dtype=float)
        total = exp.sum()
        if total > 0:
            exp = exp * (obs.sum() / total)

    statistic, p_value = chisquare(f_obs=obs, f_exp=exp)
    p_value = 1.0 if np.isnan(p_value) else float(p_value)
    return {"test": "chi_square_test",
            "chi_square_statistic": float(statistic),
            "p_value": p_value,
            "significant": bool(p_value < 0.05)}

def ks_test(sample_a, sample_b):
    """
    Tests whether two samples come from the same distribution.
    """
    statistic, p_value = ks_2samp(sample_a, sample_b)
    return {"test": "ks_test", 
            "ks_statistic": float(statistic), 
            "p_value": float(p_value), 
            "significant": bool(p_value < 0.05)}