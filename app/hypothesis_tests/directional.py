from scipy.stats import linregress
import pymannkendall as mk
import numpy as np

def linear_regression_test(series):
    """
    Tests whether a metric has a statistically significant linear trend.
    """
    x = np.arange(len(series))
    result = linregress(x, series)
    return {
        "test": "linear_regression_test",
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_squared": float(result.rvalue ** 2),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < 0.05)
    }

def mann_kendall_test(series, prewhiten=False):
    """
    Non-parametric trend test.
    More robust than linear regression for noisy business data.

    When ``prewhiten`` is True the series is run through Trend-Free
    Pre-Whitening (Yue et al. 2002). Autocorrelation otherwise inflates the
    test's significance (understates the p-value). Plain pre-whitening removes
    part of the trend along with the autocorrelation and badly saps power, so we
    use the trend-free variant: it detrends first, removes lag-1 autocorrelation,
    then restores the trend — correcting the false-positive risk while preserving
    genuine trend detection.
    """
    result = (mk.trend_free_pre_whitening_modification_test(series)
              if prewhiten else mk.original_test(series))
    return {
        "test": "mann_kendall_test",
        "trend": result.trend,
        "tau": float(getattr(result, "Tau", getattr(result, "tau", 0))),
        "p_value": float(result.p),
        "significant": bool(result.p < 0.05),
        "prewhitened": bool(prewhiten),
    }