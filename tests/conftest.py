import pytest
import numpy as np

@pytest.fixture
def churn_series():
    """24-month upward churn trend — directional signal is clear."""
    np.random.seed(42)
    return list(np.linspace(0.05, 0.15, 24) + np.random.normal(0, 0.01, 24))

@pytest.fixture
def flat_series():
    """Genuinely constant series — no trend possible."""
    return list(np.full(30, 0.08))

@pytest.fixture
def two_groups():
    """Two groups with clearly different means (120 vs 100)."""
    np.random.seed(42)
    return {
        "groups": [
            list(np.random.normal(120, 15, 150)),
            list(np.random.normal(100, 18, 150)),
        ]
    }

@pytest.fixture
def identical_groups():
    """Two groups drawn from the same distribution."""
    np.random.seed(0)
    return {
        "groups": [
            list(np.random.normal(100, 10, 100)),
            list(np.random.normal(100, 10, 100)),
        ]
    }

@pytest.fixture
def three_groups():
    """Three groups with different means (100 / 110 / 120)."""
    np.random.seed(42)
    return {
        "groups": [
            list(np.random.normal(100, 10, 80)),
            list(np.random.normal(110, 10, 80)),
            list(np.random.normal(120, 10, 80)),
        ]
    }

@pytest.fixture
def did_data():
    """DiD data: treatment jumps ~12k, control flat."""
    np.random.seed(42)
    return {
        "treatment_before": list(np.random.normal(50000, 3000, 12)),
        "treatment_after":  list(np.random.normal(62000, 3000, 12)),
        "control_before":   list(np.random.normal(48000, 3000, 12)),
        "control_after":    list(np.random.normal(49000, 3000, 12)),
    }

@pytest.fixture
def its_data():
    """ITS data: clear level shift at month 12."""
    np.random.seed(42)
    pre  = list(np.random.normal(50000, 3000, 12))
    post = list(np.random.normal(62000, 3000, 12))
    return {"series": pre + post, "intervention_index": 12}

@pytest.fixture
def change_point_data():
    """Series with an obvious structural break around index 30."""
    np.random.seed(42)
    return {
        "series": list(np.concatenate([
            np.random.normal(0, 1, 30),
            np.random.normal(5, 1, 30),
        ]))
    }

@pytest.fixture
def z_test_data():
    """Conversion rate anomaly: 0.042 vs historical 0.08 ± 0.01."""
    return {
        "current_value": 0.042,
        "historical_mean": 0.08,
        "historical_std": 0.01,
    }

@pytest.fixture
def chi_data_with_expected():
    return {"observed": [20, 30, 50], "expected": [25, 25, 50]}

@pytest.fixture
def chi_data_no_expected():
    """Validator allows expected to be absent (uniform distribution assumed)."""
    return {"observed": [20, 30, 50]}

@pytest.fixture
def ks_data():
    """Two clearly different distributions (mean 0 vs mean 1)."""
    np.random.seed(42)
    return {
        "sample_a": list(np.random.normal(0, 1, 150)),
        "sample_b": list(np.random.normal(1, 1, 150)),
    }