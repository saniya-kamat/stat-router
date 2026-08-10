"""
Regression tests for the bugs found during the pre-presentation audit.

Each test pins down a specific fix so the behaviour can't silently regress:
  - directional routing precedence (slope vs autocorrelation)
  - prewhitening actually being applied
  - chi-square crashes (sum mismatch, zero expected, single category)
  - degenerate / identical-group inputs (kruskal crash, NaN p-values)
"""

import numpy as np
import pytest

from app.test_selector import select_test, has_autocorrelation
from app.hypothesis_engine import run_hypothesis, run_test
from app.hypothesis_tests.directional import mann_kendall_test
from app.hypothesis_tests.comparative import (
    t_test, anova_test, kruskal_wallis_test,
)
from app.hypothesis_tests.distributional import chi_square_test
from app.hypothesis_tests.causal_adjacent import interrupted_time_series
from app.data_checks import validate_input


# ── DIRECTIONAL ROUTING PRECEDENCE ────────────────────────────────────────────
class TestDirectionalRouting:

    def test_slope_mode_wins_over_autocorrelation(self):
        """
        A trending series is always autocorrelated. Slope mode must still route
        to linear regression — the autocorrelation check must not pre-empt it.
        """
        trend = list(np.linspace(0, 10, 30))
        assert has_autocorrelation(trend)                    # precondition
        r = select_test({"claim_type": "directional", "mode": "slope"}, trend)
        assert r["test_name"] == "linear_regression_test"

    def test_trend_mode_on_autocorrelated_series_flags_prewhiten(self):
        trend = list(np.linspace(0, 10, 30))
        r = select_test({"claim_type": "directional"}, trend)
        assert r["test_name"] == "mann_kendall_test"
        assert r.get("preprocess") == "prewhiten"

    def test_slope_mode_end_to_end_returns_slope(self):
        np.random.seed(42)
        trend = list(np.linspace(0.05, 0.15, 24) + np.random.normal(0, 0.01, 24))
        r = run_hypothesis({"claim_type": "directional", "mode": "slope"}, trend)
        assert r["selected_test"] == "linear_regression_test"
        assert "slope" in r["result"]


# ── PREWHITENING ──────────────────────────────────────────────────────────────
class TestPrewhitening:

    def test_prewhiten_flag_changes_result_shape(self):
        series = list(np.linspace(0, 5, 40))
        plain = mann_kendall_test(series)
        pw    = mann_kendall_test(series, prewhiten=True)
        assert plain["prewhitened"] is False
        assert pw["prewhitened"] is True

    def test_run_hypothesis_applies_prewhitening_on_autocorrelated_trend(self):
        np.random.seed(42)
        trend = list(np.linspace(0.05, 0.15, 24) + np.random.normal(0, 0.01, 24))
        r = run_hypothesis({"claim_type": "directional"}, trend)
        assert r["selected_test"] == "mann_kendall_test"
        assert r["result"]["prewhitened"] is True

    def test_prewhitening_preserves_real_trend_significance(self):
        """Trend-free prewhitening must NOT kill a genuine strong trend."""
        np.random.seed(42)
        trend = list(np.linspace(0.05, 0.15, 24) + np.random.normal(0, 0.01, 24))
        pw = mann_kendall_test(trend, prewhiten=True)
        assert pw["trend"] == "increasing"
        assert pw["significant"] is True

    def test_prewhitening_keeps_flat_noise_insignificant(self):
        np.random.seed(1)
        flat = list(np.random.normal(50, 1, 30))
        pw = mann_kendall_test(flat, prewhiten=True)
        assert pw["significant"] is False


# ── CHI-SQUARE ────────────────────────────────────────────────────────────────
class TestChiSquare:

    def test_sum_mismatch_does_not_crash(self):
        """Expected totals != observed totals used to raise a ValueError."""
        r = chi_square_test([45, 30, 15, 10], [10, 10, 10, 10])
        assert r["chi_square_statistic"] == pytest.approx(30.0)
        assert 0 <= r["p_value"] <= 1

    def test_expected_as_proportions_rescaled(self):
        r = chi_square_test([45, 30, 15, 10], [0.25, 0.25, 0.25, 0.25])
        assert r["chi_square_statistic"] == pytest.approx(30.0)

    def test_uniform_expected_when_none(self):
        r = chi_square_test([33, 33, 34], None)
        assert r["significant"] is False

    def test_validator_rejects_single_category(self):
        with pytest.raises(ValueError, match="at least 2 categories"):
            validate_input({"claim_type": "distributional"}, {"observed": [100]})

    def test_validator_rejects_zero_expected(self):
        with pytest.raises(ValueError, match="greater than zero"):
            validate_input(
                {"claim_type": "distributional"},
                {"observed": [20, 30, 50], "expected": [0, 50, 50]},
            )

    def test_validator_rejects_negative_observed(self):
        with pytest.raises(ValueError, match="negative"):
            validate_input(
                {"claim_type": "distributional"},
                {"observed": [10, -5, 20]},
            )


# ── DEGENERATE / IDENTICAL INPUTS ─────────────────────────────────────────────
class TestDegenerateInputs:

    def test_kruskal_all_identical_does_not_crash(self):
        """scipy raises 'All numbers are identical'; we must not."""
        g = [5.0, 5.0, 5.0, 5.0]
        r = kruskal_wallis_test(g, list(g), list(g))
        assert r["significant"] is False
        assert r["p_value"] == 1.0

    def test_t_test_identical_constant_groups_p_is_one(self):
        g = [5.0, 5.0, 5.0]
        r = t_test(g, list(g))
        assert r["p_value"] == 1.0
        assert r["significant"] is False

    def test_anova_identical_constant_groups_p_is_one(self):
        g = [5.0, 5.0, 5.0]
        r = anova_test(g, list(g), list(g))
        assert r["p_value"] == 1.0
        assert r["significant"] is False

    def test_its_flat_series_p_is_one(self):
        r = interrupted_time_series([5.0] * 20, 10)
        assert r["p_value"] == 1.0
        assert r["significant"] is False

    def test_two_value_groups_run_without_error(self):
        r = run_test("t_test", {"groups": [[1.0, 2.0], [3.0, 4.0]]})
        assert "p_value" in r and 0 <= r["p_value"] <= 1


# ── NON-NUMERIC INPUT REJECTED (NaN leaks) ────────────────────────────────────
class TestNonNumericRejected:

    def test_ks_with_nan_rejected(self):
        with pytest.raises(ValueError, match="non-numeric"):
            validate_input(
                {"claim_type": "distributional"},
                {"sample_a": [1.0, 2.0, float("nan"), 4.0], "sample_b": [5.0, 6.0, 7.0, 8.0]},
            )

    def test_ztest_with_nan_rejected(self):
        with pytest.raises(ValueError, match="valid number"):
            validate_input(
                {"claim_type": "distributional"},
                {"current_value": float("nan"), "historical_mean": 0.08, "historical_std": 0.01},
            )
