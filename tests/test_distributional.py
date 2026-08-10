import pytest
import numpy as np

class TestDistributional:

    def test_z_test_anomaly_is_significant(self, z_test_data):
        from app.hypothesis_tests.distributional import z_test
        r = z_test(**z_test_data)
        assert r["test"] == "z_test"
        assert r["significant"] is True
        assert r["z_score"] < 0, "value is below the historical mean"

    def test_z_test_on_mean_not_significant(self):
        from app.hypothesis_tests.distributional import z_test
        r = z_test(current_value=0.08, historical_mean=0.08, historical_std=0.01)
        assert r["z_score"] == pytest.approx(0.0)
        assert r["significant"] is False

    def test_z_test_significant_is_python_bool(self, z_test_data):
        from app.hypothesis_tests.distributional import z_test
        r = z_test(**z_test_data)
        assert isinstance(r["significant"], bool)

    def test_z_test_p_value_valid_range(self, z_test_data):
        from app.hypothesis_tests.distributional import z_test
        r = z_test(**z_test_data)
        assert 0 <= r["p_value"] <= 1

    def test_chi_square_test_with_expected(self, chi_data_with_expected):
        from app.hypothesis_tests.distributional import chi_square_test
        r = chi_square_test(
            chi_data_with_expected["observed"],
            chi_data_with_expected["expected"],
        )
        assert r["test"] == "chi_square_test"
        assert r["chi_square_statistic"] >= 0
        assert 0 <= r["p_value"] <= 1

    def test_chi_square_uniform_expected(self):
        from app.hypothesis_tests.distributional import chi_square_test
        r = chi_square_test([33, 33, 34], None)  # uniform → not significant
        assert r["significant"] is False

    def test_ks_test_different_distributions_significant(self, ks_data):
        from app.hypothesis_tests.distributional import ks_test
        r = ks_test(ks_data["sample_a"], ks_data["sample_b"])
        assert r["test"] == "ks_test"
        assert r["significant"] is True
        assert 0 < r["ks_statistic"] <= 1

    def test_ks_test_same_distribution_not_significant(self):
        from app.hypothesis_tests.distributional import ks_test
        np.random.seed(42)
        a = list(np.random.normal(0, 1, 300))
        b = list(np.random.normal(0, 1, 300))
        r = ks_test(a, b)
        assert r["significant"] is False