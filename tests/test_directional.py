class TestDirectional:

    def test_linear_regression_upward_trend_is_significant(self, churn_series):
        from app.hypothesis_tests.directional import linear_regression_test
        r = linear_regression_test(churn_series)
        assert r["test"] == "linear_regression_test"
        assert r["slope"] > 0, "slope should be positive for an upward trend"
        assert r["significant"] is True

    def test_linear_regression_flat_series_not_significant(self, flat_series):
        from app.hypothesis_tests.directional import linear_regression_test
        r = linear_regression_test(flat_series)
        assert r["slope"] == 0.0
        assert r["significant"] is False

    def test_linear_regression_result_has_all_keys(self, churn_series):
        from app.hypothesis_tests.directional import linear_regression_test
        r = linear_regression_test(churn_series)
        assert set(r.keys()) == {"test", "slope", "intercept", "r_squared", "p_value", "significant"}

    def test_linear_regression_significant_is_python_bool(self, churn_series):
        from app.hypothesis_tests.directional import linear_regression_test
        r = linear_regression_test(churn_series)
        assert isinstance(r["significant"], bool), (
            "significant must be a Python bool, not numpy.bool_ — "
            "use bool(p_value < 0.05) in the test function"
        )

    def test_mann_kendall_detects_increasing_trend(self, churn_series):
        from app.hypothesis_tests.directional import mann_kendall_test
        r = mann_kendall_test(churn_series)
        assert r["test"] == "mann_kendall_test"
        assert r["trend"] == "increasing"
        assert r["significant"] is True

    def test_mann_kendall_flat_series_no_trend(self, flat_series):
        from app.hypothesis_tests.directional import mann_kendall_test
        r = mann_kendall_test(flat_series)
        assert r["trend"] in ("no trend", "no_trend")
        assert r["significant"] is False

    def test_mann_kendall_result_has_all_keys(self, churn_series):
        from app.hypothesis_tests.directional import mann_kendall_test
        r = mann_kendall_test(churn_series)
        assert set(r.keys()) == {"test", "trend", "tau", "p_value", "significant", "prewhitened"}

    def test_mann_kendall_tau_in_valid_range(self, churn_series):
        from app.hypothesis_tests.directional import mann_kendall_test
        r = mann_kendall_test(churn_series)
        assert -1 <= r["tau"] <= 1

    def test_p_value_in_valid_range(self, churn_series):
        from app.hypothesis_tests.directional import linear_regression_test
        r = linear_regression_test(churn_series)
        assert 0 <= r["p_value"] <= 1