import pytest

class TestSelectTest:

    def test_directional_default_returns_mann_kendall(self, churn_series):
        from app.test_selector import select_test
        r = select_test({"claim_type": "directional"}, churn_series)
        assert r["test_name"] == "mann_kendall_test"

    def test_directional_slope_mode_returns_linear_regression(self, churn_series):
        from app.test_selector import select_test
        r = select_test({"claim_type": "directional", "mode": "slope"}, churn_series)
        assert r["test_name"] == "linear_regression_test"

    def test_comparative_2_groups_routes_correctly(self, two_groups):
        from app.test_selector import select_test
        r = select_test({"claim_type": "comparative"}, two_groups)
        assert r["test_name"] in ("t_test", "mann_whitney_test")

    def test_comparative_3_groups_routes_correctly(self, three_groups):
        from app.test_selector import select_test
        r = select_test({"claim_type": "comparative"}, three_groups)
        assert r["test_name"] in ("anova_test", "kruskal_wallis_test")

    def test_causal_its_routes_correctly(self, its_data):
        from app.test_selector import select_test
        r = select_test({"claim_type": "causal_adjacent"}, its_data)
        assert r["test_name"] == "interrupted_time_series"

    def test_causal_did_routes_correctly(self, did_data):
        from app.test_selector import select_test
        data = {**did_data, "control_group": True}
        r = select_test({"claim_type": "causal_adjacent"}, data)
        assert r["test_name"] == "difference_in_differences"

    def test_causal_change_point_routes_correctly(self, change_point_data):
        from app.test_selector import select_test
        data = {**change_point_data, "mode": "change_point"}
        r = select_test({"claim_type": "causal_adjacent"}, data)
        assert r["test_name"] == "detect_change_point"

    def test_distributional_default_returns_z_test(self, z_test_data):
        from app.test_selector import select_test
        r = select_test({"claim_type": "distributional"}, z_test_data)
        assert r["test_name"] == "z_test"

    def test_distributional_categorical_returns_chi_square(self, chi_data_with_expected):
        from app.test_selector import select_test
        data = {**chi_data_with_expected, "type": "categorical"}
        r = select_test({"claim_type": "distributional"}, data)
        assert r["test_name"] == "chi_square_test"

    def test_distributional_distribution_mode_returns_ks(self, ks_data):
        from app.test_selector import select_test
        data = {**ks_data, "mode": "distribution"}
        r = select_test({"claim_type": "distributional"}, data)
        assert r["test_name"] == "ks_test"

    def test_unknown_claim_type_raises(self, churn_series):
        from app.test_selector import select_test
        with pytest.raises(ValueError, match="Unknown claim_type"):
            select_test({"claim_type": "invented"}, churn_series)