import pytest

class TestRunTest:

    # Directional
    def test_run_mann_kendall(self, churn_series):
        from app.hypothesis_engine import run_test
        r = run_test("mann_kendall_test", churn_series)
        assert r["test"] == "mann_kendall_test"

    def test_run_linear_regression(self, churn_series):
        from app.hypothesis_engine import run_test
        r = run_test("linear_regression_test", churn_series)
        assert r["test"] == "linear_regression_test"

    # Comparative
    def test_run_t_test_with_dict_input(self, two_groups):
        from app.hypothesis_engine import run_test
        r = run_test("t_test", two_groups)
        assert r["test"] == "t_test"
        assert r["significant"] is True

    def test_run_mann_whitney_with_dict_input(self, two_groups):
        from app.hypothesis_engine import run_test
        r = run_test("mann_whitney_test", two_groups)
        assert r["test"] == "mann_whitney_test"

    def test_run_anova_with_dict_input(self, three_groups):
        from app.hypothesis_engine import run_test
        r = run_test("anova_test", three_groups)
        assert r["test"] == "anova_test"

    def test_run_kruskal_with_dict_input(self, three_groups):
        from app.hypothesis_engine import run_test
        r = run_test("kruskal_wallis_test", three_groups)
        assert r["test"] == "kruskal_wallis_test"

    def test_run_comparative_missing_groups_key_raises(self, churn_series):
        from app.hypothesis_engine import run_test
        with pytest.raises(ValueError, match="'groups' key"):
            run_test("t_test", churn_series)

    # Causal-Adjacent
    def test_run_did(self, did_data):
        from app.hypothesis_engine import run_test
        r = run_test("difference_in_differences", did_data)
        assert r["test"] == "difference_in_differences"

    def test_run_its(self, its_data):
        from app.hypothesis_engine import run_test
        r = run_test("interrupted_time_series", its_data)
        assert r["test"] == "interrupted_time_series"

    def test_run_detect_change_point(self, change_point_data):
        from app.hypothesis_engine import run_test
        r = run_test("detect_change_point", change_point_data)
        assert r["test"] == "detect_change_point"

    # Distributional
    def test_run_z_test(self, z_test_data):
        from app.hypothesis_engine import run_test
        r = run_test("z_test", z_test_data)
        assert r["test"] == "z_test"

    def test_run_chi_square_with_expected(self, chi_data_with_expected):
        from app.hypothesis_engine import run_test
        r = run_test("chi_square_test", chi_data_with_expected)
        assert r["test"] == "chi_square_test"

    def test_run_chi_square_without_expected(self, chi_data_no_expected):
        from app.hypothesis_engine import run_test
        r = run_test("chi_square_test", chi_data_no_expected)
        assert r["test"] == "chi_square_test"

    def test_run_ks_test(self, ks_data):
        from app.hypothesis_engine import run_test
        r = run_test("ks_test", ks_data)
        assert r["test"] == "ks_test"

    # Invalid usage tests
    def test_none_data_raises(self):
        from app.hypothesis_engine import run_test
        with pytest.raises(ValueError, match="Data is None"):
            run_test("mann_kendall_test", None)

    def test_unknown_test_name_raises(self, churn_series):
        from app.hypothesis_engine import run_test
        with pytest.raises(ValueError, match="Unknown test"):
            run_test("invented_test", churn_series)

    def test_interpret_significant_result(self):
        from app.hypothesis_engine import interpret_result
        assert "Reject" in interpret_result({"significant": True})

    def test_interpret_non_significant_result(self):
        from app.hypothesis_engine import interpret_result
        assert "Fail to reject" in interpret_result({"significant": False})