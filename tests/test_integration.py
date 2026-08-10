class TestRunHypothesis:

    def test_directional_end_to_end(self, churn_series):
        from app.hypothesis_engine import run_hypothesis
        r = run_hypothesis({"claim_type": "directional"}, churn_series)
        assert r["selected_test"] in ("mann_kendall_test", "linear_regression_test")
        assert "verdict" in r and "result" in r

    def test_directional_slope_mode_end_to_end(self, churn_series):
        from app.hypothesis_engine import run_hypothesis
        r = run_hypothesis({"claim_type": "directional", "mode": "slope"}, churn_series)
        assert r["selected_test"] == "linear_regression_test"

    def test_comparative_end_to_end(self, two_groups):
        from app.hypothesis_engine import run_hypothesis
        r = run_hypothesis({"claim_type": "comparative"}, two_groups)
        assert r["selected_test"] in ("t_test", "mann_whitney_test")
        assert r["result"]["significant"] is True

    def test_causal_its_end_to_end(self, its_data):
        from app.hypothesis_engine import run_hypothesis
        r = run_hypothesis({"claim_type": "causal_adjacent"}, its_data)
        assert r["selected_test"] == "interrupted_time_series"
        assert r["result"]["significant"] is True

    def test_causal_did_end_to_end(self, did_data):
        from app.hypothesis_engine import run_hypothesis
        data = {**did_data, "control_group": True}
        r = run_hypothesis({"claim_type": "causal_adjacent"}, data)
        assert r["selected_test"] == "difference_in_differences"

    def test_distributional_z_end_to_end(self, z_test_data):
        from app.hypothesis_engine import run_hypothesis
        r = run_hypothesis({"claim_type": "distributional"}, z_test_data)
        assert r["selected_test"] == "z_test"
        assert r["result"]["significant"] is True

    def test_distributional_chi_no_expected_end_to_end(self, chi_data_no_expected):
        from app.hypothesis_engine import run_hypothesis
        data = {**chi_data_no_expected, "type": "categorical"}
        r = run_hypothesis({"claim_type": "distributional"}, data)
        assert r["selected_test"] == "chi_square_test"

    def test_verdict_string_is_present_and_non_empty(self, churn_series):
        from app.hypothesis_engine import run_hypothesis
        r = run_hypothesis({"claim_type": "directional"}, churn_series)
        assert isinstance(r["verdict"], str) and len(r["verdict"]) > 0