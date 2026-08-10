import pytest
import numpy as np

class TestCausalAdjacent:

    def test_did_positive_estimate(self, did_data):
        from app.hypothesis_tests.causal_adjacent import difference_in_differences
        r = difference_in_differences(
            did_data["treatment_before"], did_data["treatment_after"],
            did_data["control_before"],  did_data["control_after"],
        )
        assert r["test"] == "difference_in_differences"
        assert r["did_estimate"] > 0, "treatment should improve more than control"

    def test_did_result_has_all_keys(self, did_data):
        from app.hypothesis_tests.causal_adjacent import difference_in_differences
        r = difference_in_differences(
            did_data["treatment_before"], did_data["treatment_after"],
            did_data["control_before"],  did_data["control_after"],
        )
        assert set(r.keys()) == {
            "test", "treatment_change", "control_change", "did_estimate",
            "t_statistic", "p_value", "significant",
        }

    def test_did_math_is_correct(self, did_data):
        """did_estimate must equal treatment_change - control_change exactly."""
        from app.hypothesis_tests.causal_adjacent import difference_in_differences
        r = difference_in_differences(
            did_data["treatment_before"], did_data["treatment_after"],
            did_data["control_before"],  did_data["control_after"],
        )
        assert r["did_estimate"] == pytest.approx(
            r["treatment_change"] - r["control_change"], rel=1e-9
        )

    def test_its_level_shift_detected(self, its_data):
        from app.hypothesis_tests.causal_adjacent import interrupted_time_series
        r = interrupted_time_series(its_data["series"], its_data["intervention_index"])
        assert r["test"] == "interrupted_time_series"
        assert r["significant"] is True
        assert r["post_mean"] > r["pre_mean"]

    def test_its_difference_equals_post_minus_pre(self, its_data):
        from app.hypothesis_tests.causal_adjacent import interrupted_time_series
        r = interrupted_time_series(its_data["series"], its_data["intervention_index"])
        assert r["difference"] == pytest.approx(
            r["post_mean"] - r["pre_mean"], rel=1e-9
        )

    def test_its_result_has_all_keys(self, its_data):
        from app.hypothesis_tests.causal_adjacent import interrupted_time_series
        r = interrupted_time_series(its_data["series"], its_data["intervention_index"])
        expected_keys = {
            "test", "intervention_index",
            "pre_mean", "post_mean", "difference",
            "t_statistic", "p_value", "significant",
        }
        assert set(r.keys()) == expected_keys

    def test_detect_change_point_true_break_in_all_breakpoints(self, change_point_data):
        """
        The PELT algorithm with pen=3 is aggressive and can return multiple
        breakpoints. The true break at index 30 should appear in all_breakpoints,
        though it may not be the first entry (breakpoints[0]).

        NOTE: pen=3 is hardcoded in detect_change_point. A higher penalty (e.g.
        pen=10) would suppress spurious early breaks; consider making it
        configurable or computing it via BIC.
        """
        from app.hypothesis_tests.causal_adjacent import detect_change_point
        r = detect_change_point(change_point_data["series"])
        assert r["test"] == "detect_change_point"
        assert r["change_point"] is not None
        # True break is at 30; check it lands within ±5 of one of the reported bkps
        series_len = len(change_point_data["series"])
        interior = [b for b in r["all_breakpoints"] if b != series_len]
        assert any(abs(b - 30) <= 5 for b in interior), (
            f"Expected a breakpoint near index 30; got {interior}"
        )

    def test_detect_change_point_breakpoints_include_terminal(self, change_point_data):
        """
        ruptures always appends len(signal) as a terminal sentinel.
        Consumers should strip it before displaying breakpoints to users.
        """
        from app.hypothesis_tests.causal_adjacent import detect_change_point
        r = detect_change_point(change_point_data["series"])
        n = len(change_point_data["series"])
        assert n in r["all_breakpoints"]

    def test_detect_change_point_flat_signal_no_break(self):
        from app.hypothesis_tests.causal_adjacent import detect_change_point
        flat = list(np.zeros(40))
        r = detect_change_point(flat)
        assert r["change_point"] is None, "no structural break in a constant series"