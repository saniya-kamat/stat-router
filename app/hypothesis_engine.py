from app.test_selector import select_test
from app.hypothesis_parser import parse_hypothesis
from app.hypothesis_tests.directional import linear_regression_test, mann_kendall_test
from app.hypothesis_tests.comparative import (
    t_test, mann_whitney_test, anova_test, kruskal_wallis_test,
    tukey_hsd, dunns_test,
)
from app.hypothesis_tests.causal_adjacent import difference_in_differences, interrupted_time_series, detect_change_point
from app.hypothesis_tests.distributional import z_test, chi_square_test, ks_test


def interpret_result(result):
    if result.get("significant"):
        return "Reject null hypothesis (statistically significant effect detected)."
    return "Fail to reject null hypothesis (insufficient evidence)."


def run_test(test_name, data, preprocess=None):
    if data is None:
        raise ValueError("Data is None.")

    if test_name in ["t_test", "mann_whitney_test", "anova_test", "kruskal_wallis_test",
                     "tukey_hsd", "dunns_test"]:
        if not isinstance(data, dict) or "groups" not in data:
            raise ValueError("Comparative tests require data with a 'groups' key.")

    if test_name == "mann_kendall_test":
        return mann_kendall_test(data, prewhiten=(preprocess == "prewhiten"))

    if test_name == "linear_regression_test":
        return linear_regression_test(data)

    if test_name == "t_test":
        return t_test(data["groups"][0], data["groups"][1])

    if test_name == "mann_whitney_test":
        return mann_whitney_test(data["groups"][0], data["groups"][1])

    if test_name == "anova_test":
        return anova_test(*data["groups"])

    if test_name == "kruskal_wallis_test":
        return kruskal_wallis_test(*data["groups"])

    if test_name == "tukey_hsd":
        return tukey_hsd(*data["groups"])

    if test_name == "dunns_test":
        return dunns_test(*data["groups"])

    if test_name == "difference_in_differences":
        return difference_in_differences(
            data["treatment_before"], data["treatment_after"],
            data["control_before"],   data["control_after"],
        )

    if test_name == "interrupted_time_series":
        return interrupted_time_series(data["series"], data["intervention_index"])

    if test_name == "detect_change_point":
        return detect_change_point(data["series"])

    if test_name == "z_test":
        return z_test(data["current_value"], data["historical_mean"], data["historical_std"])

    if test_name == "chi_square_test":
        return chi_square_test(data["observed"], data.get("expected"))

    if test_name == "ks_test":
        return ks_test(data["sample_a"], data["sample_b"])

    raise ValueError(f"Unknown test: {test_name}")


def run_hypothesis(hypothesis, data):
    """
    Full pipeline:
      raw hypothesis dict
        → parse_hypothesis (normalise, infer direction/mode)
        → select_test      (Shapiro-Wilk + autocorrelation routing)
        → run_test         (execute)
    """
    parsed    = parse_hypothesis(hypothesis)
    selection = select_test(parsed, data)
    test_name = selection["test_name"]
    result    = run_test(test_name, data, preprocess=selection.get("preprocess"))
    return {
        "hypothesis":    hypothesis,
        "selected_test": test_name,
        "result":        result,
        "verdict":       interpret_result(result)
    }