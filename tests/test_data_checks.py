import pytest

class TestValidateInput:

    # Happy paths (working when given correct input)
    def test_valid_directional_passes(self, churn_series):
        from app.data_checks import validate_input
        assert validate_input({"claim_type": "directional"}, churn_series) is True

    def test_valid_comparative_passes(self, two_groups):
        from app.data_checks import validate_input
        assert validate_input({"claim_type": "comparative"}, two_groups) is True

    def test_valid_causal_its_passes(self, its_data):
        from app.data_checks import validate_input
        assert validate_input({"claim_type": "causal_adjacent"}, its_data) is True

    def test_valid_causal_did_passes(self, did_data):
        from app.data_checks import validate_input
        assert validate_input({"claim_type": "causal_adjacent"}, did_data) is True

    def test_valid_distributional_z_passes(self, z_test_data):
        from app.data_checks import validate_input
        assert validate_input({"claim_type": "distributional"}, z_test_data) is True

    def test_valid_distributional_ks_passes(self, ks_data):
        from app.data_checks import validate_input
        assert validate_input({"claim_type": "distributional"}, ks_data) is True

    def test_valid_distributional_chi_no_expected_passes(self, chi_data_no_expected):
        from app.data_checks import validate_input
        assert validate_input({"claim_type": "distributional"}, chi_data_no_expected) is True

    # Hypothesis errors
    def test_none_hypothesis_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="Missing hypothesis"):
            validate_input(None, [1, 2, 3])

    def test_non_dict_hypothesis_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="Hypothesis must be a dictionary"):
            validate_input("directional", [1, 2, 3])

    def test_missing_claim_type_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="claim_type"):
            validate_input({"metric": "churn"}, [1, 2, 3])

    def test_unknown_claim_type_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="Unknown claim_type"):
            validate_input({"claim_type": "invented"}, [1, 2, 3])

    def test_none_data_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="Missing data"):
            validate_input({"claim_type": "directional"}, None)

    # Directional validation errors
    def test_directional_non_list_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="list/array"):
            validate_input({"claim_type": "directional"}, {"values": [1, 2, 3]})

    def test_directional_too_short_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="at least 3"):
            validate_input({"claim_type": "directional"}, [1.0, 2.0])

    def test_directional_nan_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="NaN"):
            validate_input({"claim_type": "directional"}, [1.0, float("nan"), 3.0])

    # Comparative validation errors
    def test_comparative_non_dict_raises(self, churn_series):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="dictionary"):
            validate_input({"claim_type": "comparative"}, churn_series)

    def test_comparative_missing_groups_key_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="'groups'"):
            validate_input({"claim_type": "comparative"}, {"data": [[1, 2], [3, 4]]})

    def test_comparative_one_group_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="At least 2 groups"):
            validate_input({"claim_type": "comparative"}, {"groups": [[1, 2, 3]]})

    def test_comparative_group_too_small_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="at least 2 values"):
            validate_input({"claim_type": "comparative"}, {"groups": [[1], [2, 3]]})

    def test_comparative_nan_in_group_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="NaN"):
            validate_input(
                {"claim_type": "comparative"},
                {"groups": [[1, 2, float("nan")], [3, 4, 5]]}
            )

    # Distributional validation errors
    def test_distributional_zero_std_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="historical_std cannot be zero"):
            validate_input(
                {"claim_type": "distributional"},
                {"current_value": 0.05, "historical_mean": 0.08, "historical_std": 0.0},
            )

    def test_distributional_empty_observed_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="empty"):
            validate_input({"claim_type": "distributional"}, {"observed": []})

    def test_distributional_length_mismatch_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="same length"):
            validate_input(
                {"claim_type": "distributional"},
                {"observed": [10, 20], "expected": [10, 20, 30]},
            )

    # Causal-adjacent validation errors
    def test_causal_series_too_short_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="at least 5"):
            validate_input({"claim_type": "causal_adjacent"}, {"series": [1, 2, 3, 4]})

    def test_causal_invalid_intervention_index_raises(self):
        from app.data_checks import validate_input
        with pytest.raises(ValueError, match="Invalid intervention_index"):
            validate_input(
                {"claim_type": "causal_adjacent"},
                {"series": list(range(20)), "intervention_index": 0},
            )