import numpy as np

# Basic validation entry point
def validate_input(hypothesis, data):
    """
    Main validation gate for Lighthouse 4.

    Ensures:
    - hypothesis structure is valid
    - data matches expected format for claim type
    """

    if hypothesis is None:
        raise ValueError("Missing hypothesis")

    if not isinstance(hypothesis, dict):
        raise ValueError("Hypothesis must be a dictionary")

    if "claim_type" not in hypothesis:
        raise ValueError("hypothesis must contain 'claim_type'")

    if data is None:
        raise ValueError("Missing data")

    claim_type = hypothesis["claim_type"]

    # Route to specialized validators
    if claim_type == "directional":
        validate_directional(data)

    elif claim_type == "comparative":
        validate_comparative(data)

    elif claim_type == "causal_adjacent":
        validate_causal(data)

    elif claim_type == "distributional":
        validate_distributional(data)

    else:
        raise ValueError(f"Unknown claim_type: {claim_type}")

    return True


# Directional validation (time series)
def validate_directional(data):
    """
    Expected formats:
    - list/array of numeric values (time series)
    """

    if not isinstance(data, (list, tuple, np.ndarray)):
        raise ValueError("Directional tests require a list/array time series")

    arr = np.array(data)

    if len(arr) < 3:
        raise ValueError("Directional tests require at least 3 data points")

    if np.any(np.isnan(arr)):
        raise ValueError("Data contains NaN values")

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("Directional data must be numeric")

    return True


# Comparative validation (group comparisons)
def validate_comparative(data):
    """
    Expected format:
    {
        "groups": [group1, group2, ...]
    }
    """

    if not isinstance(data, dict):
        raise ValueError("Comparative data must be a dictionary")

    if "groups" not in data:
        raise ValueError("Comparative data must contain 'groups'")

    groups = data["groups"]

    if not isinstance(groups, list):
        raise ValueError("'groups' must be a list of arrays")

    if len(groups) < 2:
        raise ValueError("At least 2 groups required for comparative tests")

    for i, g in enumerate(groups):
        arr = np.array(g)

        if len(arr) < 2:
            raise ValueError(f"Group {i} must have at least 2 values")

        if np.any(np.isnan(arr)):
            raise ValueError(f"Group {i} contains NaN values")

    return True


# Causal validation
def validate_causal(data):
    """
    Expected formats:
    - DiD:
        {
            "treatment_before": [...],
            "treatment_after": [...],
            "control_before": [...],
            "control_after": [...]
        }

    - ITS:
        {
            "series": [...],
            "intervention_index": int
        }

    - Change point:
        {
            "series": [...],
            "mode": "change_point"
        }
    """

    if not isinstance(data, dict):
        raise ValueError("Causal data must be a dictionary")

    # Difference-in-differences
    if "control_before" in data:
        required = [
            "treatment_before",
            "treatment_after",
            "control_before",
            "control_after"
        ]

        for r in required:
            if r not in data:
                raise ValueError(f"Missing required field: {r}")

        return True

    # ITS / Change point
    if "series" in data:
        series = np.array(data["series"])

        if len(series) < 5:
            raise ValueError("Causal time series must have at least 5 points")

        if np.any(np.isnan(series)):
            raise ValueError("Series contains NaN values")

        if "intervention_index" in data:
            idx = data["intervention_index"]

            if not isinstance(idx, int):
                raise ValueError("intervention_index must be integer")

            if idx <= 0 or idx >= len(series):
                raise ValueError("Invalid intervention_index")

        return True

    raise ValueError("Invalid causal data format")


# Distributional validation
def validate_distributional(data):
    """
    Expected formats:

    - z-test:
        {
            "current_value": float,
            "historical_mean": float,
            "historical_std": float
        }

    - chi-square:
        {
            "observed": [...],
            "expected": [...]
        }

    - KS test:
        {
            "sample_a": [...],
            "sample_b": [...]
        }
    """

    if not isinstance(data, dict):
        raise ValueError("Distributional data must be a dictionary")

    # Z-test
    if "current_value" in data:
        for key in ["current_value", "historical_mean", "historical_std"]:
            if key not in data:
                raise ValueError(f"Missing {key}")

        for key in ["current_value", "historical_mean", "historical_std"]:
            if np.isnan(float(data[key])):
                raise ValueError(f"{key} is not a valid number")

        if data["historical_std"] == 0:
            raise ValueError("historical_std cannot be zero")

        return True

    # Chi-square
    if "observed" in data:
        obs = np.array(data["observed"], dtype=float)
        exp = np.array(data.get("expected") or [], dtype=float)

        if len(obs) == 0:
            raise ValueError("Observed values cannot be empty")

        if len(obs) < 2:
            raise ValueError("Chi-square needs at least 2 categories")

        if np.any(np.isnan(obs)):
            raise ValueError("Observed values contain NaN")

        if np.any(obs < 0):
            raise ValueError("Observed counts cannot be negative")

        if obs.sum() <= 0:
            raise ValueError("Observed counts must sum to a positive total")

        if len(exp) > 0:
            if len(obs) != len(exp):
                raise ValueError("Observed and expected must have same length")
            if np.any(exp <= 0):
                raise ValueError("Expected counts must all be greater than zero")

        return True

    # KS test
    if "sample_a" in data and "sample_b" in data:
        a = np.array(data["sample_a"], dtype=float)
        b = np.array(data["sample_b"], dtype=float)

        if len(a) < 2 or len(b) < 2:
            raise ValueError("KS test requires at least 2 samples per group")

        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            raise ValueError("Samples contain non-numeric values")

        return True

    raise ValueError("Invalid distributional data format")