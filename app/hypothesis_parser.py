def normalize_text(text: str) -> str:
    """
    Basic cleanup for future free-text parsing.
    """
    return text.lower().strip()


def infer_direction(text: str):
    """
    Extract directional intent from text or structured input.
    """
    if not text:
        return None

    text = normalize_text(text)

    if any(word in text for word in ["increase", "increasing", "up", "growing", "rise"]):
        return "increasing"

    if any(word in text for word in ["decrease", "decreasing", "down", "fall", "drop"]):
        return "decreasing"

    return None


def infer_claim_type(hypothesis: dict):
    """
    Returns the claim_type from the hypothesis dict.
    Raises ValueError if missing — the frontend is responsible for always
    sending a resolved claim_type; a silent default would run the wrong test.
    """
    if "claim_type" in hypothesis:
        return hypothesis["claim_type"]

    raise ValueError(
        "Missing claim_type in hypothesis payload. "
        "The frontend must resolve the analysis type before submitting."
    )


def infer_mode(hypothesis: dict):
    """
    Determines if directional test is slope-based or just trend detection.
    """
    if hypothesis.get("mode"):
        return hypothesis["mode"]

    # default behavior
    return "trend"


def parse_hypothesis(hypothesis: dict) -> dict:
    """
    Main parser:
    Converts raw hypothesis input into a normalized internal schema.
    """

    if not isinstance(hypothesis, dict):
        raise ValueError("Hypothesis must be a dictionary")

    parsed = {}

    # Core fields
    parsed["claim_type"] = infer_claim_type(hypothesis)
    parsed["metric"] = hypothesis.get("metric")

    # Directional logic
    parsed["direction"] = hypothesis.get("direction") or infer_direction(
        str(hypothesis.get("text", ""))
    )

    parsed["mode"] = infer_mode(hypothesis)

    # Optional fields (causal / comparative)
    parsed["groups"] = hypothesis.get("groups")
    parsed["control_group"] = hypothesis.get("control_group")

    parsed["time_range"] = hypothesis.get("time_range")
    parsed["intervention_date"] = hypothesis.get("intervention_date")

    # Safety defaults
    if parsed["claim_type"] == "directional" and not parsed["direction"]:
        parsed["direction"] = "unknown"

    return parsed