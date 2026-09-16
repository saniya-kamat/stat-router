from flask import Blueprint, request, jsonify, render_template
import traceback

from app.hypothesis_engine import run_hypothesis
from app.data_checks import validate_input

routes = Blueprint("routes", __name__)

# Tests the /debug endpoint is permitted to run
_ALLOWED_TESTS = {
    "mann_kendall_test", "linear_regression_test",
    "t_test", "mann_whitney_test",
    "anova_test", "kruskal_wallis_test",
    "difference_in_differences", "interrupted_time_series",
    "detect_change_point", "z_test", "chi_square_test", "ks_test",
    "tukey_hsd", "dunns_test",
}
_MAX_DEBUG_BYTES = 2 * 1024 * 1024  # 2 MB

@routes.route("/")
def index():
    return render_template("index.html")

# Health Check
@routes.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "statrouter-hypothesis-service"
    })


# Main Hypothesis Endpoint
@routes.route("/run-hypothesis", methods=["POST"])
def run_hypothesis_route():

    try:
        payload = request.get_json()

        if not payload:
            return jsonify({
                "status": "error",
                "type": "bad_request",
                "message": "Missing or invalid JSON body"
            }), 400

        hypothesis = payload.get("hypothesis")
        data = payload.get("data")

        # Validation Layer
        validate_input(hypothesis, data)

        # Execution Layer
        result = run_hypothesis(hypothesis, data)

        return jsonify({
            "status": "success",
            "result": result
        }), 200


    except ValueError as ve:
        return jsonify({
            "status": "error",
            "type": "validation_error",
            "message": str(ve)
        }), 400


    except KeyError as ke:
        return jsonify({
            "status": "error",
            "type": "missing_field_error",
            "message": f"Missing field: {str(ke)}"
        }), 400


    except Exception as e:
        traceback.print_exc()  # server-side log only — don't leak internals to the client
        return jsonify({
            "status": "error",
            "type": "internal_error",
            "message": str(e)
        }), 500


# Debug Endpoint
@routes.route("/debug/run-test", methods=["POST"])
def debug_run_test():

    try:
        payload = request.get_json()

        if not payload:
            return jsonify({"error": "Missing JSON body"}), 400

        if request.content_length and request.content_length > _MAX_DEBUG_BYTES:
            return jsonify({"error": "Payload too large (2 MB limit)"}), 413

        test_name = payload.get("test_name")
        data = payload.get("data")

        if not test_name:
            return jsonify({"error": "Missing test_name"}), 400

        if test_name not in _ALLOWED_TESTS:
            return jsonify({"error": f"Unknown test: {test_name!r}"}), 400

        from app.hypothesis_engine import run_test

        result = run_test(test_name, data)

        return jsonify({
            "status": "success",
            "test_name": test_name,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }), 500
