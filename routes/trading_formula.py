from __future__ import annotations

"""
Flask application exposing a single endpoint for evaluating LaTeX
formulas.  See the `formula_engine` module for the underlying
conversion and evaluation logic.
"""

from formula_engine import evaluate_formula

app = Flask(__name__)  # type: ignore


@app.route('/trading-formula', methods=['POST'])  # type: ignore
def trading_formula():
    """Compute the value of one or more LaTeX formulas.

    The request body must be a JSON array.  Each element should be
    an object with at least a ``formula`` string and a ``variables``
    mapping.  A ``type`` field may specify 'compute' to indicate
    computation should be performed.  The response contains a list
    of objects with a ``result`` field per input element, rounded to
    four decimal places.
    """
    try:
        data = request.get_json(force=True)  # type: ignore
        if not isinstance(data, list):
            raise ValueError
    except Exception:
        return jsonify({'error': 'Invalid JSON input'}), 400  # type: ignore

    results: list[dict[str, float | None]] = []
    for case in data:
        if not isinstance(case, dict):
            results.append({'result': None})
            continue
        formula: str = case.get('formula', '')
        variables: dict[str, float] = case.get('variables', {})
        case_type: str = case.get('type', 'compute')
        if case_type != 'compute':
            results.append({'result': None})
            continue
        try:
            value = evaluate_formula(formula, variables)
            rounded_value = round(float(value), 4)
            results.append({'result': float(f"{rounded_value:.4f}")})
        except Exception:
            results.append({'result': None})

    return jsonify(results)  # type: ignore

