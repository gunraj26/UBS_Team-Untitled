import json
import logging
import math
import re

from flask import request
from routes import app

logger = logging.getLogger(__name__)


def safe_eval(expr, variables):
    # allow math functions
    safe_dict = {k: v for k, v in math.__dict__.items()}
    safe_dict.update({
        "max": max,
        "min": min,
        "log": math.log,
        "sqrt": math.sqrt,
        "sum": sum
    })
    safe_dict.update(variables)
    return eval(expr, {"__builtins__": {}}, safe_dict)


def latex_to_python(formula, variables):
    expr = formula
    # remove LaTeX wrappers
    expr = expr.replace("$$", "").replace("\\[", "").replace("\\]", "")
    expr = expr.replace("\\text{", "").replace("}", "")
    expr = expr.replace("\\cdot", "*").replace("\\times", "*")
    expr = expr.replace("\\frac", "frac").replace("\\log", "log")
    expr = expr.replace("\\sqrt", "sqrt")

    # handle fraction \frac{a}{b} -> (a)/(b)
    while "frac" in expr:
        expr = re.sub(r"frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expr)

    # replace variable names with plain form
    for k in variables.keys():
        expr = expr.replace(k, str(k))

    # replace exponentials e^x
    expr = re.sub(r"e\^(\{([^{}]+)\}|([A-Za-z0-9_\+\-\*/\.\(\)]+))", r"math.exp(\2\3)", expr)

    return expr


@app.route('/trading-formula', methods=['POST'])
def trading_formula():
    data = request.get_json()
    logger.info("data sent for evaluation {}".format(data))
    results = []
    for item in data:
        formula = item.get("formula", "")
        variables = item.get("variables", {})
        try:
            expr = latex_to_python(formula, variables)

            # handle summation patterns: \sum_{t=1}^{T} expr
            sum_pattern = re.compile(r"\\sum_{t=1}\^{T}\s*([^\n]+)")
            match = sum_pattern.search(expr)
            if match:
                inner = match.group(1)
                T = int(variables.get("T", 0))
                sum_expr = f"sum(({inner}) for t in range(1, {T}+1))"
                expr = sum_pattern.sub(sum_expr, expr)

                # replace C_t, sigma_t, W_t, Y_t with dict lookups
                for var in list(variables.keys()):
                    if "_" in var:
                        base, idx = var.split("_")
                        expr = expr.replace(f"{base}_{idx}", f"variables['{var}']")

            # generic CF_t issue: replace with lookup
            expr = re.sub(r"CF_t", "variables[f'CF_{t}']", expr)

            value = safe_eval(expr.split("=")[-1], variables)
            results.append({"result": round(float(value), 4)})
        except Exception as e:
            logger.error(f"Error evaluating formula {formula}: {e}")
            results.append({"result": None})
    return json.dumps(results)
