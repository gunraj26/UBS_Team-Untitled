
# routes/trading-forluma.py
# NOTE: The filename intentionally matches the user's requested spelling: "trading-forluma.py"
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Callable

from flask import Blueprint, request, jsonify

trading_formula_bp = Blueprint("trading_formula", __name__)

# -------------------------------
# Utilities for parsing LaTeX-ish
# -------------------------------

GREEK_MAP = {
    "\\alpha": "alpha", "\\beta": "beta", "\\gamma": "gamma", "\\delta": "delta",
    "\\epsilon": "epsilon", "\\zeta": "zeta", "\\eta": "eta", "\\theta": "theta",
    "\\iota": "iota", "\\kappa": "kappa", "\\lambda": "lambda", "\\mu": "mu",
    "\\nu": "nu", "\\xi": "xi", "\\omicron": "omicron", "\\pi": "pi", "\\rho": "rho",
    "\\sigma": "sigma", "\\tau": "tau", "\\upsilon": "upsilon", "\\phi": "phi",
    "\\chi": "chi", "\\psi": "psi", "\\omega": "omega"
}

SAFE_FUNCS: Dict[str, Any] = {
    # math
    "abs": abs,
    "max": max,
    "min": min,
    "round": round,
    "log": math.log,          # natural log
    "ln": math.log,           # alias
    "exp": math.exp,
    "sqrt": math.sqrt,
    "pow": pow,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "e": math.e,
    "pi": math.pi,
}

def _strip_math_delimiters(expr: str) -> str:
    # Remove leading/trailing $ or $$
    expr = expr.strip()
    if expr.startswith("$$") and expr.endswith("$$"):
        return expr[2:-2].strip()
    if expr.startswith("$") and expr.endswith("$"):
        return expr[1:-1].strip()
    return expr

def _rhs(expr: str) -> str:
    # If there's an equals sign, evaluate RHS
    if "=" in expr:
        return expr.split("=", 1)[1].strip()
    return expr.strip()

def _replace_text(expr: str) -> str:
    # Replace \text{...} with its content, remove spaces inside, keep underscores
    def repl(m):
        inner = m.group(1)
        inner = inner.replace(" ", "")
        return inner
    expr = re.sub(r"\\text\{([^}]*)\}", repl, expr)
    return expr

def _replace_greek(expr: str) -> str:
    for k, v in GREEK_MAP.items():
        expr = expr.replace(k, v)
    return expr

def _replace_mult(expr: str) -> str:
    # Replace \times and \cdot with *
    expr = expr.replace("\\times", "*").replace("\\cdot", "*")
    return expr

def _replace_brackets(expr: str) -> str:
    # Convert square brackets used as function-like indices to underscores for variables like E[R_m]
    # but keep square brackets for max/min arguments? We'll first handle function-like E[...]
    # Replace <IDENT>[<INSIDE>] with <IDENT>_<INSIDE> when IDENT is a bare word or greek-term combo.
    def idx_repl(m):
        ident = m.group(1)
        inside = m.group(2)
        # Remove spaces in inside
        inside_clean = re.sub(r"\s+", "", inside)
        # Convert commas and nested brackets to underscores
        inside_clean = inside_clean.replace(",", "_")
        return f"{ident}_{inside_clean}"
    # Apply repeatedly until no change to catch nested
    prev = None
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*\[([^\[\]]+)\]")
    while prev != expr:
        prev = expr
        expr = pattern.sub(idx_repl, expr)
    # For remaining square brackets used as parentheses, switch to parentheses
    expr = expr.replace("[", "(").replace("]", ")")
    return expr

def _replace_frac(expr: str) -> str:
    # Recursively replace \frac{a}{b} with ((a)/(b))
    # We'll parse using a stack to find balanced braces
    while True:
        m = re.search(r"\\frac\s*\{", expr)
        if not m:
            break
        start = m.start()
        # find numerator block
        i = m.end()  # position after '\frac{'
        depth = 1
        num_start = i
        while i < len(expr) and depth > 0:
            if expr[i] == "{":
                depth += 1
            elif expr[i] == "}":
                depth -= 1
            i += 1
        num = expr[num_start:i-1]
        # now expect '/{'
        if i >= len(expr) or expr[i] != "/":
            # malformed; bail
            break
        i += 1
        if i >= len(expr) or expr[i] != "{":
            break
        i += 1
        depth = 1
        den_start = i
        while i < len(expr) and depth > 0:
            if expr[i] == "{":
                depth += 1
            elif expr[i] == "}":
                depth -= 1
            i += 1
        den = expr[den_start:i-1]
        replacement = f"(({num})/({den}))"
        expr = expr[:start] + replacement + expr[i:]
    return expr

def _replace_functions(expr: str) -> str:
    # \max( -> max( ; \min( -> min( ; \log( -> log(
    expr = expr.replace("\\max", "max").replace("\\min", "min")
    expr = expr.replace("\\log", "log")
    expr = expr.replace("\\ln", "ln")
    return expr

def _replace_power(expr: str) -> str:
    # Convert caret ^ to **, including forms like e^{x}
    # First handle e^{...} and e^x
    # Replace \mathrm{e} or just e
    # Ensure standalone 'e' stays as math.e via SAFE_FUNCS
    # For general a^b, we replace ^ with ** when appropriate.
    # Handle braces
    expr = re.sub(r"e\^\{([^}]*)\}", r"(e**(\1))", expr)
    expr = re.sub(r"e\^([A-Za-z0-9_\.]+)", r"(e**(\1))", expr)
    # General power: replace a^{b} -> (a**(b))
    def p_repl(m):
        base = m.group(1).strip()
        exp = m.group(2).strip()
        return f"(({base})**({exp}))"
    expr = re.sub(r"([A-Za-z0-9_\)\]]+)\s*\^\{([^}]*)\}", p_repl, expr)
    # Simple a^b
    expr = re.sub(r"([A-Za-z0-9_\)\]]+)\s*\^\s*([A-Za-z0-9_\.]+)", r"((\1)**(\2))", expr)
    return expr

def _replace_subscripts(expr: str) -> str:
    # Convert X_\alpha -> X_alpha ; X_{foo} -> X_foo
    expr = re.sub(r"_\{([^}]*)\}", lambda m: "_" + re.sub(r"\s+", "", m.group(1)), expr)
    # Greek already replaced, so \alpha should be alpha, but just in case:
    for k, v in GREEK_MAP.items():
        expr = expr.replace("_" + k, "_" + v)
    return expr

def _cleanup_multiplication(expr: str) -> str:
    # Insert explicit * between a)(b), number followed by ident, ident followed by ident, closing ) followed by ident/number, etc.
    # Cases: 2E_R_m -> 2*E_R_m ; )E_R_m -> )*E_R_m ; E_R_m( -> E_R_m*( ; (number or ident)( -> )*(
    # number + ident
    expr = re.sub(r"(?<=\d)\s*(?=[A-Za-z\(])", "*", expr)
    # ident )( ident/number
    expr = re.sub(r"\)\s*(?=[A-Za-z\(])", ")*", expr)
    # ident followed by ident
    expr = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s+(?=[A-Za-z_\(])", r"\1*", expr)
    return expr

def _tokenize_sum(expr: str) -> str:
    # Replace LaTeX \sum_{i=1}^{n} (EXPR) with SUM('i', 1, n, (EXPR))
    # We parse iteratively.
    def parse_sum(s: str, start: int) -> Tuple[str, int]:
        # expects s[start:] begins with \sum_{i=...}^{...}
        m1 = re.match(r"\\sum_\{([A-Za-z]+)\s*=\s*([^}]*)\}\s*\^\{([^}]*)\}", s[start:])
        if not m1:
            return None, start
        idx_var, lo, hi = m1.group(1), m1.group(2), m1.group(3)
        pos = start + m1.end()
        # The expression to be summed should be the next balanced (...) or {...} or [...] or a bare token/expression until a delimiter.
        def grab_balanced(pos: int) -> Tuple[str, int]:
            if pos >= len(s):
                return "", pos
            open_char = s[pos]
            pairs = {"(": ")", "{": "}", "[": "]"}
            if open_char in pairs:
                close_char = pairs[open_char]
                depth = 1
                i = pos + 1
                while i < len(s) and depth > 0:
                    if s[i] == open_char:
                        depth += 1
                    elif s[i] == close_char:
                        depth -= 1
                    i += 1
                return s[pos+1:i-1], i
            # otherwise, read until a comma or plus/minus end; conservative
            i = pos
            while i < len(s) and s[i] not in ",+-":
                i += 1
            return s[pos:i], i
        inner, next_pos = grab_balanced(pos)
        replacement = f"SUM('{idx_var}', ({lo}), ({hi}), ({inner}))"
        return replacement, next_pos

    i = 0
    out = []
    while i < len(expr):
        if expr.startswith("\\sum_{", i):
            rep, ni = parse_sum(expr, i)
            if rep is None:
                out.append(expr[i])
                i += 1
            else:
                out.append(rep)
                i = ni
        else:
            out.append(expr[i])
            i += 1
    return "".join(out)

def SUM(idx: str, lo: float, hi: float, inner_expr: str, env: Dict[str, Any] = None) -> float:
    # Inclusive bounds. idx is the loop var name.
    # Evaluate inner_expr for each idx = lo..hi (integers).
    if env is None:
        env = {}
    # evaluate bounds
    lo_val = int(eval(str(lo), {"__builtins__": {}}, {**SAFE_FUNCS, **env}))
    hi_val = int(eval(str(hi), {"__builtins__": {}}, {**SAFE_FUNCS, **env}))
    total = 0.0
    for k in range(lo_val, hi_val + 1):
        local_env = dict(env)
        local_env[idx] = k
        total += float(eval(inner_expr, {"__builtins__": {}}, {**SAFE_FUNCS, **local_env}))
    return total

def _replace_sum(expr: str) -> str:
    return _tokenize_sum(expr)

# ultra-simple fallback for \frac{NUM}{DEN} when NUM/DEN don't have nested braces
_SIMPLE_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")

def _fallback_simple_frac(expr: str) -> str:
    # repeatedly rewrite simple \frac{...}{...} until none remain
    prev = None
    while prev != expr:
        prev = expr
        expr = _SIMPLE_FRAC_RE.sub(r"((\1)/(\2))", expr)
    return expr


def latex_to_python(expr: str) -> str:
     expr = _strip_math_delimiters(expr)
     expr = _rhs(expr)
    # First: eliminate \frac early
     expr = _replace_frac(expr)
     expr = _fallback_simple_frac(expr)
     expr = _replace_text(expr)
     expr = _replace_greek(expr)
     expr = _replace_subscripts(expr)
     expr = _replace_brackets(expr)  # E[R_p] -> E_R_p (and [] -> ())
     expr = _replace_functions(expr)
     expr = _replace_power(expr)
     expr = _replace_sum(expr)
     expr = _replace_mult(expr)
    # Safety: kill any remaining \frac
     expr = _replace_frac(expr)
     expr = _fallback_simple_frac(expr)
     # normalize whitespace
     expr = re.sub(r"\s+", " ", expr).strip()
     # add explicit multiplications where implicit
     expr = _cleanup_multiplication(expr)
     return expr


def safe_eval(expr: str, vars_env: Dict[str, Any]) -> float:
    # Final sanitation guard: no LaTeX control sequences should remain
    if "\\frac" in expr:
        expr = _replace_frac(expr)
        expr = _fallback_simple_frac(expr)
    if re.search(r"\\[A-Za-z]+", expr):
        # If anything else LaTeX-y survived, make it loud and clear
        raise ValueError(f"Unrecognized LaTeX tokens remain after parsing: {expr!r}")
    # Evaluate expr in a restricted environment including SUM helper
    env = dict(vars_env)
    # expose SUM as callable with closure over env via a wrapper
    def SUM_wrapper(idx, lo, hi, inner_expr):
        return SUM(idx, lo, hi, inner_expr, env)
    local = {**SAFE_FUNCS, **env, "SUM": SUM_wrapper}
    try:
        val = eval(expr, {"__builtins__": {}}, local)
    except NameError as e:
        # help by showing unknown symbol
        raise ValueError(f"Unknown variable or function in expression: {e}")
    except ZeroDivisionError:
        raise ValueError("Division by zero in expression")
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {expr!r} with error: {e}")
    return float(val)

def normalize_var_keys(variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize user variable keys so they align with transformed names:
    - Remove spaces
    - Keep underscores
    - Convert greek LaTeX names to plain (alpha, beta, ...)
    """
    norm = {}
    for k, v in variables.items():
        kk = str(k)
        kk = kk.replace(" ", "")
        # convert LaTeX greek like \alpha to alpha if used in keys (rare)
        for gk, gv in GREEK_MAP.items():
            kk = kk.replace(gk, gv)
        norm[kk] = v
    return norm

def compute_one(formula: str, variables: Dict[str, Any]) -> float:
    py_expr = latex_to_python(formula)
    env = normalize_var_keys(variables)
    result = safe_eval(py_expr, env)
    return result

def round4(x: float) -> float:
    # Round to four decimals in a stable way
    return float(f"{x:.4f}")

# -------------------------------
# Flask route
# -------------------------------

@trading_formula_bp.route("/trading-formula", methods=["POST"])
def trading_formula():
    """
    Expects Content-Type: application/json
    Body: JSON array, each item: { "name": str, "formula": str, "variables": {...}, "type": "compute" }
    Returns: JSON array, each item: { "result": number rounded to 4 dp }
    """
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a JSON array"}), 400
        results: List[Dict[str, Any]] = []
        for case in data:
            if not isinstance(case, dict):
                return jsonify({"error": "Each test case must be an object"}), 400
            formula = case.get("formula")
            variables = case.get("variables", {})
            if not isinstance(formula, str):
                return jsonify({"error": f"Invalid or missing 'formula' in case {case!r}"}), 400
            if not isinstance(variables, dict):
                return jsonify({"error": f"'variables' must be an object in case {case.get('name', '<unnamed>')}"}), 400
            val = compute_one(formula, variables)
            results.append({"result": round4(val)})
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------------------
# Optional: lightweight self-test when run directly
# -------------------------------
if __name__ == "__main__":
    # Quick local tests & demo (does not start a server)
    samples = [
        {
          "name": "test1",
          "formula": r"Fee = \text{TradeAmount} \times \text{BrokerageRate} + \text{FixedCharge}",
          "variables": {
            "TradeAmount": 10000.0,
            "BrokerageRate": 0.0025,
            "FixedCharge": 10.0
          },
          "type": "compute"
        },
        {
          "name": "test2",
          "formula": r"Fee = \max(\text{TradeAmount} \times \text{BrokerageRate}, \text{MinimumFee})",
          "variables": {
            "TradeAmount": 1000.0,
            "BrokerageRate": 0.003,
            "MinimumFee": 15.0
          },
          "type": "compute"
        },
        {
          "name": "test3",
          "formula": r"Fee = \frac{\text{TradeAmount} - \text{Discount}}{\text{ConversionRate}}",
          "variables": {
            "TradeAmount": 11300.0,
            "Discount": 500.0,
            "ConversionRate": 1.2
          },
          "type": "compute"
        },
        {
          "name": "test15",
          "formula": r"$$E[R_i] = R_f + \beta_i (E[R_m] - R_f)$$",
          "variables":{"R_f":0.02,"beta_i":1.2,"E_R_m":0.08},
          "type":"compute"
        },
        {
          "name": "test16",
          "formula": r"$$VaR = Z_\alpha \cdot \sigma_p \cdot V$$",
          "variables":{"Z_alpha":1.65,"sigma_p":0.15,"V":100000.0},
          "type":"compute"
        },
        {
          "name": "test17",
          "formula": r"$$SR = \frac{E[R_p]-R_f}{\sigma_p}$$",
          "variables":{"E_R_p":0.1,"R_f":0.02,"sigma_p":0.15},
          "type":"compute"
        },
        # Edge cases
        {
          "name": "sum1",
          "formula": r"S = \sum_{i=1}^{3}(i^2)",
          "variables": {},
          "type":"compute"
        },
        {
          "name": "exp1",
          "formula": r"A = e^{0.5} + e^1",
          "variables": {},
          "type":"compute"
        },
        {
          "name": "log1",
          "formula": r"L = \log( e^{1} )",
          "variables": {},
          "type":"compute"
        },
        {
          "name": "frac-nest",
          "formula": r"X = \frac{1}{\frac{1}{2}}",
          "variables": {},
          "type": "compute"
        },
    ]
    out = []
    for s in samples:
        res = compute_one(s["formula"], s["variables"])
        out.append({"result": round4(res)})
    import json
    print(json.dumps(out, indent=2))
