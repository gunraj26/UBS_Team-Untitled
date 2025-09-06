# routes/formula_engine.py
import math
import re
from typing import Dict, Any

def strip_latex_delimiters(formula: str) -> str:
    formula = formula.strip()
    if formula.startswith("$$") and formula.endswith("$$"):
        formula = formula[2:-2]
    if formula.startswith("$") and formula.endswith("$"):
        formula = formula[1:-1]
    for delim_start, delim_end in [("\\[", "\\]"), ("\\(", "\\)")]:
        if formula.startswith(delim_start) and formula.endswith(delim_end):
            formula = formula[len(delim_start):-len(delim_end)]
    return formula

def remove_assignment(s: str) -> str:
    depth = 0
    for i, c in enumerate(s):
        if c in "{(":
            depth += 1
        elif c in "})":
            depth -= 1
        elif c == "=" and depth == 0:
            return s[i + 1:].strip()
    return s.strip()

def replace_frac(s: str) -> str:
    pattern = re.compile(r"\\frac{")
    while True:
        match = pattern.search(s)
        if not match:
            break
        start = match.start()
        i = match.end()
        brace_count = 1
        while i < len(s) and brace_count > 0:
            brace_count += 1 if s[i] == "{" else -1 if s[i] == "}" else 0
            i += 1
        numerator = s[match.end(): i - 1]
        if i < len(s) and s[i] == "{":
            i += 1
        brace_count = 1
        j = i
        while j < len(s) and brace_count > 0:
            brace_count += 1 if s[j] == "{" else -1 if s[j] == "}" else 0
            j += 1
        denominator = s[i: j - 1]
        replacement = f"({numerator})/({denominator})"
        s = s[:start] + replacement + s[j:]
    return s

def handle_sum(s: str) -> str:
    pos = 0
    while True:
        idx = s.find("\\sum_", pos)
        if idx == -1:
            break
        if s[idx + 5] != "{":
            raise ValueError("Unexpected \\sum syntax; missing '{' after _")
        brace_start = idx + 6
        brace_count = 1
        k = brace_start
        while k < len(s) and brace_count > 0:
            brace_count += 1 if s[k] == "{" else -1 if s[k] == "}" else 0
            k += 1
        brace_end = k - 1
        sub = s[brace_start:brace_end]
        if "=" not in sub:
            raise ValueError("Unexpected \\sum syntax; missing '=' in lower limit")
        var, start_expr = [part.strip() for part in sub.split("=", 1)]
        if brace_end + 1 >= len(s) or s[brace_end + 1] != "^":
            raise ValueError("Unexpected \\sum syntax; missing '^'")
        if s[brace_end + 2] != "{":
            raise ValueError("Unexpected \\sum syntax; missing '{' for upper limit")
        upper_start = brace_end + 3
        brace_count = 1
        l = upper_start
        while l < len(s) and brace_count > 0:
            brace_count += 1 if s[l] == "{" else -1 if s[l] == "}" else 0
            l += 1
        upper_end = l - 1
        end_expr = s[upper_start:upper_end].strip()
        after_upper = l
        m = after_upper
        while m < len(s) and s[m].isspace():
            m += 1
        summand_start = m
        depth = 0
        n = summand_start
        while n < len(s):
            c = s[n]
            if c in "({[":
                depth += 1
            elif c in ")}]":
                depth -= 1
            elif depth == 0 and c in "+-" and n != summand_start:
                break
            n += 1
        summand_end = n
        summand = s[summand_start:summand_end].strip()
        replacement = (
            f"sum(( {summand} ) for {var} in range(int({start_expr}), int({end_expr})+1))"
        )
        s = s[:idx] + replacement + s[summand_end:]
        pos = idx + len(replacement)
    return s

def flatten_subscripts(s: str) -> str:
    s = re.sub(r"_\{([^{}]+)\}", r"_\1", s)
    def replace_brackets(match: re.Match) -> str:
        return match.group(1) + "_" + match.group(2)
    s = re.sub(r"([A-Za-z0-9_]+)\[([^\]]+)\]", replace_brackets, s)
    return s

def replace_caret(s: str) -> str:
    return s.replace("^", "**")

def replace_exp(s: str) -> str:
    s = re.sub(r"e\^\{([^{}]+)\}", lambda m: f"math.exp({m.group(1)})", s)
    s = re.sub(r"e\^([A-Za-z0-9_]+)", r"math.exp(\1)", s)
    return s

def replace_functions(s: str) -> str:
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\max", "max").replace("\\min", "min")
    s = s.replace("\\log", "math.log").replace("\\ln", "math.log")
    return s

def replace_text_commands(s: str) -> str:
    def repl(match: re.Match) -> str:
        content = match.group(1)
        return content.replace(" ", "")
    s = re.sub(r"\\text\{([^{}]*)\}", repl, s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", repl, s)
    return s

def replace_greek(s: str) -> str:
    greek_map = {
        "\\alpha": "alpha",
        "\\beta": "beta",
        "\\gamma": "gamma",
        "\\delta": "delta",
        "\\sigma": "sigma",
    }
    for k, v in greek_map.items():
        s = s.replace(k, v)
    return s

def insert_implicit_mul(s: str) -> str:
    function_names = {"max", "min", "sum", "log", "exp", "range", "int", "float", "abs"}
    result = []
    token = ""
    i = 0
    length = len(s)
    while i < length:
        c = s[i]
        result.append(c)
        if c.isalpha() or c == "_" or c.isdigit():
            token += c
        elif c == ".":
            token = ""
        else:
            token = ""
        if c.isdigit() or c.isalpha() or c == "_" or c == ")":
            j = i + 1
            has_space = False
            while j < length and s[j].isspace():
                has_space = True
                j += 1
            if j < length:
                next_c = s[j]
                if c.isdigit():
                    if next_c.isdigit() or next_c == ".":
                        i += 1
                        continue
                if ((c.isalpha() or c == "_" or c.isdigit())
                    and (next_c.isalpha() or next_c == "_" or next_c.isdigit())
                    and not has_space):
                    i += 1
                    continue
                if (next_c == "(" or next_c.isalpha() or next_c == "_" or next_c.isdigit()):
                    if s[j:j+3] == "for":
                        pass
                    elif next_c == "(":
                        if token.lower() in function_names:
                            pass
                        else:
                            result.append("*")
                    else:
                        result.append("*")
        i += 1
    return "".join(result)

def replace_dynamic_vars(expr: str, variables: Dict[str, Any]) -> str:
    iterator_vars = set(re.findall(r"for\\s+([A-Za-z_][A-Za-z0-9_]*)\\s+in\\s+range", expr))
    def repl(match: re.Match) -> str:
        name = match.group(1)
        idx = match.group(2)
        if idx in iterator_vars:
            return f'variables["{name}_" + str({idx})]'
        else:
            return match.group(0)
    return re.sub(r"([A-Za-z][A-Za-z0-9]*)_([A-Za-z])", repl, expr)

def preprocess_formula(formula: str) -> str:
    formula = strip_latex_delimiters(formula)
    formula = remove_assignment(formula)
    formula = replace_text_commands(formula)
    formula = replace_functions(formula)
    formula = replace_greek(formula)
    formula = replace_frac(formula)
    formula = insert_implicit_mul(formula)
    formula = handle_sum(formula)
    formula = flatten_subscripts(formula)
    formula = replace_exp(formula)
    formula = replace_caret(formula)
    return formula

def evaluate_formula(expr: str, variables: Dict[str, Any]) -> float:
    expr = replace_dynamic_vars(expr, variables)
    local_env: Dict[str, Any] = {}
    local_env["math"] = math
    local_env["max"] = max
    local_env["min"] = min
    local_env["sum"] = sum
    local_env["range"] = range
    local_env["int"] = int
    local_env["float"] = float
    local_env["str"] = str
    local_env["abs"] = abs
    for k, v in variables.items():
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            local_env[k] = v
    local_env["variables"] = variables
    try:
        global_env = {"__builtins__": {}}
        global_env.update(local_env)
        result = eval(expr, global_env, {})
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {e}")
    return float(round(result + 1e-10, 4))
