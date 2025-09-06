"""
formula_engine
================

This module provides functionality for translating LaTeX-formatted
financial formulas into executable Python expressions and evaluating
them with supplied variable values.  It is designed to support the
subset of LaTeX commonly encountered in quantitative finance: basic
arithmetic, fractions, exponents, maximum/minimum, logarithms,
exponentials, square roots, summations, textual variables and
subscripts.  Unsupported constructs are left unchanged and may
produce evaluation errors.

The main entry point is :func:`evaluate_formula`, which accepts a
LaTeX string (possibly with a leading assignment) and a mapping of
variables to numeric values.  The helper :func:`latex_to_python`
performs a series of regex-driven substitutions to convert LaTeX
syntax into Python syntax.  Fractions and summations require more
involved parsing and are handled by :func:`parse_frac` and
:func:`parse_sum` respectively.  Implicit multiplication is
inserted when adjacency suggests multiplication but no operator is
present.

These functions are entirely independent of Flask and can be used
directly in a non-web context.  The companion ``app.py`` file
exposes a Flask REST API that wraps :func:`evaluate_formula`.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

__all__ = [
    'evaluate_formula',
    'latex_to_python',
    'parse_frac',
    'parse_sum',
    'find_matching_brace',
    'insert_implicit_multiplication',
]


def find_matching_brace(s: str, start: int) -> int:
    """Return the index of the matching closing brace/parenthesis."""
    if start < 0 or start >= len(s):
        return -1
    open_char = s[start]
    if open_char not in '({[':
        return -1
    close_char = {'(': ')', '{': '}', '[': ']'}[open_char]
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_frac(expr: str) -> str:
    """Replace all LaTeX ``\frac{a}{b}`` constructs with ``((a)/(b))``."""
    while '\\frac' in expr:
        idx = expr.find('\\frac')
        if idx == -1:
            break
        pos = idx + len('\\frac')
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
        if pos >= len(expr) or expr[pos] != '{':
            break
        start_num = pos
        end_num = find_matching_brace(expr, start_num)
        if end_num == -1:
            break
        numerator = expr[start_num + 1:end_num]
        pos = end_num + 1
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
        if pos >= len(expr) or expr[pos] != '{':
            break
        start_den = pos
        end_den = find_matching_brace(expr, start_den)
        if end_den == -1:
            break
        denominator = expr[start_den + 1:end_den]
        replacement = f'(({numerator})/({denominator}))'
        expr = expr[:idx] + replacement + expr[end_den + 1:]
    return expr


def parse_sum(expr: str) -> str:
    """Convert LaTeX summations into Python ``sum`` constructs."""
    while True:
        idx = expr.find('\\sum_')
        if idx == -1:
            break
        pos = idx + len('\\sum_')
        if pos < len(expr) and expr[pos] == '(':
            end_lower = find_matching_brace(expr, pos)
            if end_lower == -1:
                break
            lower_content = expr[pos + 1:end_lower]
            pos = end_lower + 1
        else:
            m = re.match(r'([a-zA-Z]\w*)(?:=([^\^]+))?', expr[pos:])
            if not m:
                break
            var_name = m.group(1)
            lower_val = m.group(2)
            if lower_val is not None:
                lower_content = f'{var_name}={lower_val}'
            else:
                lower_content = f'{var_name}=1'
            pos += m.end()
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
        if pos >= len(expr) or expr[pos] != '^':
            break
        pos += 1
        if pos < len(expr) and expr[pos] == '(':
            end_upper = find_matching_brace(expr, pos)
            if end_upper == -1:
                break
            upper_content = expr[pos + 1:end_upper]
            pos = end_upper + 1
        else:
            m2 = re.match(r'([a-zA-Z]\w*|\d+(?:\.\d*)?)', expr[pos:])
            if not m2:
                break
            upper_content = m2.group(1)
            pos += m2.end()
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
        start_summand = pos
        depth = 0
        while pos < len(expr):
            ch = expr[pos]
            if ch == '(':
                depth += 1
            elif ch == ')':
                if depth > 0:
                    depth -= 1
                if depth == 0:
                    j = pos + 1
                    while j < len(expr) and expr[j].isspace():
                        j += 1
                    if j < len(expr) and expr[j] in '*/':
                        break
            if (ch in '+-') and depth == 0 and pos > start_summand:
                break
            pos += 1
        summand = expr[start_summand:pos]
        tmp = summand.lstrip()
        if tmp.startswith('*'):
            tmp = tmp.lstrip('*').lstrip()
        summand = tmp
        if '=' in lower_content:
            var, start_val = lower_content.split('=', 1)
        else:
            var = lower_content
            start_val = '1'
        replacement = (
            f'(sum(({summand}) for {var} in '
            f'range(int({start_val}), int({upper_content}) + 1)))'
        )
        expr = expr[:idx] + replacement + expr[pos:]
    return expr


def tokenize_for_multiplication(expr: str) -> List[str]:
    """Tokenize an expression for implicit multiplication detection."""
    tokens: List[str] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or (ch == '.' and i + 1 < n and expr[i + 1].isdigit()):
            j = i + 1
            while j < n and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        if ch.isalpha() or ch == '_' or ch == '.':
            j = i + 1
            while j < n and (expr[j].isalnum() or expr[j] in '._'):
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        if ch == '*':
            if i + 1 < n and expr[i + 1] == '*':
                tokens.append('**')
                i += 2
                continue
            tokens.append('*')
            i += 1
            continue
        if ch in '+-/%,' or ch == ':':
            tokens.append(ch)
            i += 1
            continue
        if ch in '()[]':
            tokens.append(ch)
            i += 1
            continue
        tokens.append(ch)
        i += 1
    return tokens


def insert_implicit_multiplication(expr: str) -> str:
    """Insert explicit ``*`` operators where multiplication is implied."""
    tokens = tokenize_for_multiplication(expr)
    if not tokens:
        return expr
    result: List[str] = []
    function_tokens = {
        'max', 'min', 'abs', 'pow', 'int', 'range', 'sum',
        'math.log', 'math.exp', 'math.sqrt',
        'math.sin', 'math.cos', 'math.tan', 'math.sinh', 'math.cosh',
        'math.tanh', 'math.floor', 'math.ceil',
    }
    for i, tok in enumerate(tokens):
        result.append(tok)
        if i >= len(tokens) - 1:
            continue
        left = tok
        right = tokens[i + 1]

        def is_value(t: str) -> bool:
            return bool(t and (t[0].isdigit() or t[0].isalpha() or t[0] in '._'))

        def is_opening(t: str) -> bool:
            return t in {'(', '['}

        if (is_value(left) or left.endswith(')') or left.endswith(']')) and (is_opening(right) or is_value(right)):
            if right in {'for', 'in'}:
                continue
            if left in {'for', 'in'}:
                continue
            if left.startswith('\\') or right.startswith('\\'):
                continue
            if left.startswith('sum_') or right.startswith('sum_'):
                continue
            if left == ')' and i >= 3 and tokens[i - 3] == '^':
                continue
            if right == '(':
                if left in function_tokens:
                    continue
                if left.startswith('math.') and left in function_tokens:
                    continue
            if left in {'+', '-', '*', '/', '**', '%', ','}:
                continue
            result.append('*')
    reconstructed: List[str] = []
    for tok in result:
        if tok == 'for':
            reconstructed.append('for ')
            continue
        if tok == 'in':
            if reconstructed and not reconstructed[-1].endswith(' '):
                reconstructed[-1] = reconstructed[-1] + ' '
            reconstructed.append('in ')
            continue
        reconstructed.append(tok)
    return ''.join(reconstructed)


def latex_to_python(expr: str) -> str:
    """Transform a LaTeX expression into a Python expression."""
    expr = expr.replace('$$', '').replace('$', '').strip()
    stripped = expr.lstrip()
    if stripped and stripped[0] != '\\' and '=' in expr:
        expr = expr.split('=', 1)[1]
    expr = expr.replace('\n', '')
    expr = expr.replace('\\left', '').replace('\\right', '')
    expr = expr.replace('\\,', '').replace('\\;', '').replace('\\!', '').replace('\\:', '')
    expr = re.sub(r'\\text\s*{\s*([^}]*)\s*}', lambda m: m.group(1).strip().replace(' ', ''), expr)

    # Handle expectations E[...] before converting brackets.
    def _replace_expectation(match: re.Match) -> str:
        content = match.group(1)
        inner = content.strip()
        control_seqs = re.findall(r'\\([a-zA-Z]+)', inner)
        if (
            re.search(r'[+\-*/^(), ]', inner)
            or any(cmd in {'max', 'min', 'log', 'ln', 'exp', 'sqrt', 'frac', 'sum'} for cmd in control_seqs)
        ):
            return '(' + inner + ')'
        flattened = re.sub(r'\{\s*([^{}]*?)\s*\}', lambda m: m.group(1), inner)
        flattened = flattened.replace(' ', '')
        return 'E_' + flattened

    expr = re.sub(r'E\s*\[([^\]]*)\]', _replace_expectation, expr)

    # Convert remaining square brackets to underscores (indexing).
    expr = expr.replace('[', '_').replace(']', '')

    def flatten_subscripts(s: str) -> str:
        result: List[str] = []
        i = 0
        n = len(s)
        while i < n:
            if s[i] == '_' and i + 1 < n and s[i + 1] == '{':
                start = i - 4 if i - 4 >= 0 else 0
                if s[start:i] == '\\sum':
                    result.append('_')
                    result.append('{')
                    i += 2
                    continue
                close_idx = find_matching_brace(s, i + 1)
                if close_idx == -1:
                    result.append('_')
                    i += 1
                    continue
                content = s[i + 2:close_idx]
                result.append('_' + content)
                i = close_idx + 1
                continue
            result.append(s[i])
            i += 1
        return ''.join(result)

    expr = flatten_subscripts(expr)
    expr = parse_frac(expr)
    expr = expr.replace('\\cdot', '*').replace('\\times', '*')
    expr = expr.replace('\\max', 'max').replace('\\min', 'min')
    expr = expr.replace('\\log', 'math.log').replace('\\ln', 'math.log').replace('\\exp', 'math.exp')
    expr = expr.replace('\\sin', 'math.sin').replace('\\cos', 'math.cos').replace('\\tan', 'math.tan')
    expr = expr.replace('\\sinh', 'math.sinh').replace('\\cosh', 'math.cosh').replace('\\tanh', 'math.tanh')
    expr = expr.replace('\\pi', 'math.pi')
    expr = re.sub(r'\\sqrt\s*{([^}]*)}', lambda m: f'(math.sqrt({m.group(1)}))', expr)

    def strip_backslash(match: re.Match) -> str:
        cmd = match.group(1)
        if cmd in {'frac', 'sum', 'cdot', 'times', 'max', 'min', 'log', 'ln', 'exp', 'sqrt', 'left', 'right'}:
            return match.group(0)
        return cmd

    expr = re.sub(r'\\([a-zA-Z]+)', strip_backslash, expr)
    expr = expr.replace('{', '(').replace('}', ')')
    expr = parse_sum(expr)
    expr = expr.replace('^', '**')
    expr = insert_implicit_multiplication(expr)
    return expr


def evaluate_formula(formula: str, variables: Dict[str, float]) -> float:
    """Evaluate a LaTeX formula given a variable mapping."""
    expression = latex_to_python(formula)
    env: Dict[str, object] = {}
    for k, v in variables.items():
        env[k] = v
    env['math'] = math
    env['max'] = max
    env['min'] = min
    env['sum'] = sum
    env['range'] = range
    env['int'] = int
    env['pow'] = pow
    env['abs'] = abs
    env['e'] = math.e
    safe_globals: Dict[str, object] = {'__builtins__': {}}
    safe_globals.update(env)
    result = eval(expression, safe_globals, {})
    return float(result)
