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
    """Return the index of the matching closing brace/parenthesis.

    ``s[start]`` must be one of ``(``, ``{`` or ``[``.  The function
    scans forward through ``s`` and keeps track of the nesting depth
    of the corresponding type of brace until it returns to zero.  If
    no matching closing brace is found, ``-1`` is returned.

    Parameters
    ----------
    s : str
        The string to search within.
    start : int
        The index of the opening brace.

    Returns
    -------
    int
        The index of the matching closing brace, or ``-1`` if no
        match exists.
    """
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
    """Replace all LaTeX ``\frac{a}{b}`` constructs with ``((a)/(b))``.

    The replacement is performed iteratively so that nested
    fractions are handled from the inside out.  If the LaTeX
    syntax is malformed (e.g., missing braces), the loop exits
    leaving the remainder unchanged.

    Parameters
    ----------
    expr : str
        The input expression possibly containing ``\frac``.

    Returns
    -------
    str
        The expression with fractions converted to Python division.
    """
    while '\\frac' in expr:
        idx = expr.find('\\frac')
        if idx == -1:
            break
        pos = idx + len('\\frac')
        # Skip whitespace
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
        # Skip whitespace
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
    """Convert LaTeX summations into Python ``sum`` constructs.

    The function searches for patterns of the form ``\sum_{i=lower}^{upper}``
    followed by an expression.  The lower bound must specify the
    summation index and its starting value (e.g., ``i=1``).  The
    upper bound may be any Python expression.  The summand is
    determined by scanning forward until the next unparenthesized
    ``+`` or ``-`` operator or the end of the string.  Summations
    are expanded into calls to Python's ``sum`` with a generator
    expression.  Bounds are cast to integers before creating the
    ``range``.

    Parameters
    ----------
    expr : str
        The string in which to replace summations.

    Returns
    -------
    str
        The transformed string with summations expanded.
    """
    while True:
        idx = expr.find('\\sum_')
        if idx == -1:
            break
        pos = idx + len('\\sum_')
        # Parse lower bound
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
        # Skip whitespace
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
        # Expect caret for upper bound
        if pos >= len(expr) or expr[pos] != '^':
            break
        pos += 1
        # Parse upper bound
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
        # Skip whitespace
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
        # Extract summand
        start_summand = pos
        depth = 0
        while pos < len(expr):
            ch = expr[pos]
            if ch == '(':
                depth += 1
            elif ch == ')':
                if depth > 0:
                    depth -= 1
                # If we've closed all nested parentheses and the next
                # non-space character is a multiplicative operator
                # ('*' or '/') at the same level, treat this as the
                # end of the summand.  This handles cases like
                # ``(\sum_{i=1}^{n} i)/2`` where the division is
                # outside the summation.
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
        # Remove any leading whitespace and stray multiplication
        # operator that may have been inserted before the summand
        tmp = summand.lstrip()
        if tmp.startswith('*'):
            tmp = tmp.lstrip('*').lstrip()
        summand = tmp
        # Break out variable and start value
        if '=' in lower_content:
            var, start_val = lower_content.split('=', 1)
        else:
            var = lower_content
            start_val = '1'
        # Build replacement
        replacement = (
            f'(sum(({summand}) for {var} in '
            f'range(int({start_val}), int({upper_content}) + 1)))'
        )
        expr = expr[:idx] + replacement + expr[pos:]
    return expr


def tokenize_for_multiplication(expr: str) -> List[str]:
    """Tokenize an expression for implicit multiplication detection.

    The tokenizer recognizes numbers, names (including dotted names),
    exponentiation operators, arithmetic operators, commas and
    parentheses/brackets.  Whitespace is ignored.  The result is
    intended solely for detecting places where multiplication
    operators may have been omitted in the LaTeX source.

    Parameters
    ----------
    expr : str
        The Python expression to tokenize.

    Returns
    -------
    list[str]
        A list of tokens in order of appearance.
    """
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
        # Handle exponentiation '**'
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
        # Any other character is treated as a single token
        tokens.append(ch)
        i += 1
    return tokens


def insert_implicit_multiplication(expr: str) -> str:
    """Insert explicit ``*`` operators where multiplication is implied.

    The rules implemented are deliberately conservative: a ``*`` is
    inserted between two adjacent tokens when the left token is
    numeric, alphabetic, an underscore, a closing parenthesis or
    bracket and the right token is alphabetic, numeric, an
    underscore, an opening parenthesis or opening bracket.  Function
    calls (e.g., ``max(``, ``math.log(``) are exempted from this
    insertion.  Dotted names like ``math.log`` are treated as
    single tokens to avoid splitting module and attribute access.

    Parameters
    ----------
    expr : str
        The Python expression potentially missing multiplication.

    Returns
    -------
    str
        The expression with multiplication operators inserted.
    """
    tokens = tokenize_for_multiplication(expr)
    if not tokens:
        return expr
    result: List[str] = []
    # Recognized function names that should not trigger implicit multiplication
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
            """Return True if the token ``t`` represents a numeric or
            alphabetic value (identifier or number).  This test does
            not consider parentheses or brackets to be values for the
            purpose of implicit multiplication.  Names containing dots
            (e.g. ``math.log``) are treated as single tokens.
            """
            return bool(t and (t[0].isdigit() or t[0].isalpha() or t[0] in '._'))
        def is_opening(t: str) -> bool:
            return t in {'(', '['}
        if (is_value(left) or left.endswith(')') or left.endswith(']')) and (is_opening(right) or is_value(right)):
            # Do not insert multiplication before control keywords such as
            # 'for' or 'in' which appear inside generator expressions.
            if right in {'for', 'in'}:
                continue
            # Also avoid inserting immediately after 'for' or 'in'
            if left in {'for', 'in'}:
                continue
            # Skip insertion when dealing with LaTeX control sequences such
            # as '\\sum_', which should remain intact until parsed by
            # ``parse_sum``.
            if left.startswith('\\') or right.startswith('\\'):
                continue
            # Skip insertion after the 'sum_' prefix of a summation
            if left.startswith('sum_') or right.startswith('sum_'):
                continue
            # Avoid inserting between the closing parenthesis of a
            # summation upper bound and the summand variable.  In the
            # token stream this pattern looks like: ... '^', '(', 'n', ')', 'i'.
            # When left is ')' and two positions back there is a caret,
            # it indicates we just closed the upper bound of a sum.
            if left == ')' and i >= 3 and tokens[i - 3] == '^':
                continue
            # Avoid inserting '*' for function calls
            if right == '(':
                if left in function_tokens:
                    continue
                if left.startswith('math.') and left in function_tokens:
                    continue
            if left in {'+', '-', '*', '/', '**', '%', ','}:
                continue
            result.append('*')
    # Reconstruct expression.  Retain spacing after control keywords
    # like 'for' and 'in' so that Python can parse generator
    # expressions correctly.
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
    """Transform a LaTeX expression into a Python expression.

    See the module documentation for a detailed description of the
    substitutions performed.  The function applies fraction and
    summation expansion, converts common LaTeX functions to their
    Python equivalents, strips unneeded commands, and inserts
    multiplication operators where omitted.

    Parameters
    ----------
    expr : str
        The LaTeX expression to convert.  It may include a leading
        assignment (e.g., ``Foo = ...``) and math mode delimiters
        ``$`` or ``$$``.

    Returns
    -------
    str
        The corresponding Python expression.
    """
    # Remove math delimiters
    expr = expr.replace('$$', '').replace('$', '').strip()
    # Remove leading assignment of the form ``Var = expr`` only when
    # the formula does not begin with a LaTeX control sequence (e.g.,
    # ``\sum_{i=1}`).  An '=' appearing in the lower bound of a
    # summation should not trigger assignment removal.
    stripped = expr.lstrip()
    if stripped and stripped[0] != '\\' and '=' in expr:
        expr = expr.split('=', 1)[1]
    expr = expr.replace('\n', '')
    # Drop LaTeX sizing/spacing commands
    expr = expr.replace('\\left', '').replace('\\right', '')
    expr = expr.replace('\\,', '').replace('\\;', '').replace('\\!', '').replace('\\:', '')
    # Convert \text{...} into plain text (spaces removed)
    expr = re.sub(r'\\text\s*{\s*([^}]*)\s*}', lambda m: m.group(1).strip().replace(' ', ''), expr)
    # Convert square brackets used for expectations or indexing
    expr = expr.replace('[', '_').replace(']', '')
    # Flatten subscripts with braces: ``_{foo}`` -> ``_foo`` except after \sum
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
    # Expand fractions before converting braces to parentheses
    expr = parse_frac(expr)
    # Convert product symbols
    expr = expr.replace('\\cdot', '*').replace('\\times', '*')
    # Convert min/max
    expr = expr.replace('\\max', 'max').replace('\\min', 'min')
    # Convert logarithms and exponentials
    expr = expr.replace('\\log', 'math.log').replace('\\ln', 'math.log').replace('\\exp', 'math.exp')
    # Convert square roots
    expr = re.sub(r'\\sqrt\s*{([^}]*)}', lambda m: f'(math.sqrt({m.group(1)}))', expr)
    # Strip backslashes from remaining commands (preserve some)
    def strip_backslash(match: re.Match) -> str:
        cmd = match.group(1)
        if cmd in {'frac', 'sum', 'cdot', 'times', 'max', 'min', 'log', 'ln', 'exp', 'sqrt', 'left', 'right'}:
            return match.group(0)
        return cmd
    expr = re.sub(r'\\([a-zA-Z]+)', strip_backslash, expr)
    # Convert remaining braces to parentheses
    expr = expr.replace('{', '(').replace('}', ')')
    # Expand summations
    expr = parse_sum(expr)
    # Replace caret with Python power operator
    expr = expr.replace('^', '**')
    # Insert implicit multiplication
    expr = insert_implicit_multiplication(expr)
    return expr


def evaluate_formula(formula: str, variables: Dict[str, float]) -> float:
    """Evaluate a LaTeX formula given a variable mapping.

    The formula is translated to Python syntax via
    :func:`latex_to_python` and then evaluated using Python's ``eval``
    with a restricted global namespace.  Provided variables are
    injected into the evaluation environment along with the ``math``
    module, built-in arithmetic functions and constants.

    Parameters
    ----------
    formula : str
        A string containing a LaTeX mathematical expression.  It
        may include a leading assignment (e.g. ``Foo = ...``) which
        will be ignored during evaluation.
    variables : Dict[str, float]
        A mapping from variable names (as they appear in the
        converted Python expression) to numeric values.

    Returns
    -------
    float
        The numeric result of the expression.
    """
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
    result = eval(expression, {'__builtins__': {}}, env)
    return float(result)
