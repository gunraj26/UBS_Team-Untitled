import os
import re
from typing import List, Tuple
import logging
from flask import request, jsonify
from routes import app
from flask import Flask, request, jsonify, abort
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# --- Config / logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mst-openai")


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")



client = OpenAI(api_key=api_key)


# --------------------------------------------------------
# Helpers: Roman <-> int
# --------------------------------------------------------

_ROMAN_VALS = {
    'I': 1, 'V': 5, 'X': 10, 'L': 50,
    'C': 100, 'D': 500, 'M': 1000
}
_ROMAN_PATTERN = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})"
    r"(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)

def is_roman(s: str) -> bool:
    return bool(_ROMAN_PATTERN.match(s))

def roman_to_int(s: str) -> int:
    total, i = 0, 0
    while i < len(s):
        v = _ROMAN_VALS[s[i]]
        if i + 1 < len(s):
            v_next = _ROMAN_VALS[s[i + 1]]
            if v < v_next:
                total += v_next - v
                i += 2
                continue
        total += v
        i += 1
    return total

def is_arabic_digits(s: str) -> bool:
    return s.isdigit()

# --------------------------------------------------------
# LLM parser using chat.completions
# --------------------------------------------------------

TIE_ORDER = ["roman", "english", "zh-hant", "zh-hans", "german", "arabic"]
TIE_ORDER_INDEX = {k: i for i, k in enumerate(TIE_ORDER)}

def guess_zh_variant(text: str) -> str:
    if any(ch in text for ch in "萬億貳參陸"):
        return "zh-hant"
    if any(ch in text for ch in "万亿贰叁陆"):
        return "zh-hans"
    return "zh-hant"

def llm_parse_number(original: str) -> Tuple[int, str]:
    system = (
        "You are a number parser. Convert the given string into JSON with two keys: "
        "`value` (an integer >=0) and `category` (one of roman, english, zh-hant, zh-hans, german, arabic). "
        "Examples: {\"value\": 42, \"category\": \"english\"}. "
        "If ambiguous zh (like 四十五), prefer zh-hant."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Text: {original}\nReply ONLY as JSON."}
            ],
            temperature=0
        )
        content_text = resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")

    import json
    try:
        parsed = json.loads(content_text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON for '{original}': {e}\nGot: {content_text}")

    value = parsed.get("value")
    category = parsed.get("category")

    if not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Invalid value for '{original}': {parsed}")

    if category not in TIE_ORDER_INDEX:
        if is_arabic_digits(original):
            category = "arabic"
        elif is_roman(original):
            category = "roman"
        else:
            category = guess_zh_variant(original)

    if is_arabic_digits(original):
        category = "arabic"
        value = int(original)

    if is_roman(original):
        local_val = roman_to_int(original)
        if not (1 <= local_val <= 3999):
            raise RuntimeError(f"Roman numeral out of range: '{original}'")
        value = local_val
        category = "roman"

    if category in ("zh-hant", "zh-hans"):
        category = guess_zh_variant(original)

    return value, category

# --------------------------------------------------------
# Sorting Logic
# --------------------------------------------------------

def sort_part_one(items: List[str]) -> List[str]:
    parsed: List[Tuple[int, str]] = []
    for s in items:
        s_stripped = s.strip()
        if not s_stripped:
            abort(400, description="Empty string detected in input.")
        if is_arabic_digits(s_stripped):
            val = int(s_stripped)
            parsed.append((val, s_stripped))
        elif is_roman(s_stripped):
            val = roman_to_int(s_stripped)
            parsed.append((val, s_stripped))
        else:
            val, cat = llm_parse_number(s_stripped)
            if cat not in ("roman", "arabic"):
                abort(400, description=f"Invalid token for Part 1: '{s_stripped}'")
            parsed.append((val, s_stripped))
    parsed.sort(key=lambda x: x[0])
    return [str(v) for v, _orig in parsed]

def sort_part_two(items: List[str]) -> List[str]:
    enriched: List[Tuple[int, int, str]] = []
    for s in items:
        s_stripped = s.strip()
        if not s_stripped:
            abort(400, description="Empty string detected in input.")
        if is_arabic_digits(s_stripped):
            val = int(s_stripped)
            cat = "arabic"
        elif is_roman(s_stripped):
            val = roman_to_int(s_stripped)
            cat = "roman"
        else:
            val, cat = llm_parse_number(s_stripped)
        tie_rank = TIE_ORDER_INDEX.get(cat, len(TIE_ORDER))
        enriched.append((val, tie_rank, s_stripped))
    enriched.sort(key=lambda t: (t[0], t[1]))
    return [s for (_v, _r, s) in enriched]

# --------------------------------------------------------
# HTTP Endpoint
# --------------------------------------------------------

@app.route("/duolingo-sort", methods=["POST"])
def duolingo_sort():
    if not request.is_json:
        abort(400, description="Content-Type must be application/json.")
    data = request.get_json(silent=True) or {}
    part = data.get("part")
    challenge_input = data.get("challengeInput") or {}
    items = challenge_input.get("unsortedList")
    if part not in ("ONE", "TWO"):
        abort(400, description="Field 'part' must be 'ONE' or 'TWO'.")
    if not isinstance(items, list) or any(not isinstance(x, str) for x in items):
        abort(400, description="'unsortedList' must be a list of strings.")
    try:
        if part == "ONE":
            sorted_list = sort_part_one(items)
        else:
            sorted_list = sort_part_two(items)
    except RuntimeError as e:
        abort(400, description=str(e))
    return jsonify({"sortedList": sorted_list})


