import json
import logging
import re
from flask import request
from routes import app

logger = logging.getLogger(__name__)

# ---------------------------
# Roman numerals
# ---------------------------
ROMAN_PAIRS = [
    ("CM", 900), ("CD", 400), ("XC", 90), ("XL", 40), ("IX", 9), ("IV", 4)
]
ROMAN_SINGLE = {
    "M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1
}

def is_roman(s: str) -> bool:
    return bool(s) and all(ch in "IVXLCDM" for ch in s)

def roman_to_int(s: str) -> int:
    i, total = 0, 0
    while i < len(s):
        pair = s[i:i+2]
        found = False
        for p, val in ROMAN_PAIRS:
            if pair == p:
                total += val
                i += 2
                found = True
                break
        if not found:
            total += ROMAN_SINGLE[s[i]]
            i += 1
    return total

# ---------------------------
# English numbers (dependency-free)
# ---------------------------
EN_UNITS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9
}
EN_TEENS = {
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19
}
EN_TENS = {
    "twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90
}
EN_SCALES = {
    "hundred":100,"thousand":1000,"million":1_000_000,"billion":1_000_000_000
}

def maybe_english(s: str) -> bool:
    t = s.lower().replace("-", " ")
    t = re.sub(r"\band\b", " ", t)
    words = [w for w in t.split() if w]
    if not words:
        return False
    dicts = (EN_UNITS | EN_TEENS | EN_TENS | EN_SCALES)
    return all(w in dicts for w in words)

def english_to_int(s: str) -> int:
    t = s.lower().replace("-", " ")
    t = re.sub(r"\band\b", " ", t)
    words = [w for w in t.split() if w]
    total, current = 0, 0
    for w in words:
        if w in EN_UNITS:
            current += EN_UNITS[w]
        elif w in EN_TEENS:
            current += EN_TEENS[w]
        elif w in EN_TENS:
            current += EN_TENS[w]
        elif w == "hundred":
            if current == 0:
                current = 1
            current *= 100
        else:
            # thousand/million/billion
            scale = EN_SCALES[w]
            if current == 0:
                current = 1
            total += current * scale
            current = 0
    return total + current

# ---------------------------
# German numbers (practical parser)
# ---------------------------
DE_UNITS = {
    "null":0,"eins":1,"ein":1,"eine":1,"einem":1,"einen":1,"einer":1,
    "zwei":2,"drei":3,"vier":4,"fuenf":5,"fünf":5,"sechs":6,"sieben":7,"acht":8,"neun":9
}
DE_TEENS = {
    "zehn":10,"elf":11,"zwoelf":12,"zwölf":12,"dreizehn":13,"vierzehn":14,"fuenfzehn":15,"fünfzehn":15,
    "sechzehn":16,"siebzehn":17,"achtzehn":18,"neunzehn":19
}
DE_TENS = {
    "zwanzig":20,"dreissig":30,"dreißig":30,"vierzig":40,"fuenfzig":50,"fünfzig":50,"sechzig":60,
    "siebzig":70,"achtzig":80,"neunzig":90
}
# Normalize ß -> ss and ue/ae/oe options are handled by allowing both spellings above.

def _de_norm(s: str) -> str:
    # preserve umlauts since we mapped both, but normalize casing and strip spaces
    return s.lower().replace(" ", "")

def maybe_german(s: str) -> bool:
    t = _de_norm(s)
    # quick heuristics
    return any(tok in t for tok in ["und","zig","zehn","hundert","tausend"]) or t in (DE_UNITS | DE_TEENS | DE_TENS)

def german_to_int(s: str) -> int:
    t = _de_norm(s)
    # handle thousands
    if "tausend" in t:
        prefix, rest = t.split("tausend", 1)
        k = german_to_int(prefix) if prefix else 1
        return k * 1000 + (german_to_int(rest) if rest else 0)
    # handle hundreds
    if "hundert" in t:
        prefix, rest = t.split("hundert", 1)
        h = german_to_int(prefix) if prefix else 1
        return h * 100 + (german_to_int(rest) if rest else 0)
    # direct teens
    if t in DE_TEENS:
        return DE_TEENS[t]
    # direct tens
    if t in DE_TENS:
        return DE_TENS[t]
    # unit + "und" + tens  (e.g., siebenundachtzig)
    m = re.fullmatch(r"([a-zäöüß]+)und([a-zäöüß]+)", t)
    if m:
        u, tens = m.group(1), m.group(2)
        # Special case "ein" as unit = 1
        u_val = DE_UNITS.get(u, None)
        if u_val is None and u == "ein":
            u_val = 1
        if u_val is None:
            # Some units may come as "einen"/"eine" already in DE_UNITS
            u_val = DE_UNITS.get(u, None)
        tens_val = DE_TENS.get(tens, None)
        if u_val is not None and tens_val is not None:
            return tens_val + u_val
    # fallback to units
    if t in DE_UNITS:
        return DE_UNITS[t]
    raise ValueError(f"Unknown German number: {s}")

# ---------------------------
# Chinese numbers (Traditional + Simplified)
# ---------------------------
CN_DIGITS = {
    "零":0,"〇":0,"○":0,"Ｏ":0,
    "一":1,"二":2,"兩":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9
}
CN_MULT = {
    "十":10,"百":100,"千":1000
}
CN_BIG = {
    "万":10_000,"萬":10_000,"亿":100_000_000,"億":100_000_000
}
TRAD_ONLY = set("萬億兩")
SIMP_ONLY = set("万亿两")

def is_chinese(s: str) -> bool:
    return any(ch in CN_DIGITS or ch in CN_MULT or ch in CN_BIG for ch in s)

def chinese_script(s: str) -> str:
    # Return "chinese_trad" or "chinese_simp"
    if any(ch in TRAD_ONLY for ch in s):
        return "chinese_trad"
    if any(ch in SIMP_ONLY for ch in s):
        return "chinese_simp"
    # ambiguous forms (e.g., 四十五) — treat as Traditional for tie-break order
    return "chinese_trad"

def _parse_cn_section(sec: str) -> int:
    """
    Parse section without 万/萬/亿/億, handling 千/百/十 and digits.
    Examples: "二千一百七十二" -> 2172, "一百零三" -> 103, "十二" -> 12, "二十" -> 20
    """
    if not sec:
        return 0
    total = 0
    num = 0
    last_unit = 1
    i = 0
    # Special-case: if section is just "十" or "二十" style
    if "十" in sec and all(ch not in "百千" for ch in sec):
        parts = sec.split("十")
        left = parts[0]
        right = "".join(parts[1:])
        tens = CN_DIGITS.get(left, 1) if left != "" else 1
        ones = CN_DIGITS.get(right, 0) if right != "" else 0
        return tens*10 + ones

    while i < len(sec):
        ch = sec[i]
        if ch in CN_DIGITS:
            num = CN_DIGITS[ch]
            i += 1
            # lookahead for unit
            if i < len(sec) and sec[i] in CN_MULT:
                unit = CN_MULT[sec[i]]
                total += (num if num != 0 else 1) * unit
                num = 0
                i += 1
            else:
                # might be trailing digit (ones place) or followed by '零'
                if i < len(sec) and sec[i] in CN_DIGITS and CN_DIGITS[sec[i]] == 0:
                    # skip zero
                    i += 1
                # accumulate and continue loop; we'll add at end
        elif ch in CN_MULT:
            unit = CN_MULT[ch]
            total += (num if num != 0 else 1) * unit
            num = 0
            i += 1
        elif ch in CN_BIG:
            # shouldn't appear here (handled at higher level), but just in case
            unit = CN_BIG[ch]
            total = (total + (num if num != 0 else 0)) * unit
            num = 0
            i += 1
        else:
            # ignore other characters (robustness)
            i += 1
    return total + num

def chinese_to_int(s: str) -> int:
    """
    Full parser handling 亿/億 and 万/萬 segments.
    """
    t = s.strip()
    if t in ("零","〇","○","Ｏ"):
        return 0

    # Split by 亿/億
    parts_yi = re.split(r"[亿億]", t)
    vals_yi = []

    # positions: [ ... , last ]
    # For A亿B => value = A*1e8 + parse(B)
    for idx, seg_yi in enumerate(parts_yi):
        # Within each 亿 segment, split by 万/萬
        parts_wan = re.split(r"[万萬]", seg_yi)
        subtotal = 0
        for jdx, seg_w in enumerate(parts_wan):
            val = _parse_cn_section(seg_w)
            # For e.g., "三萬二千" => parts_wan = ["三", "二千"]
            power = len(parts_wan) - jdx - 1  # 1 for 万 place, 0 for units
            subtotal += val * (10_000 ** power)
        vals_yi.append(subtotal)

    total = 0
    for idx, v in enumerate(vals_yi):
        power = len(vals_yi) - idx - 1  # 亿 power
        total += v * (100_000_000 ** power)
    return total

# ---------------------------
# Arabic numerals
# ---------------------------
def is_arabic(s: str) -> bool:
    return s.isdigit()

# ---------------------------
# Language detection + priority
# ---------------------------
def detect_and_convert(num_str: str):
    s = num_str.strip()
    # Arabic
    if is_arabic(s):
        return int(s), "arabic"
    # Roman
    if is_roman(s):
        return roman_to_int(s), "roman"
    # Chinese
    if is_chinese(s):
        return chinese_to_int(s), chinese_script(s)
    # English
    if maybe_english(s):
        return english_to_int(s), "english"
    # German
    if maybe_german(s):
        return german_to_int(s), "german"
    raise ValueError(f"Unknown number format: {num_str}")

LANG_PRIORITY = {
    "roman": 0,
    "english": 1,
    "chinese_trad": 2,
    "chinese_simp": 3,
    "german": 4,
    "arabic": 5,
}

# ---------------------------
# Endpoint
# ---------------------------
@app.route('/duolingo-sort', methods=['POST'])
def duolingo_sort():
    data = request.get_json()
    logging.info("data sent for evaluation {}".format(data))

    part = data.get("part")
    unsorted_list = data.get("challengeInput", {}).get("unsortedList", [])

    if not isinstance(unsorted_list, list):
        return json.dumps({"sortedList": []})

    if part == "ONE":
        # Roman + Arabic only
        ints = []
        for s in unsorted_list:
            st = s.strip()
            if is_arabic(st):
                ints.append(int(st))
            elif is_roman(st):
                ints.append(roman_to_int(st))
            else:
                # If an unexpected token appears, try broad detect; else fail fast
                val, _ = detect_and_convert(st)
                ints.append(val)
        result = [str(x) for x in sorted(ints)]
        logging.info("My result :{}".format(result))
        return json.dumps({"sortedList": result})

    # part TWO (default)
    enriched = []
    for s in unsorted_list:
        val, lang = detect_and_convert(s)
        enriched.append((val, LANG_PRIORITY[lang], s))
    enriched.sort(key=lambda t: (t[0], t[1]))
    result = [s for _, _, s in enriched]

    logging.info("My result :{}".format(result))
    return json.dumps({"sortedList": result})
