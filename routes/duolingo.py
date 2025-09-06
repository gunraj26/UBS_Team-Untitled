import json
import logging
import re
from flask import request
from routes import app

logger = logging.getLogger(__name__)

# ---------------------------
# Roman numerals (case-insensitive)
# ---------------------------
ROMAN_PAIRS = [
    ("CM", 900), ("CD", 400), ("XC", 90), ("XL", 40), ("IX", 9), ("IV", 4)
]
ROMAN_SINGLE = {
    "M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1
}

def is_roman(s: str) -> bool:
    if not s: return False
    t = s.upper()
    return all(ch in "IVXLCDM" for ch in t)

def roman_to_int(s: str) -> int:
    t = s.upper()
    i, total = 0, 0
    while i < len(t):
        pair = t[i:i+2]
        found = False
        for p, val in ROMAN_PAIRS:
            if pair == p:
                total += val
                i += 2
                found = True
                break
        if not found:
            total += ROMAN_SINGLE[t[i]]
            i += 1
    return total

# ---------------------------
# English numbers (dependency-free, up to billions)
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
    if not words: return False
    dicts = (EN_UNITS | EN_TEENS | EN_TENS | EN_SCALES)
    return all(w in dicts for w in words)

def english_to_int(s: str) -> int:
    t = s.lower().replace("-", " ")
    t = re.sub(r"\band\b", " ", t)
    words = [w for w in t.split() if w]
    total, current = 0, 0
    for w in words:
        if w in EN_UNITS: current += EN_UNITS[w]
        elif w in EN_TEENS: current += EN_TEENS[w]
        elif w in EN_TENS: current += EN_TENS[w]
        elif w == "hundred":
            if current == 0: current = 1
            current *= 100
        else:  # thousand/million/billion
            scale = EN_SCALES[w]
            if current == 0: current = 1
            total += current * scale
            current = 0
    return total + current

# ---------------------------
# German numbers (robust + conservative)
# ---------------------------
import re

DE_UNITS = {
    "null":0,"ein":1,"eins":1,"zwei":2,"drei":3,"vier":4,"fuenf":5,"funf":5,
    "sechs":6,"sieben":7,"acht":8,"neun":9
}
DE_TEENS = {
    "zehn":10,"elf":11,"zwoelf":12,"zwolf":12,
    "dreizehn":13,"vierzehn":14,"fuenfzehn":15,"funfzehn":15,
    "sechzehn":16,"siebzehn":17,"achtzehn":18,"neunzehn":19
}
DE_TENS = {
    "zwanzig":20,"dreissig":30,"dreißig":30,"vierzig":40,
    "fuenfzig":50,"funfzig":50,"sechzig":60,"siebzig":70,"achtzig":80,"neunzig":90
}
# tokens used for quick-likelihood check
_DE_ALL = set(DE_UNITS) | set(DE_TEENS) | set(DE_TENS) | {
    "und","hundert","tausend","million","millionen","milliarde","milliarden"
}

def _de_norm(s: str) -> str:
    t = s.lower()
    # normalize umlauts/ß and remove separators
    t = (t.replace("ä","ae").replace("ö","oe").replace("ü","ue")
           .replace("ß","ss").replace("-", "").replace(" ", ""))
    # fold inflections of "ein*"
    t = (t.replace("eine","ein").replace("einen","ein").replace("einem","ein")
           .replace("einer","ein").replace("eines","ein"))
    # common small slips seen in data
    t = t.replace("eunzehn", "neunzehn")            # missing leading 'n'
    t = t.replace("reiund", "dreiund")              # dropped 'd'
    t = t.replace("dreisig", "dreissig")            # alt misspelling of 30
    # fix missing 't' in 'hundert' only when actually missing (avoid 'hundertt')
    t = re.sub(r"hunder(?!t)", "hundert", t)
    return t

def maybe_german(s: str) -> bool:
    t = _de_norm(s)
    return any(tok in t for tok in [
        "und","zig","zehn","hundert","tausend","million","milliard","null"
    ]) or t in _DE_ALL

def german_to_int(s: str) -> int:
    t = _de_norm(s)
    return _de_parse(t)

def _split_first(t: str, options: list[str]):
    # options must be ordered longest-first
    for w in options:
        i = t.find(w)
        if i != -1:
            return t[:i], t[i+len(w):], w
    return None, None, None

def _de_parse(t: str) -> int:
    if not t:
        return 0

    # Milliarde(n) = 1e9
    left, rest, w = _split_first(t, ["milliarden","milliarde"])
    if w:
        mult = _de_parse(left) if left else 1
        return mult * 1_000_000_000 + _de_parse(rest)

    # Million(en) = 1e6
    left, rest, w = _split_first(t, ["millionen","million"])
    if w:
        mult = _de_parse(left) if left else 1
        return mult * 1_000_000 + _de_parse(rest)

    # tausend = 1e3
    left, rest, w = _split_first(t, ["tausend"])
    if w:
        mult = _de_parse(left) if left else 1
        return mult * 1000 + _de_parse(rest)

    # hundert = 100
    left, rest, w = _split_first(t, ["hundert"])
    if w:
        mult = _de_parse(left) if left else 1
        return mult * 100 + _de_parse(rest)

    # direct matches
    if t in DE_TEENS: return DE_TEENS[t]
    if t in DE_TENS:  return DE_TENS[t]
    if t in DE_UNITS: return DE_UNITS[t]

    # unit-und-tens (e.g., siebenundachtzig)
    m = re.fullmatch(r"([a-z]+)und([a-z]+)", t)
    if m:
        u, ten = m.group(1), m.group(2)
        # tolerate trailing 'e' (rare), map 'eine'→'ein' already in _de_norm
        if u not in DE_UNITS and u.endswith("e") and u[:-1] in DE_UNITS:
            u = u[:-1]
        u_val  = DE_UNITS.get(u)
        ten_val = DE_TENS.get(ten)
        if u_val is not None and ten_val is not None:
            return ten_val + u_val

    # tens + unit glued (e.g., dreissigeins)
    for tk, tv in DE_TENS.items():
        if t.startswith(tk) and t[len(tk):] in DE_UNITS:
            return tv + DE_UNITS[t[len(tk):]]

    # no aggressive fallback — keep it strict to avoid wrong parses
    raise ValueError(f"Unknown German number: {t}")


# ---------------------------
# Chinese numbers (Traditional + Simplified)
# ---------------------------
CN_DIGITS = {
    "零":0,"〇":0,"○":0,"Ｏ":0,
    "一":1,"二":2,"兩":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9
}
CN_MULT = { "十":10,"百":100,"千":1000 }
CN_BIG  = { "万":10_000,"萬":10_000,"亿":100_000_000,"億":100_000_000 }

TRAD_ONLY = set("萬億兩")
SIMP_ONLY = set("万亿两")

def is_chinese(s: str) -> bool:
    return any(ch in CN_DIGITS or ch in CN_MULT or ch in CN_BIG for ch in s)

def chinese_script(s: str) -> str:
    if any(ch in TRAD_ONLY for ch in s): return "chinese_trad"
    if any(ch in SIMP_ONLY for ch in s): return "chinese_simp"
    # ambiguous (e.g., 四十五) — prefer Traditional for tie-break order
    return "chinese_trad"

def _parse_cn_section(sec: str) -> int:
    if not sec:
        return 0

    # Special pure "十" patterns
    if "十" in sec and all(ch not in "百千" for ch in sec):
        left, _, right = sec.partition("十")
        tens = CN_DIGITS.get(left, 1) if left != "" else 1
        ones = CN_DIGITS.get(right, 0) if right != "" else 0
        return tens*10 + ones

    total, num, i = 0, 0, 0
    while i < len(sec):
        ch = sec[i]
        if ch in CN_DIGITS:
            num = CN_DIGITS[ch]
            i += 1
            if i < len(sec) and sec[i] in CN_MULT:
                total += (num if num != 0 else 1) * CN_MULT[sec[i]]
                num = 0
                i += 1
            else:
                # optional zero skipping (e.g., 零)
                if i < len(sec) and sec[i] in CN_DIGITS and CN_DIGITS[sec[i]] == 0:
                    i += 1
        elif ch in CN_MULT:
            total += (num if num != 0 else 1) * CN_MULT[ch]
            num = 0
            i += 1
        elif ch in CN_BIG:
            total = (total + (num if num != 0 else 0)) * CN_BIG[ch]
            num = 0
            i += 1
        else:
            i += 1
    return total + num

def chinese_to_int(s: str) -> int:
    t = s.strip()
    if t in ("零","〇","○","Ｏ"):
        return 0
    # split by 亿/億, then inside by 万/萬
    parts_yi = re.split(r"[亿億]", t)
    vals_yi = []
    for seg_yi in parts_yi:
        parts_wan = re.split(r"[万萬]", seg_yi)
        subtotal = 0
        for j, seg_w in enumerate(parts_wan):
            val = _parse_cn_section(seg_w)
            power = len(parts_wan) - j - 1
            subtotal += val * (10_000 ** power)
        vals_yi.append(subtotal)
    total = 0
    for i, v in enumerate(vals_yi):
        power = len(vals_yi) - i - 1
        total += v * (100_000_000 ** power)
    return total

# ---------------------------
# Arabic numerals (tolerant to separators)
# ---------------------------
def is_arabic(s: str) -> bool:
    t = s.strip()
    t = t.replace(",", "").replace("_", "").replace(" ", "")
    return t.isdigit()

def arabic_to_int(s: str) -> int:
    t = s.strip().replace(",", "").replace("_", "").replace(" ", "")
    return int(t)

# ---------------------------
# Language detection + priority for duplicates
# ---------------------------
def detect_and_convert(num_str: str):
    s = num_str.strip()

    # Arabic
    if is_arabic(s):
        return arabic_to_int(s), "arabic"

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
        logging.info("My result :[]")
        return json.dumps({"sortedList": []})

    if part == "ONE":
        # Roman + Arabic only; return numeric strings
        ints = []
        for s in unsorted_list:
            st = s.strip()
            if is_arabic(st):
                ints.append(arabic_to_int(st))
            elif is_roman(st):
                ints.append(roman_to_int(st))
            else:
                # be forgiving: try general detect if someone slipped in English/Chinese/German
                val, _ = detect_and_convert(st)
                ints.append(val)
        result = [str(x) for x in sorted(ints)]
        logging.info("My result :{}".format(result))
        return json.dumps({"sortedList": result})

    # Part TWO
    enriched = []
    for s in unsorted_list:
        val, lang = detect_and_convert(s)
        enriched.append((val, LANG_PRIORITY[lang], s))
    enriched.sort(key=lambda t: (t[0], t[1]))
    result = [s for _, _, s in enriched]

    logging.info("My result :{}".format(result))
    return json.dumps({"sortedList": result})
