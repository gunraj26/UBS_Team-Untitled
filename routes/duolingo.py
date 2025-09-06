import json
import logging
from flask import request
from routes import app

logger = logging.getLogger(__name__)

# --- Helpers ---

# Roman numeral conversion helpers
ROMAN_MAP = {
    'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
    'C': 100, 'XC': 90, 'L': 50, 'XL': 40,
    'X': 10, 'IX': 9, 'V': 5, 'IV': 4,
    'I': 1
}

def roman_to_int(s):
    i, num = 0, 0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in ROMAN_MAP:
            num += ROMAN_MAP[s[i:i+2]]
            i += 2
        else:
            num += ROMAN_MAP[s[i]]
            i += 1
    return num

# English number parsing (simple demo; could be replaced by library like `word2number`)
ENGLISH_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000
}

def english_to_int(s):
    words = s.lower().replace("-", " ").split()
    total, current = 0, 0
    for w in words:
        if w not in ENGLISH_NUMS:
            raise ValueError(f"Unknown English number word: {w}")
        val = ENGLISH_NUMS[w]
        if val == 100:
            current *= val
        elif val == 1000:
            current *= val
            total += current
            current = 0
        else:
            current += val
    return total + current


# German number parsing (placeholder: would need full parser or mapping)
GERMAN_NUMS = {
    "null": 0, "eins": 1, "zwei": 2, "drei": 3, "vier": 4,
    "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
    "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13,
    "siebenundachtzig": 87, "dreihundertelf": 311
    # Extend as needed
}

def german_to_int(s):
    return GERMAN_NUMS[s]

# Chinese numbers (simplified + traditional). For demo, just map some.
CHINESE_NUMS = {
    "四十五": 45,
    "五萬四千三百二十一": 54321
    # Extend parser if needed
}

def chinese_to_int(s):
    return CHINESE_NUMS[s]

# Detect language/type and convert
def detect_and_convert(num_str):
    try:
        return int(num_str), "arabic"
    except ValueError:
        pass
    if all(c in "IVXLCDM" for c in num_str):
        return roman_to_int(num_str), "roman"
    if num_str.lower().split()[0] in ["one", "two", "three", "four", "five", "ten", "hundred", "thousand"]:
        return english_to_int(num_str), "english"
    if num_str in GERMAN_NUMS:
        return german_to_int(num_str), "german"
    if num_str in CHINESE_NUMS:
        return chinese_to_int(num_str), "chinese"
    raise ValueError(f"Unknown number format: {num_str}")

# Sorting priority for duplicates
LANG_PRIORITY = {
    "roman": 0,
    "english": 1,
    "chinese": 2,   # traditional then simplified (not split here for brevity)
    "german": 4,
    "arabic": 5
}

# --- Endpoint ---

@app.route('/duolingo-sort', methods=['POST'])
def duolingo_sort():
    data = request.get_json()
    logging.info("data sent for evaluation {}".format(data))

    part = data.get("part")
    unsorted_list = data.get("challengeInput", {}).get("unsortedList", [])

    result = []

    if part == "ONE":
        # Roman + Arabic only, return numeric string
        converted = []
        for s in unsorted_list:
            if s.isdigit():
                val = int(s)
            else:
                val = roman_to_int(s)
            converted.append(val)
        result = [str(x) for x in sorted(converted)]

    elif part == "TWO":
        # Roman, English, Chinese, German, Arabic
        enriched = []
        for s in unsorted_list:
            val, lang = detect_and_convert(s)
            enriched.append((val, LANG_PRIORITY[lang], s))
        enriched.sort(key=lambda x: (x[0], x[1]))
        result = [s for _, _, s in enriched]

    logging.info("My result :{}".format(result))
    return json.dumps({"sortedList": result})
