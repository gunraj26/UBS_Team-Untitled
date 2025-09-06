import os
import logging
from flask import request, jsonify
from routes import app
import requests
import json

logger = logging.getLogger(__name__)

# Get API key and project ID from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROJECT = os.environ.get("OPENAI_PROJECT")

if not OPENAI_API_KEY or not OPENAI_PROJECT:
    logger.warning("⚠️ OPENAI_API_KEY or OPENAI_PROJECT not set. Calls will fail.")

# Use raw requests (works with sk-proj keys)
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "OpenAI-Project": OPENAI_PROJECT
}


@app.route('/operation-safeguard', methods=['POST'])
def operation_safeguard():
    """
    Expects JSON like:
    {
      "challenge_one": {...},
      "challenge_two": [[x1, y1], [x2, y2], ...],
      "challenge_three": "log string"
    }
    """
    payload = request.get_json(force=True)

    # Challenge 1 (placeholder)
    challenge_one_out = "TODO"

    # Challenge 2 (coordinate analysis via OpenAI)
    coords = payload.get("challenge_two", [])
    challenge_two_out = None
    if coords:
        coords_text = "\n".join(str(c) for c in coords)
        prompt = f"""
You are analyzing suspicious network traffic coordinates.
Coordinates observed:
{coords_text}

Apply the following hints:
1. Look for spatial relationships, not raw numbers.
2. Remove anomalies that disrupt harmony.
3. Authentic coordinates should resemble something simple/significant.
4. Extract the critical number parameter.

Example:
If challenge_two = [
  [0,10],[1,10],[2,10],[3,10],[4,10],
  [0,9],[0,8],[1,8],[2,8],[3,8],[4,8],
  [0,7],[0,6],[1,6],[2,6],[3,6],[4,6],
  [4,5],[4,4],[3,4],[2,4],[1,4],[0,4],
  [0,3],[0,2],[1,2],[2,2],[3,2],[4,2],[4,1],[4,0],
  [50,50],[-3,99],[200,-150],[999,999],  # decoys
  [7,-42],[-100,500]                     # anomalies
]
This pattern is the digit "5".

Respond strictly in JSON, e.g.:
{{ "parameter": 5 }}
"""

        try:
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a skilled cryptanalyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0
            }
            resp = requests.post(OPENAI_URL, headers=HEADERS, data=json.dumps(data))
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                logger.info("Challenge 2 OpenAI response: %s", content)
                challenge_two_out = content
            else:
                logger.error("OpenAI API error: %s", resp.text)
                challenge_two_out = {"error": resp.text}
        except Exception as e:
            logger.error("Challenge 2 OpenAI call failed: %s", e)
            challenge_two_out = {"error": str(e)}

    # Challenge 3 (placeholder)
    challenge_three_out = "TODO"

    # Challenge 4 (placeholder)
    challenge_four_out = "TODO"

    return jsonify({
        "challenge_one": challenge_one_out,
        "challenge_two": challenge_two_out,
        "challenge_three": challenge_three_out,
        "challenge_four": challenge_four_out
    })
