import os
import logging
from flask import request, jsonify
from routes import app
from openai import OpenAI

logger = logging.getLogger(__name__)

# Get API key from environment variable (Render/locally set OPENAI_API_KEY)
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)


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

    # Challenge 1 (placeholder, not yet implemented)
    challenge_one_in = payload.get("challenge_one", {})
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

        Respond strictly in JSON:
        {{ "parameter": <number> }}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a skilled cryptanalyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            logger.info("Challenge 2 OpenAI response: %s", content)
            challenge_two_out = content
        except Exception as e:
            logger.error("Challenge 2 OpenAI call failed: %s", e)
            challenge_two_out = {"error": str(e)}

    # Challenge 3 (placeholder)
    challenge_three_in = payload.get("challenge_three", "")
    challenge_three_out = "TODO"

    # Challenge 4 (placeholder)
    challenge_four_out = "TODO"

    return jsonify({
        "challenge_one": challenge_one_out,
        "challenge_two": challenge_two_out,
        "challenge_three": challenge_three_out,
        "challenge_four": challenge_four_out
    })
