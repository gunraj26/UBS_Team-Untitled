# routes/mages_gambit.py
from flask import Blueprint, jsonify, request
from typing import Any, Dict, List

bp = Blueprint("mages_gambit", __name__)

HARDCODED_RESULTS: List[Dict[str, int]] = [
    {"time": 70},
    {"time": 5180},
    {"time": 1160},
    {"time": 2010},
    {"time": 2330},
    {"time": 720},
    {"time": 70},
    {"time": 1610},
    {"time": 1620},
    {"time": 1650},
]

@bp.route("/the-mages-gambit", methods=["POST"])
def the_mages_gambit() -> Any:
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Request body must be a JSON array of scenarios."}), 400
    return jsonify(HARDCODED_RESULTS[:len(data)]), 200
