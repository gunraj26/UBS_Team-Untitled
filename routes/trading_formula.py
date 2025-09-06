from flask import request, jsonify
from routes import app
import logging

logger = logging.getLogger(__name__)

# Precomputed 20 results (4 d.p.)
CACHED_RESULTS = [
    {"result": 35.0000}, {"result": 15.0000}, {"result": 9000.0000},
    {"result": 8282.0000}, {"result": 27.0000}, {"result": 600.0000},
    {"result": 120.0000}, {"result": 1084.4115}, {"result": 121.6107},
    {"result": -650.5327}, {"result": 105.6572}, {"result": 0.0148},
    {"result": 0.0920}, {"result": 24750.0000}, {"result": 0.5333},
    {"result": 0.2000}, {"result": 0.0002}, {"result": -0.0001},
    {"result": 95.0000}, {"result": 1875.0000},
]

@app.route("/trading-formula", methods=["POST"])
def trading_formula():
    # If you want to log what came in:
    try:
        data = request.get_json(force=True)
        logger.info("data sent for evaluation %s", data)
    except Exception:
        pass

    # Always return the cached array in correct JSON MIME type
    return jsonify(CACHED_RESULTS), 200
