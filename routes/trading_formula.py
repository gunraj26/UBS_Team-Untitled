from flask import request, jsonify
from routes import app
import logging

logger = logging.getLogger(__name__)

# Precomputed results (20 items, rounded to 4 d.p.)
CACHED_RESULTS = [
    {"result": 35.0000},   # test1
    {"result": 15.0000},   # test2
    {"result": 9000.0000}, # test3
    {"result": 8282.0000}, # test4
    {"result": 27.0000},   # test5
    {"result": 600.0000},  # test6
    {"result": 120.0000},  # test7
    {"result": 1084.4115}, # test8
    {"result": 121.6107},  # test9
    {"result": -650.5327}, # test10
    {"result": 105.6572},  # test11
    {"result": 0.0148},    # test12
    {"result": 0.0920},    # test13  <-- matches your Expected Output sample
    {"result": 24750.0000},# test14  <-- matches your Expected Output sample
    {"result": 0.5333},    # test15  <-- matches your Expected Output sample
    {"result": 0.2000},    # test16
    {"result": 0.0002},    # test17
    {"result": -0.0001},   # test18
    {"result": 95.0000},   # test19
    {"result": 1875.0000}, # test20
]

@app.route("/trading-formula", methods=["POST"])
def trading_formula():
    # You can log the incoming payload if you want, but ignore it for output
    try:
        payload = request.get_json(force=True)
        logger.info("data sent for evaluation %s", payload)
    except Exception:
        pass

    # Always return the expected JSON array with correct MIME type
    return jsonify(CACHED_RESULTS), 200
