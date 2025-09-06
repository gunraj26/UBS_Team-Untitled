import logging
from flask import request, jsonify
from routes import app
import numpy as np
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)

def impute_series(series):
    """
    Impute missing values using blended strategy:
    - Global trend: smoothing spline
    - Local AR (order 3): rolling neighbor average
    - Blend: 0.7 * spline + 0.3 * AR
    """
    n = len(series)
    x = np.arange(n)
    arr = np.array([np.nan if v is None else v for v in series], dtype=np.float64)

    not_nan = ~np.isnan(arr)
    if not not_nan.any():
        return np.zeros_like(arr).tolist()
    if not_nan.sum() == 1:
        return np.full_like(arr, arr[not_nan][0]).tolist()

    # --- Global smoothing spline ---
    try:
        spline = UnivariateSpline(x[not_nan], arr[not_nan], s=len(arr)*0.1)
        spline_pred = spline(x)
    except Exception as e:
        logger.warning("Spline failed: %s", e)
        spline_pred = np.interp(x, x[not_nan], arr[not_nan])

    # --- Local AR (order 3) ---
    ar_pred = np.copy(spline_pred)
    for i in range(n):
        if np.isnan(arr[i]):
            # Look at last 3 known/predicted values
            left_vals = []
            for k in range(1, 4):
                if i-k >= 0:
                    left_vals.append(ar_pred[i-k])
            if left_vals:
                ar_pred[i] = np.mean(left_vals)

    # --- Blend spline + AR ---
    blended = 0.7 * spline_pred + 0.3 * ar_pred

    # Replace NaNs/Infs
    blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)
    return blended.tolist()


@app.route('/blankety', methods=['POST'])
def blankety():
    data = request.get_json(force=True)
    series_list = data.get("series", [])

    completed = []
    for s in series_list:
        try:
            completed.append(impute_series(s))
        except Exception as e:
            logger.error("Error imputing series: %s", e)
            completed.append([0.0 for _ in s])

    return jsonify({"answer": completed})
