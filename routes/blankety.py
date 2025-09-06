import logging
from flask import request, jsonify
from routes import app
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

def impute_series(series):
    """
    Impute missing values with adaptive strategy:
    - all nulls -> zeros
    - short gaps -> linear interp
    - longer gaps -> cubic spline
    - final smoothing -> Savitzky-Golay filter
    """
    arr = np.array([np.nan if v is None else v for v in series], dtype=np.float64)
    n = len(arr)
    x = np.arange(n)

    not_nan = ~np.isnan(arr)

    if not not_nan.any():
        return np.zeros_like(arr).tolist()

    if not_nan.sum() == 1:
        # Only one known point: fill everything with that value
        val = arr[not_nan][0]
        return np.full_like(arr, val).tolist()

    try:
        # Choose interpolation method adaptively
        if not_nan.sum() < 4:
            # Too few points, stick to linear
            f = interp1d(x[not_nan], arr[not_nan], kind="linear", fill_value="extrapolate")
        else:
            # Use cubic spline for more shape preservation
            f = interp1d(x[not_nan], arr[not_nan], kind="cubic", fill_value="extrapolate")

        arr[~not_nan] = f(x[~not_nan])

    except Exception as e:
        logger.warning("Interpolation failed, falling back to linear: %s", e)
        arr[~not_nan] = np.interp(x[~not_nan], x[not_nan], arr[not_nan])

    # Apply smoothing filter if series is long
    if n >= 7:
        try:
            arr = savgol_filter(arr, window_length=min(21, n - (n+1)%2), polyorder=3)
        except Exception as e:
            logger.warning("Savitzky-Golay failed, skipping smoothing: %s", e)

    # Replace any NaNs/Infs that slipped through
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr.tolist()


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
