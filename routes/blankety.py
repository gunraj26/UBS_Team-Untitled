import logging
from flask import request, jsonify
from routes import app
import numpy as np

logger = logging.getLogger(__name__)

def impute_series(series):
    """
    Impute missing values (None) in a 1D list using linear interpolation.
    Falls back to nearest non-null value at edges.
    """
    # Convert None to np.nan
    arr = np.array([np.nan if v is None else v for v in series], dtype=np.float64)

    # Indices of non-nan values
    not_nan = ~np.isnan(arr)
    x = np.arange(len(arr))

    if not_nan.sum() == 0:
        # Entire series is null -> fill with zeros
        return np.zeros_like(arr).tolist()

    # Interpolate missing values
    arr[~not_nan] = np.interp(x[~not_nan], x[not_nan], arr[not_nan])
    return arr.tolist()


@app.route('/blankety', methods=['POST'])
def blankety():
    """
    POST endpoint for imputation.
    Input:
    {
      "series": [
        [0.12, 0.17, null, 0.30, 0.29, null, ...],
        ...
      ]
    }
    Output:
    {
      "answer": [
        [0.12, 0.17, 0.23, 0.30, 0.29, 0.31, ...],
        ...
      ]
    }
    """
    data = request.get_json(force=True)
    series_list = data.get("series", [])

    completed = []
    for s in series_list:
        try:
            completed.append(impute_series(s))
        except Exception as e:
            logger.error("Error imputing series: %s", e)
            completed.append([0.0 for _ in s])  # fallback

    return jsonify({"answer": completed})
