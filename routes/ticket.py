import logging
from flask import request, jsonify
from routes import app   # reuse the same Flask app instance

logger = logging.getLogger(__name__)

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def latency_points(customer_loc, center_loc):
    d = manhattan_distance(customer_loc, center_loc)
    if d <= 2:
        return 30
    elif d <= 4:
        return 20
    else:
        return 0

@app.route('/ticketing-agent', methods=['POST'])
def ticketing_agent():
    payload = request.get_json(force=True)
    customers = payload.get("customers", [])
    concerts = payload.get("concerts", [])
    priority_map = payload.get("priority", {}) or {}

    # Preprocess concerts
    concert_centers = {
        c["name"]: (int(c["booking_center_location"][0]), int(c["booking_center_location"][1]))
        for c in concerts
    }

    result = {}
    for cust in customers:
        cname = cust["name"]
        vip_pts = 100 if cust["vip_status"] else 0
        cx, cy = int(cust["location"][0]), int(cust["location"][1])
        card = cust["credit_card"]
        preferred_concert = priority_map.get(card)

        best_score = -1
        best_dist = float("inf")
        best_con = None

        for con_name, center in concert_centers.items():
            lat_pts = latency_points((cx, cy), center)
            card_pts = 50 if preferred_concert == con_name else 0
            total = vip_pts + card_pts + lat_pts
            dist = manhattan_distance((cx, cy), center)

            if (total > best_score) or (total == best_score and dist < best_dist):
                best_score = total
                best_dist = dist
                best_con = con_name

        result[cname] = best_con

    logging.info(f"Ticketing result: {result}")
    return jsonify(result)
