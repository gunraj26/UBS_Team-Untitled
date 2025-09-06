from flask import Blueprint, request, jsonify
import math
import json

bp = Blueprint('ink_archive', __name__)

def find_best_cycle(goods, ratios):
    """Find the most profitable trading cycle"""
    n = len(goods)
    best_gain = 0
    best_cycle = []
    
    # Create rate lookup dictionary
    rate_dict = {}
    for ratio in ratios:
        from_good, to_good, rate = int(ratio[0]), int(ratio[1]), ratio[2]
        if rate > 0:
            rate_dict[(from_good, to_good)] = rate
    
    # Try all possible cycles of length 3, 4, and 5
    # Length 3 cycles
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and k != i:
                    if (i, j) in rate_dict and (j, k) in rate_dict and (k, i) in rate_dict:
                        rate = rate_dict[(i, j)] * rate_dict[(j, k)] * rate_dict[(k, i)]
                        if rate > 1:
                            gain = (rate - 1) * 100
                            if gain > best_gain:
                                best_gain = gain
                                best_cycle = [goods[i], goods[j], goods[k], goods[i]]
    
    # Length 4 cycles
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if len(set([i, j, k, l])) == 4:  # All different
                        if ((i, j) in rate_dict and (j, k) in rate_dict and 
                            (k, l) in rate_dict and (l, i) in rate_dict):
                            rate = (rate_dict[(i, j)] * rate_dict[(j, k)] * 
                                   rate_dict[(k, l)] * rate_dict[(l, i)])
                            if rate > 1:
                                gain = (rate - 1) * 100
                                if gain > best_gain:
                                    best_gain = gain
                                    best_cycle = [goods[i], goods[j], goods[k], goods[l], goods[i]]
    
    # Length 5 cycles (for goods with more than 4 items)
    if n >= 5:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        for m in range(n):
                            if len(set([i, j, k, l, m])) == 5:  # All different
                                if ((i, j) in rate_dict and (j, k) in rate_dict and 
                                    (k, l) in rate_dict and (l, m) in rate_dict and (m, i) in rate_dict):
                                    rate = (rate_dict[(i, j)] * rate_dict[(j, k)] * 
                                           rate_dict[(k, l)] * rate_dict[(l, m)] * rate_dict[(m, i)])
                                    if rate > 1:
                                        gain = (rate - 1) * 100
                                        if gain > best_gain:
                                            best_gain = gain
                                            best_cycle = [goods[i], goods[j], goods[k], goods[l], goods[m], goods[i]]
    
    return best_cycle, best_gain

@bp.route('/The-Ink-Archive', methods=['POST'])
def ink_archive():
    try:
        data = request.json
        
        # Log the payload
        print("PAYLOAD RECEIVED:")
        print(json.dumps(data, indent=2))
        print("=" * 50)
        
        results = []
        
        for challenge in data:
            goods = challenge['goods']
            ratios = challenge['ratios']
            
            path, gain = find_best_cycle(goods, ratios)
            
            results.append({
                "path": path,
                "gain": gain
            })
        
        print("RESPONSE SENT:")
        print(json.dumps(results, indent=2))
        print("=" * 50)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500