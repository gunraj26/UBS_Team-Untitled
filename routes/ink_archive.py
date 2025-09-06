from flask import Blueprint, request, jsonify
import math

bp = Blueprint('ink_archive', __name__)

def find_best_cycle(goods, ratios):
    """Find the most profitable trading cycle"""
    n = len(goods)
    best_gain = 0
    best_cycle = []
    
    # Create adjacency list for easier lookups
    graph = {}
    for i in range(n):
        graph[i] = {}
    
    for ratio in ratios:
        from_good, to_good, rate = int(ratio[0]), int(ratio[1]), ratio[2]
        if rate > 0:
            graph[from_good][to_good] = rate
    
    # Try all possible cycles of length 3 and 4
    # Length 3 cycles
    for i in range(n):
        for j in graph.get(i, {}):
            for k in graph.get(j, {}):
                if i in graph.get(k, {}):
                    # Found a 3-cycle: i -> j -> k -> i
                    rate = graph[i][j] * graph[j][k] * graph[k][i]
                    if rate > 1:
                        gain = (rate - 1) * 100
                        if gain > best_gain:
                            best_gain = gain
                            best_cycle = [goods[i], goods[j], goods[k], goods[i]]
    
    # Length 4 cycles
    for i in range(n):
        for j in graph.get(i, {}):
            for k in graph.get(j, {}):
                for l in graph.get(k, {}):
                    if i in graph.get(l, {}):
                        # Found a 4-cycle: i -> j -> k -> l -> i
                        rate = graph[i][j] * graph[j][k] * graph[k][l] * graph[l][i]
                        if rate > 1:
                            gain = (rate - 1) * 100
                            if gain > best_gain:
                                best_gain = gain
                                best_cycle = [goods[i], goods[j], goods[k], goods[l], goods[i]]
    
    return best_cycle, best_gain

@bp.route('/The-Ink-Archive', methods=['POST'])
def ink_archive():
    try:
        data = request.json
        results = []
        
        for challenge in data:
            goods = challenge['goods']
            ratios = challenge['ratios']
            
            path, gain = find_best_cycle(goods, ratios)
            
            results.append({
                "path": path,
                "gain": gain
            })
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500