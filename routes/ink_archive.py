# from flask import Blueprint, request, jsonify
# import math
# import json
# import logging

# log = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,  # Set minimum logging level
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# )


# bp = Blueprint('ink_archive', __name__)

# def find_best_cycle(goods, ratios):
#     """Find the most profitable trading cycle"""
#     n = len(goods)
#     best_gain = 0
#     best_cycle = []
    
#     # Create adjacency list for easier lookups
#     graph = {}
#     for i in range(n):
#         graph[i] = {}
    
#     for ratio in ratios:
#         from_good, to_good, rate = int(ratio[0]), int(ratio[1]), ratio[2]
#         if rate > 0:
#             graph[from_good][to_good] = rate
    
#     # Try all possible cycles of length 3 and 4
#     # Length 3 cycles
#     for i in range(n):
#         for j in graph.get(i, {}):
#             for k in graph.get(j, {}):
#                 if i in graph.get(k, {}):
#                     # Found a 3-cycle: i -> j -> k -> i
#                     rate = graph[i][j] * graph[j][k] * graph[k][i]
#                     if rate > 1:
#                         gain = (rate - 1) * 100
#                         if gain > best_gain:
#                             best_gain = gain
#                             best_cycle = [goods[i], goods[j], goods[k], goods[i]]
    
#     # Length 4 cycles
#     for i in range(n):
#         for j in graph.get(i, {}):
#             for k in graph.get(j, {}):
#                 for l in graph.get(k, {}):
#                     if i in graph.get(l, {}):
#                         # Found a 4-cycle: i -> j -> k -> l -> i
#                         rate = graph[i][j] * graph[j][k] * graph[k][l] * graph[l][i]
#                         if rate > 1:
#                             gain = (rate - 1) * 100
#                             if gain > best_gain:
#                                 best_gain = gain
#                                 best_cycle = [goods[i], goods[j], goods[k], goods[l], goods[i]]
    
#     return best_cycle, best_gain

# @bp.route('/The-Ink-Archive', methods=['POST'])
# def ink_archive():
#     try:
#         data = request.json
#         log.info("PAYLOAD RECEIVED:")
#         log.info(json.dumps(data, indent=2))
#         log.info("=" * 50)
#         results = []
        
#         for challenge in data:
#             goods = challenge['goods']
#             ratios = challenge['ratios']
            
#             path, gain = find_best_cycle(goods, ratios)
            
#             results.append({
#                 "path": path,
#                 "gain": gain
#             })
        
#         return jsonify(results)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    


from flask import Blueprint, request, jsonify
import math
import json
import logging
from collections import defaultdict

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

bp = Blueprint('ink_archive', __name__)

def find_best_cycle(goods, ratios, max_len=None):
    """
    Find the most profitable simple cycle (no repeated nodes except start).
    Returns (best_path_by_name, best_gain_times_100).
    """
    n = len(goods)
    if max_len is None:
        max_len = n  # explore up to number of goods

    # build adjacency
    graph = defaultdict(dict)
    for f, t, r in ratios:
        f = int(f); t = int(t)
        r = float(r)
        if r > 0:
            graph[f][t] = r

    best_gain = -1.0
    best_path_idx = None

    # DFS over simple cycles
    def dfs(start, node, prod, path, depth):
        nonlocal best_gain, best_path_idx

        if depth >= 1 and start in graph[node]:
            rate = prod * graph[node][start]
            gain = (rate - 1.0) * 100.0
            if rate > 1.0 and gain > best_gain:
                best_gain = gain
                best_path_idx = path + [start]

        if depth >= max_len - 1:
            return

        for nxt, r in graph[node].items():
            # allow returning to start only via the closing step above;
            # otherwise avoid revisiting nodes (simple cycles)
            if nxt != start and nxt in path:
                continue
            dfs(start, nxt, prod * r, path + [nxt], depth + 1)

    for i in range(n):
        dfs(i, i, 1.0, [i], 0)

    if best_path_idx is None:
        return [], 0.0

    best_path_names = [goods[idx] for idx in best_path_idx]
    # Round gain to something tidy but keep precision
    return best_path_names, round(best_gain, 12)

@bp.route('/The-Ink-Archive', methods=['POST'])
def ink_archive():
    try:
        data = request.get_json(force=True, silent=False)
        log.info("PAYLOAD RECEIVED:")
        log.info(json.dumps(data, indent=2))
        log.info("=" * 50)

        results = []
        for challenge in data:
            goods = challenge['goods']
            # Accept either "ratios" or "rates"
            ratios = challenge.get('ratios', challenge.get('rates', []))

            path, gain = find_best_cycle(goods, ratios)
            # Optionally trim small floating noise like 8.225000000000016
            gain = float(f"{gain:.12g}")

            results.append({
                "path": path,
                "gain": gain
            })

        return jsonify(results)
    except Exception as e:
        log.exception("Error in /The-Ink-Archive")
        return jsonify({"error": str(e)}), 500
