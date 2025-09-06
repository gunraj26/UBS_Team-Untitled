# routes/investigate.py
from flask import request, jsonify
from collections import OrderedDict
from . import app

def extra_channels(edges):
    """
    Return list of {spy1, spy2} edges that are NOT bridges (i.e., lie on a cycle).
    Keeps the first-seen orientation from `edges`.
    """
    # Build graph
    adj = {}
    def add(u): adj.setdefault(u, [])

    # Track unique undirected edges by normalized key "a|b" (a <= b)
    unique_key_first_idx = {}
    ordered_unique = []

    for i, e in enumerate(edges or []):
        a = str(e.get("spy1", ""))
        b = str(e.get("spy2", ""))
        if not a or not b:
            continue
        key = f"{a}|{b}" if a <= b else f"{b}|{a}"

        if key not in unique_key_first_idx:
            unique_key_first_idx[key] = i
            ordered_unique.append((a, b, key, i))

        add(a); add(b)
        adj[a].append((b, key))
        adj[b].append((a, key))

    # Tarjan bridges
    tin, low, visited = {}, {}, set()
    timer = [0]
    bridge_keys = set()

    def dfs(u, parent_key):
        visited.add(u)
        timer[0] += 1
        tin[u] = low[u] = timer[0]
        for v, key in adj.get(u, []):
            if key == parent_key:
                continue
            if v not in visited:
                dfs(v, key)
                low[u] = min(low[u], low[v])
                if low[v] > tin[u]:
                    bridge_keys.add(key)
            else:
                low[u] = min(low[u], tin[v])

    for node in list(adj.keys()):
        if node not in visited:
            dfs(node, None)

    # Anything not a bridge is an extra channel (part of a cycle)
    out = []
    for a, b, key, _ in sorted(ordered_unique, key=lambda x: x[3]):
        if key not in bridge_keys:
            original = edges[unique_key_first_idx[key]]
            out.append({"spy1": original["spy1"], "spy2": original["spy2"]})
    return out

@app.route('/investigate', methods=['POST'])
def investigate():
    payload = request.get_json(silent=True) or {}
    networks = payload.get("networks", []) or []
    result = {"networks": []}

    for net in networks:
        network_id = net.get("networkId")
        edges = net.get("network", [])
        extras = extra_channels(edges)

        # enforce field order: networkId → extraChannels
        ordered = OrderedDict()
        ordered["networkId"] = network_id
        ordered["extraChannels"] = extras

        result["networks"].append(ordered)

    return jsonify(result)





