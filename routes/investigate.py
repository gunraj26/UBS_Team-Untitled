# routes/investigate.py
import logging
from flask import request, jsonify
from . import app

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def extra_channels(edges):
    logger.debug("extra_channels() called with %d edges", len(edges) if edges else 0)
    adj = {}
    def add(u): adj.setdefault(u, [])

    uniq_first_idx = {}
    ordered_unique = []

    for i, e in enumerate(edges or []):
        logger.debug("Processing edge %d: %s", i, e)
        if isinstance(e, dict):
            a, b = str(e.get("spy1", "")), str(e.get("spy2", ""))
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            a, b = str(e[0]), str(e[1])
        else:
            logger.warning("Skipping invalid edge at index %d: %s", i, e)
            continue

        if not a or not b:
            logger.warning("Skipping edge with empty endpoint at index %d: %s", i, e)
            continue

        key = f"{a}|{b}" if a <= b else f"{b}|{a}"
        if key not in uniq_first_idx:
            uniq_first_idx[key] = i
            ordered_unique.append((a, b, key, i))

        add(a); add(b)
        adj[a].append((b, key))
        adj[b].append((a, key))

    logger.debug("Adjacency built with %d nodes", len(adj))

    tin, low, seen = {}, {}, set()
    t = [0]
    bridges = set()

    def dfs(u, parent_key):
        seen.add(u)
        t[0] += 1
        tin[u] = low[u] = t[0]
        for v, key in adj.get(u, []):
            if key == parent_key:
                continue
            if v not in seen:
                dfs(v, key)
                low[u] = min(low[u], low[v])
                if low[v] > tin[u]:
                    bridges.add(key)
            else:
                low[u] = min(low[u], tin[v])

    for node in adj.keys():
        if node not in seen:
            dfs(node, None)

    logger.debug("Found %d bridge edges", len(bridges))

    out = []
    for _, _, key, idx in sorted(ordered_unique, key=lambda x: x[3]):
        if key not in bridges:
            e = edges[uniq_first_idx[key]]
            if isinstance(e, dict):
                out.append({"spy1": e["spy1"], "spy2": e["spy2"]})
            else:
                out.append({"spy1": str(e[0]), "spy2": str(e[1])})

    logger.debug("extra_channels() returning %d extra channels", len(out))
    return out


@app.post("/investigate")
def investigate():
    try:
        data = request.get_json(silent=True)
        logger.info("Raw request JSON: %s", data)

        if data is None:
            logger.error("Request body was not valid JSON or Content-Type missing")
            return jsonify(error="Invalid or missing JSON"), 400

        if isinstance(data, dict):
            logger.debug("Top-level payload is dict")
            networks = data.get("networks", [])
        elif isinstance(data, list):
            logger.debug("Top-level payload is list")
            networks = data
        else:
            logger.error("Unexpected top-level type: %s", type(data))
            return jsonify(error="JSON must be object with 'networks' or a list"), 400

        if not isinstance(networks, list):
            logger.error("'networks' is not a list: %s", type(networks))
            return jsonify(error="'networks' must be a list"), 400

        result = {"networks": []}
        for i, net in enumerate(networks):
            logger.info("Processing network %d: %s", i, net)
            if not isinstance(net, dict):
                logger.error("Network at index %d is not an object: %s", i, net)
                return jsonify(error=f"Network at index {i} must be an object"), 400

            network_id = net.get("networkId", f"network_{i}")
            edges = net.get("network", [])
            logger.debug("Network %s has %d edges", network_id, len(edges) if isinstance(edges, list) else -1)

            if not isinstance(edges, list):
                logger.error("'network' field is not a list in network %s", network_id)
                return jsonify(error=f"'network' for {network_id} must be a list of edges"), 400

            result["networks"].append({
                "networkId": network_id,
                "extraChannels": extra_channels(edges)
            })

        logger.info("Investigation complete, returning %d networks", len(result["networks"]))
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Unhandled exception in /investigate: %s", e)
        return jsonify(error="Internal server error"), 500
