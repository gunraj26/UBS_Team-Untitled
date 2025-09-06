import os
import json
import logging
from typing import List, Tuple, Dict, Any
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

import json
import logging

from flask import request

from routes import app


load_dotenv()

# --- Config / logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mst-openai")


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


# --- Kruskal MST ---
def kruskal_mst(num_nodes: int, edges: List[Tuple[int, int, int]]) -> int:
    parent = list(range(num_nodes))
    rank = [0] * num_nodes

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    total, used = 0, 0
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if 0 <= u < num_nodes and 0 <= v < num_nodes and union(u, v):
            total += w
            used += 1
            if used == num_nodes - 1:
                break
    return total

# --- Vision system prompt ---
VISION_SYS_PROMPT = (
    "You are an expert visual graph reader. The image shows an undirected, connected, weighted graph. "
    "Black filled circles are nodes. Colored lines are edges. Near the midpoint of each edge there is an integer weight "
    "drawn (same color as the edge).\n\n"
    "Assign node IDs deterministically as follows: find black circle centers and sort by (y, then x), ascending "
    "(top-to-bottom then left-to-right). Number them 0,1,2,... in that order.\n\n"
    "Return STRICT JSON ONLY with this schema:\n"
    "{\n"
    '  "nodes": <int>,\n'
    '  "edges": [{"u": <int>, "v": <int>, "w": <int>}, ...]\n'
    "}\n"
    "No markdown fences, no prose. Weights are positive integers. Edges are undirected (list each once)."
)

def call_openai_extract(img_b64: str) -> Dict[str, Any]:
    """Call OpenAI vision to get nodes + edges JSON."""
    resp = client.responses.create(
        model="gpt-4.1",  # change to gpt-4o-mini for cheaper/faster runs
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": VISION_SYS_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract the graph and return ONLY the JSON object."},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{img_b64}",
                    },
                ],
            },
        ],
        temperature=0,
    )
    text = resp.output_text.strip()

    # Handle possible code fences
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]

    # Extract JSON
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON: {text[:200]}...")
    data = json.loads(text[start:end+1])

    if "nodes" not in data or "edges" not in data:
        raise ValueError("Missing 'nodes' or 'edges' in model output.")

    edges = [(int(e["u"]), int(e["v"]), int(e["w"])) for e in data["edges"]]
    return {"nodes": int(data["nodes"]), "edges": edges}

@app.route("/mst-calculation", methods=["POST"])
def mst_calculation():
    """
    Input JSON: [{"image": "<base64 png>"}, {"image": "<base64 png>"}]
    Output JSON: [{"value": int}, {"value": int}]
    """
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY is not set"}), 500

    cases = request.get_json(force=True)
    results = []

    for idx, case in enumerate(cases):
        img_b64 = (case or {}).get("image", "") or ""
        if not img_b64:
            logger.warning(f"Case {idx}: missing image")
            results.append({"value": 0})
            continue

        # ✅ Log the raw base64 string
        logger.info(f"Case {idx}: received base64 image (length={len(img_b64)})")
        logger.debug(f"Case {idx}: base64 string = {img_b64}")

        try:
            parsed = call_openai_extract(img_b64)
            num_nodes = parsed["nodes"]
            edges = parsed["edges"]
            logger.info(f"Case {idx}: nodes={num_nodes}, edges={len(edges)}")
            value = kruskal_mst(num_nodes, edges)
        except Exception as e:
            logger.error(f"Case {idx}: vision parse failed: {e}")
            value = 0

        results.append({"value": int(value)})

    return jsonify(results)



