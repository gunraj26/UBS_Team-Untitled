import json
import logging
from collections import deque
from typing import Dict, Tuple, Set, List, Optional

from flask import request
from routes import app

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------
# In-memory game state
# ----------------------------
# Keyed by (challenger_id, game_id)
STATE: Dict[Tuple[str, str], Dict] = {}

DIRS = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}
DIR_ORDER = ["N", "S", "W", "E"]  # deterministic tiebreakers


# ----------------------------
# Helpers
# ----------------------------
def within_bounds(x: int, y: int, L: int) -> bool:
    return 0 <= x < L and 0 <= y < L


def add_cell(known: Dict[Tuple[int, int], str], pos: Tuple[int, int], tag: str):
    """Record knowledge for a cell as 'W' or 'E' and never downgrade walls."""
    if tag not in ("W", "E"):
        return
    cur = known.get(pos)
    if cur == "W":
        return
    if cur == "E" and tag == "E":
        return
    known[pos] = tag


def parse_move_result(mr) -> Optional[Tuple[int, int]]:
    """
    Accept move_result in either format:
    - [x, y]
    - {"x": x, "y": y, ...}
    Return (x, y) or None.
    """
    if isinstance(mr, list) and len(mr) == 2:
        return int(mr[0]), int(mr[1])
    if isinstance(mr, dict) and "x" in mr and "y" in mr:
        return int(mr["x"]), int(mr["y"])
    return None


def update_known_from_scan(center: Tuple[int, int], scan: List[List[str]], L: int,
                           known: Dict[Tuple[int, int], str]):
    cx, cy = center
    if not (isinstance(scan, list) and len(scan) == 5 and all(len(r) == 5 for r in scan)):
        return
    for j in range(5):
        for i in range(5):
            symbol = scan[j][i]
            dx, dy = i - 2, j - 2
            x, y = cx + dx, cy + dy
            if symbol == "X":
                continue  # out of bounds
            if within_bounds(x, y, L):
                if symbol == "W":
                    add_cell(known, (x, y), "W")
                else:  # "_" or "C"
                    add_cell(known, (x, y), "E")


def scan_would_reveal(pos: Tuple[int, int], L: int, known: Dict[Tuple[int, int], str]) -> bool:
    x0, y0 = pos
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x, y = x0 + dx, y0 + dy
            if within_bounds(x, y, L) and (x, y) not in known:
                return True
    return False


def frontier_centers(L: int, known: Dict[Tuple[int, int], str]) -> List[Tuple[int, int]]:
    """Cells where a scan would still reveal unknown tiles; exclude known walls."""
    out = []
    for y in range(L):
        for x in range(L):
            if known.get((x, y)) == "W":
                continue
            # is any neighbor within scan radius unknown?
            needed = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    xx, yy = x + dx, y + dy
                    if within_bounds(xx, yy, L) and (xx, yy) not in known:
                        needed = True
                        break
                if needed:
                    break
            if needed:
                out.append((x, y))
    return out


def bfs_next_step(start: Tuple[int, int], goal: Tuple[int, int], L: int,
                  known: Dict[Tuple[int, int], str]) -> Optional[str]:
    """
    Return the first move (N/S/E/W) on a shortest path from start to goal,
    avoiding cells known as walls. Unknown cells are allowed (we're exploring).
    """
    if start == goal:
        return None

    q = deque([start])
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {start: start}
    while q:
        x, y = q.popleft()
        for d in DIR_ORDER:
            dx, dy = DIRS[d]
            nx, ny = x + dx, y + dy
            np = (nx, ny)
            if not within_bounds(nx, ny, L):
                continue
            if known.get(np) == "W":
                continue
            if np in parent:
                continue
            parent[np] = (x, y)
            if np == goal:
                # reconstruct first step
                cur = np
                while parent[cur] != start:
                    cur = parent[cur]
                fx, fy = cur
                if fx == start[0] and fy == start[1]:
                    # shouldn't happen; safety
                    continue
                # deduce direction from start -> cur
                dx2, dy2 = fx - start[0], fy - start[1]
                for D, (vx, vy) in DIRS.items():
                    if (vx, vy) == (dx2, dy2):
                        return D
            q.append(np)

    # no path found (boxed by walls we already know); try greedy nudge
    sx, sy = start
    tx, ty = goal
    greedy: List[str] = []
    if tx > sx: greedy.append("E")
    if tx < sx: greedy.append("W")
    if ty > sy: greedy.append("S")
    if ty < sy: greedy.append("N")
    # add remaining as fallbacks
    for D in DIR_ORDER:
        if D not in greedy: greedy.append(D)
    for D in greedy:
        dx, dy = DIRS[D]
        nx, ny = sx + dx, sy + dy
        if within_bounds(nx, ny, L) and known.get((nx, ny)) != "W":
            return D
    return None


def decide_action(state: Dict) -> Dict:
    L = state["grid_size"]
    known = state["known"]
    crows = state["crows"]
    planned = state["planned_targets"]

    # If we already know all walls, submit.
    if len([1 for v in known.values() if v == "W"]) >= state["num_walls"]:
        return {"action_type": "submit"}

    # 1) If any crow stands where a scan would reveal new info, scan right away.
    for cid, pos in crows.items():
        if scan_would_reveal(pos, L, known):
            return {"action_type": "scan", "crow_id": cid}

    # 2) Build/refresh frontier set
    fronts = set(frontier_centers(L, known))
    if not fronts:
        # nothing left to reveal; submit what we have
        return {"action_type": "submit"}

    # prune invalid targets
    for cid in list(planned.keys()):
        tgt = planned[cid]
        if tgt not in fronts or known.get(tgt) == "W":
            planned.pop(cid, None)

    # assign targets if missing (greedy nearest)
    taken = set(planned.values())
    for cid, pos in crows.items():
        if cid in planned:
            continue
        best, bestd = None, 10**9
        for f in fronts:
            if f in taken:
                continue
            d = abs(pos[0] - f[0]) + abs(pos[1] - f[1])
            if d < bestd:
                best, bestd = f, d
        if best is not None:
            planned[cid] = best
            taken.add(best)

    # pick crow with shortest distance to its target
    choice = None
    bestd = 10**9
    for cid, pos in crows.items():
        tgt = planned.get(cid)
        if not tgt:
            continue
        d = abs(pos[0] - tgt[0]) + abs(pos[1] - tgt[1])
        if d < bestd:
            bestd = d
            choice = cid

    if not choice:
        # fallback: scan with any
        any_cid = next(iter(crows.keys()))
        return {"action_type": "scan", "crow_id": any_cid}

    # choose path step via BFS
    start = crows[choice]
    goal = planned[choice]
    step = bfs_next_step(start, goal, L, known)
    if step is None:
        # at target: scanning will be handled on next loop top, but do it now
        return {"action_type": "scan", "crow_id": choice}
    return {"action_type": "move", "crow_id": choice, "direction": step}


def walls_as_strings(known: Dict[Tuple[int, int], str]) -> List[str]:
    return [f"{x}-{y}" for (x, y), t in known.items() if t == "W"]


# ----------------------------
# Flask endpoint
# ----------------------------
@app.route('/fog-of-wall', methods=['POST'])
def fog_of_wall():
    payload = request.get_json(force=True, silent=True) or {}
    logger.info("Fog-of-Wall request: %s", payload)

    challenger_id = str(payload.get("challenger_id", ""))
    game_id = str(payload.get("game_id", ""))
    if not challenger_id or not game_id:
        return json.dumps({"error": "challenger_id and game_id are required"}), 400
    key = (challenger_id, game_id)

    # Initialize
    if key not in STATE and payload.get("test_case"):
        tc = payload["test_case"]
        L = int(tc["length_of_grid"])
        num_walls = int(tc["num_of_walls"])
        crows_list = tc.get("crows", [])
        crows = {str(c["id"]): (int(c["x"]), int(c["y"])) for c in crows_list}
        known: Dict[Tuple[int, int], str] = {}
        for pos in crows.values():
            add_cell(known, pos, "E")
        STATE[key] = {
            "grid_size": L,
            "num_walls": num_walls,
            "crows": crows,
            "known": known,
            "planned_targets": {},
            "last_failed_edge": set(),  # {(x,y,dir)} edges we already bumped into
        }

    if key not in STATE:
        return json.dumps({"error": "Unknown game. Send initial test_case first."}), 400

    state = STATE[key]
    L = state["grid_size"]
    known = state["known"]
    crows = state["crows"]

    # ----------------------------
    # Ingest result of previous action
    # ----------------------------
    prev = payload.get("previous_action")
    if prev:
        act = prev.get("your_action")
        cid = str(prev.get("crow_id")) if prev.get("crow_id") is not None else None

        if act == "move" and cid in crows:
            old = crows[cid]
            new_xy = parse_move_result(prev.get("move_result"))
            direction = prev.get("direction")
            if new_xy is not None:
                # bump detection
                if direction in DIRS and new_xy == old:
                    dx, dy = DIRS[direction]
                    wx, wy = old[0] + dx, old[1] + dy
                    if within_bounds(wx, wy, L):
                        add_cell(known, (wx, wy), "W")
                    # mark this edge as failed to avoid repeating
                    state["last_failed_edge"].add((old[0], old[1], direction))
                else:
                    # successful move
                    crows[cid] = new_xy
                    add_cell(known, new_xy, "E")

        elif act == "scan" and cid in crows:
            scan = prev.get("scan_result")
            update_known_from_scan(crows[cid], scan, L, known)

    # ----------------------------
    # Decide next action
    # ----------------------------
    decision = decide_action(state)

    # Avoid repeating the exact same failed edge (loop breaker)
    if decision.get("action_type") == "move":
        cid = decision["crow_id"]
        direction = decision["direction"]
        x, y = state["crows"][cid]
        if (x, y, direction) in state["last_failed_edge"]:
            # try another direction that still goes somewhere legal
            for D in DIR_ORDER:
                if D == direction:
                    continue
                dx, dy = DIRS[D]
                nx, ny = x + dx, y + dy
                if within_bounds(nx, ny, L) and known.get((nx, ny)) != "W":
                    decision["direction"] = D
                    break

    # ----------------------------
    # Emit response or submit
    # ----------------------------
    if decision["action_type"] == "submit":
        submission = walls_as_strings(known)
        resp = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "action_type": "submit",
            "submission": submission,
        }
        logger.info("Submitting %d/%d walls", len(submission), state["num_walls"])
        # tidy cleanup so the next test_case starts fresh
        STATE.pop(key, None)
        return json.dumps(resp)

    if decision["action_type"] == "scan":
        resp = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": decision["crow_id"],
            "action_type": "scan",
        }
        logger.info("Action: scan with crow %s", decision["crow_id"])
        return json.dumps(resp)

    # move
    resp = {
        "challenger_id": challenger_id,
        "game_id": game_id,
        "crow_id": decision["crow_id"],
        "action_type": "move",
        "direction": decision["direction"],
    }
    logger.info("Action: move crow %s %s", decision["crow_id"], decision["direction"])
    return json.dumps(resp)
