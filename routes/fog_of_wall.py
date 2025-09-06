import json
import logging
from collections import defaultdict, deque
from typing import Dict, Tuple, Set, List, Optional

from flask import request
from routes import app  # matches your sample import

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------
# In-memory game state
# ----------------------------
# Keyed by (challenger_id, game_id)
STATE: Dict[Tuple[str, str], Dict] = {}

# Directions and vector deltas
DIRS = {
    "N": (0, -1),
    "S": (0, 1),
    "W": (-1, 0),
    "E": (1, 0),
}
DIR_KEYS = ["N", "S", "E", "W"]  # provide stable, deterministic tie-breaking


# ----------------------------
# Utility helpers
# ----------------------------
def within_bounds(x: int, y: int, L: int) -> bool:
    return 0 <= x < L and 0 <= y < L


def add_cell(known: Dict[Tuple[int, int], str], pos: Tuple[int, int], tag: str):
    """
    Record knowledge for a cell. Tag is 'W' (wall) or 'E' (empty).
    Never downgrade 'W' to 'E'. If already known as 'E', keep it.
    """
    if tag not in ("W", "E"):
        return
    if pos in known:
        if known[pos] == "W":
            return
        if known[pos] == "E":
            return
    known[pos] = tag


def parse_initial_test_case(payload: dict):
    test_case = payload.get("test_case") or {}
    crows_list = test_case.get("crows", [])
    crows = {str(c["id"]): (int(c["x"]), int(c["y"])) for c in crows_list}
    num_walls = int(test_case.get("num_of_walls"))
    L = int(test_case.get("length_of_grid"))
    return crows, num_walls, L


def initialize_game_state(key: Tuple[str, str], payload: dict):
    crows, num_walls, L = parse_initial_test_case(payload)
    # seed known map: mark initial crow cells as empty
    known: Dict[Tuple[int, int], str] = {}
    for _, (x, y) in crows.items():
        add_cell(known, (x, y), "E")

    STATE[key] = {
        "grid_size": L,
        "num_walls": num_walls,
        "known": known,                 # {(x,y): 'W' or 'E'}
        "crows": crows,                 # {"id": (x,y)}
        "planned_targets": {},          # {"id": (tx,ty)} frontier goals
        "last_action": None,            # store last action info if needed
    }


def neighbors4(x: int, y: int) -> List[Tuple[int, int]]:
    return [(x + dx, y + dy) for dx, dy in DIRS.values()]


def chebyshev_dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def manhattan_dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def scan_needed_here(pos: Tuple[int, int], L: int, known: Dict[Tuple[int, int], str]) -> bool:
    """
    Decide whether a scan at 'pos' would reveal anything new:
    returns True if any cell in the 5x5 centered at pos is unknown & in-bounds.
    """
    x0, y0 = pos
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x, y = x0 + dx, y0 + dy
            if within_bounds(x, y, L) and (x, y) not in known:
                return True
    return False


def update_known_from_scan(center: Tuple[int, int], scan: List[List[str]], L: int,
                           known: Dict[Tuple[int, int], str]):
    cx, cy = center
    for j in range(5):
        for i in range(5):
            symbol = scan[j][i]
            dx, dy = i - 2, j - 2
            x, y = cx + dx, cy + dy
            if symbol == "X":
                # out of bounds, ignore
                continue
            if within_bounds(x, y, L):
                if symbol == "W":
                    add_cell(known, (x, y), "W")
                elif symbol in ("_", "C"):
                    add_cell(known, (x, y), "E")


def infer_wall_from_bump(prev_pos: Tuple[int, int], direction: str, new_pos: Tuple[int, int],
                         L: int, known: Dict[Tuple[int, int], str]):
    """
    If a move results in the same position, we hit a wall in the intended direction.
    Mark the blocked neighbor (if in bounds) as 'W'.
    """
    if prev_pos == new_pos:
        dx, dy = DIRS[direction]
        wx, wy = prev_pos[0] + dx, prev_pos[1] + dy
        if within_bounds(wx, wy, L):
            add_cell(known, (wx, wy), "W")


def current_walls(known: Dict[Tuple[int, int], str]) -> Set[Tuple[int, int]]:
    return {pos for pos, tag in known.items() if tag == "W"}


def current_walls_as_strings(known: Dict[Tuple[int, int], str]) -> List[str]:
    return [f"{x}-{y}" for (x, y), tag in known.items() if tag == "W"]


def frontier_centers(L: int, known: Dict[Tuple[int, int], str]) -> List[Tuple[int, int]]:
    """
    A frontier center is a cell where performing a scan would reveal at least one unknown,
    and the center cell itself is not known to be a wall.
    We consider all in-bounds cells; we prefer ones that are known empty, but allow unknown centers too.
    """
    candidates = []
    for y in range(L):
        for x in range(L):
            if known.get((x, y)) == "W":
                continue
            # If any unknown within 2 cells, it's a frontier
            needs = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    xx, yy = x + dx, y + dy
                    if within_bounds(xx, yy, L) and (xx, yy) not in known:
                        needs = True
                        break
                if needs:
                    break
            if needs:
                candidates.append((x, y))
    return candidates


def choose_crow_and_action(state: Dict) -> Dict:
    """
    Decide the next action:
    - If any crow stands where a scan would reveal new info, scan with that crow.
    - Else, move the crow that is closest to any frontier center toward its assigned/nearest frontier.
    - If there are no frontier centers (everything known or all walls found), fallback: submit if possible, else no-op scan.
    """
    L = state["grid_size"]
    known = state["known"]
    crows = state["crows"]
    planned = state["planned_targets"]

    # If we already know all walls, submit.
    if len(current_walls(known)) >= state["num_walls"]:
        return {"action_type": "submit"}

    # 1) Scan if standing over a useful scan spot
    for cid, pos in crows.items():
        if scan_needed_here(pos, L, known):
            return {"action_type": "scan", "crow_id": cid}

    # 2) Build global frontier list
    frontiers = frontier_centers(L, known)
    if not frontiers:
        # Nothing left to reveal — either we already know all walls (handled above)
        # or everything is known empty — submit whatever walls we have.
        return {"action_type": "submit"}

    # 3) Assign or refresh planned targets if the targets no longer make sense
    #    (e.g., center became wall somehow).
    invalid = []
    for cid, tgt in planned.items():
        if tgt not in frontiers or known.get(tgt) == "W":
            invalid.append(cid)
    for cid in invalid:
        planned.pop(cid, None)

    # 4) If some crows lack a target, greedily assign nearest unique frontier
    unassigned_crows = [cid for cid in crows.keys() if cid not in planned]
    # Simple greedy: for each unassigned crow, pick nearest frontier not yet taken
    taken: Set[Tuple[int, int]] = set(planned.values())
    for cid in unassigned_crows:
        pos = crows[cid]
        best = None
        best_d = 10**9
        for f in frontiers:
            if f in taken:
                continue
            d = manhattan_dist(pos, f)
            if d < best_d:
                best_d = d
                best = f
        if best is not None:
            planned[cid] = best
            taken.add(best)

    # 5) Pick the crow that can get to its target fastest
    best_crow = None
    best_d = 10**9
    best_target = None
    for cid, pos in crows.items():
        tgt = planned.get(cid)
        if not tgt:
            continue
        d = manhattan_dist(pos, tgt)
        if d < best_d:
            best_d = d
            best_crow = cid
            best_target = tgt

    # If nobody has a target (very unlikely), scan with any crow as a fallback
    if best_crow is None:
        any_cid = next(iter(crows.keys()))
        return {"action_type": "scan", "crow_id": any_cid}

    # 6) Move best_crow one step toward best_target, preferring moves that reduce Manhattan distance
    cx, cy = crows[best_crow]
    tx, ty = best_target
    dx = tx - cx
    dy = ty - cy

    candidates = []
    if dx != 0:
        candidates.append("E" if dx > 0 else "W")
    if dy != 0:
        candidates.append("S" if dy > 0 else "N")

    # Stable tie-breaking / also try orthogonal alternatives if primary choices are blocked as known walls.
    tried: List[str] = []
    for d in candidates:
        tried.append(d)
    # Add remaining directions as backup (may still bump to learn walls)
    for d in DIR_KEYS:
        if d not in tried:
            tried.append(d)

    for direction in tried:
        vx, vy = DIRS[direction]
        nx, ny = cx + vx, cy + vy
        if not within_bounds(nx, ny, L):
            continue
        if state["known"].get((nx, ny)) == "W":
            continue  # avoid known wall
        # attempt this direction
        return {"action_type": "move", "crow_id": best_crow, "direction": direction}

    # If all directions are either OOB or known walls, attempt a scan as a last resort
    return {"action_type": "scan", "crow_id": best_crow}


def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


# ----------------------------
# Flask endpoint
# ----------------------------
@app.route('/fog-of-wall', methods=['POST'])
def fog_of_wall():
    """
    Core endpoint implementing the Fog of Wall strategy.

    Request: JSON as specified (initial test case, or follow-up with previous_action).
    Response: JSON with one of: scan / move / submit.

    State is kept per (challenger_id, game_id).
    """
    payload = request.get_json(force=True, silent=True) or {}
    logger.info("Fog-of-Wall request: %s", payload)

    challenger_id = str(payload.get("challenger_id", ""))
    game_id = str(payload.get("game_id", ""))
    if not challenger_id or not game_id:
        return json.dumps({"error": "challenger_id and game_id are required"}), 400

    key = (challenger_id, game_id)

    # Initialize per-game state on first message of a test case
    if key not in STATE and payload.get("test_case"):
        initialize_game_state(key, payload)

    # If somehow still missing (bad sequence), fail gracefully
    if key not in STATE:
        return json.dumps({"error": "Unknown game. Send initial test_case first."}), 400

    state = STATE[key]
    known = state["known"]
    crows = state["crows"]
    L = state["grid_size"]

    # --------------------------------
    # Ingest the result of our previous action (if any)
    # --------------------------------
    prev = payload.get("previous_action")
    if prev:
        action = prev.get("your_action")
        cid = str(prev.get("crow_id")) if prev.get("crow_id") is not None else None

        if action == "move":
            # Update the crow's position and infer wall if we bumped
            move_result = prev.get("move_result")
            direction = prev.get("direction")
            if cid in crows and isinstance(move_result, list) and len(move_result) == 2 and direction in DIRS:
                old_pos = crows[cid]
                new_pos = (int(move_result[0]), int(move_result[1]))
                # Learn from bump
                infer_wall_from_bump(old_pos, direction, new_pos, L, known)
                # Update crow position and mark as empty
                crows[cid] = new_pos
                add_cell(known, new_pos, "E")

        elif action == "scan":
            scan_result = prev.get("scan_result")
            if cid in crows and isinstance(scan_result, list) and len(scan_result) == 5:
                center = crows[cid]
                update_known_from_scan(center, scan_result, L, known)

    # --------------------------------
    # Decide next action
    # --------------------------------
    decision = choose_crow_and_action(state)

    # Submit if all walls found or no more frontiers
    if decision["action_type"] == "submit":
        walls_list = current_walls_as_strings(known)
        response = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "action_type": "submit",
            "submission": walls_list,
        }
        logger.info("Submitting %d walls (known target=%d)", len(walls_list), state["num_walls"])
        # Clean up state for the next test case (optional but tidy)
        STATE.pop(key, None)
        return json.dumps(response)

    # Scan action
    if decision["action_type"] == "scan":
        crow_id = decision["crow_id"]
        response = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": crow_id,
            "action_type": "scan",
        }
        logger.info("Action: scan with crow %s", crow_id)
        return json.dumps(response)

    # Move action
    if decision["action_type"] == "move":
        crow_id = decision["crow_id"]
        direction = decision["direction"]
        response = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": crow_id,
            "action_type": "move",
            "direction": direction,
        }
        logger.info("Action: move crow %s %s", crow_id, direction)
        return json.dumps(response)

    # Fallback (shouldn't happen)
    logger.warning("No valid decision; defaulting to scan with first crow.")
    any_cid = next(iter(crows.keys()))
    return json.dumps({
        "challenger_id": challenger_id,
        "game_id": game_id,
        "crow_id": any_cid,
        "action_type": "scan",
    })
