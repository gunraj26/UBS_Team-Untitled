"""
Flask endpoint implementing an efficient strategy for the Fog‑of‑Wall challenge.

This solution uses a *preplanned scanning pattern* to cover the entire grid with
a small number of scans. The maze is a square grid of size ``length_of_grid``.
Each scan reveals the 5×5 area centred on a crow. By placing scans on a
lattice with spacing 5 cells (centres at positions 2, 7, 12, …), we can cover
every cell in the grid in a predictable number of scans.  Crows are assigned
to these scanning centres and move using a breadth‑first search (BFS) path
finder that avoids known walls but allows unknown cells.  Once a crow reaches
a scanning centre it performs a scan (only if that scan would reveal new
information), otherwise it moves toward its next assigned centre.  When all
walls have been discovered (or no remaining scan centres need scanning), the
agent submits the discovered wall positions.

The endpoint expects POST requests in the format specified by the problem
statement.  State is kept per ``(challenger_id, game_id)`` so multiple test
cases can run concurrently.  The code defensively handles malformed inputs,
different shapes of ``move_result``, and avoids loops when bumping into walls.

"""

import json
import logging
from collections import deque
from typing import Dict, Tuple, List, Set, Optional

from flask import request
from routes import app  # Assumes there is an existing Flask app instance

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Global state keyed by (challenger_id, game_id)
STATE: Dict[Tuple[str, str], Dict] = {}

# Direction vectors and a stable order for iteration
DIRS = {
    "N": (0, -1),
    "S": (0, 1),
    "W": (-1, 0),
    "E": (1, 0),
}
DIR_ORDER = ["N", "S", "W", "E"]  # used for deterministic tie‑breaking


def within_bounds(x: int, y: int, L: int) -> bool:
    """Return True if (x, y) lies within a 0≤x,y< L grid."""
    return 0 <= x < L and 0 <= y < L


def add_cell(known: Dict[Tuple[int, int], str], pos: Tuple[int, int], tag: str) -> None:
    """Record knowledge for a cell as 'W' (wall) or 'E' (empty). Don't downgrade walls."""
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
    Accept a ``move_result`` either as a two‑element list [x, y] or as a dict
    with 'x' and 'y' keys.  Returns a tuple (x, y) or ``None`` if parsing
    fails.
    """
    if isinstance(mr, list) and len(mr) == 2:
        try:
            return int(mr[0]), int(mr[1])
        except (TypeError, ValueError):
            return None
    if isinstance(mr, dict) and "x" in mr and "y" in mr:
        try:
            return int(mr["x"]), int(mr["y"])
        except (TypeError, ValueError):
            return None
    return None


def update_known_from_scan(center: Tuple[int, int], scan: List[List[str]], L: int,
                           known: Dict[Tuple[int, int], str]) -> None:
    """
    Update the ``known`` map based on a 5×5 scan result centred at ``center``.
    The scan is a list of five lists of five characters.  Cells labelled 'W'
    become walls; '_' and 'C' (the scanning crow itself) mark empty cells; 'X'
    represents out‑of‑bounds and is ignored.  Only in‑bounds coordinates are
    recorded.
    """
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
                else:  # '_' or 'C'
                    add_cell(known, (x, y), "E")


def scan_would_reveal(center: Tuple[int, int], L: int,
                      known: Dict[Tuple[int, int], str]) -> bool:
    """
    Return True if a scan at ``center`` would reveal at least one unknown
    in‑bounds cell.  It checks the 5×5 neighbourhood around ``center``.
    """
    x0, y0 = center
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x, y = x0 + dx, y0 + dy
            if within_bounds(x, y, L) and (x, y) not in known:
                return True
    return False


def compute_scan_positions(L: int) -> List[int]:
    """
    Compute the list of axis positions (along x or y) for scanning centres.
    Starting at coordinate 2 and stepping by 5 ensures each 5×5 scan covers
    adjacent ranges without gaps.  For tiny grids (L <= 3) this returns [0]
    so there is at least one centre.
    """
    positions: List[int] = []
    pos = 2
    # Place centres every 5 cells (2, 7, 12, …) as long as pos < L
    while pos < L:
        positions.append(pos)
        pos += 5
    # Fallback for small grids: ensure there's at least one centre within [0, L‑1]
    if not positions:
        positions.append(min(2, L - 1))
    return positions


def bfs_next_step(start: Tuple[int, int], goal: Tuple[int, int], L: int,
                  known: Dict[Tuple[int, int], str]) -> Optional[str]:
    """
    Compute the first step along a shortest path from ``start`` to ``goal``
    avoiding cells known to be walls.  Unknown cells are treated as passable.
    Returns a direction character ('N', 'S', 'E', 'W') or ``None`` if
    already at the goal or no path exists.
    """
    if start == goal:
        return None
    # BFS queue holds positions; ``parent`` maps child -> parent to reconstruct path
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
                continue  # cannot go through known walls
            if np in parent:
                continue  # already visited
            parent[np] = (x, y)
            if np == goal:
                # Reconstruct the path back to start to find the first step
                cur = np
                while parent[cur] != start:
                    cur = parent[cur]
                fx, fy = cur
                dx2, dy2 = fx - start[0], fy - start[1]
                for direction, (vx, vy) in DIRS.items():
                    if (vx, vy) == (dx2, dy2):
                        return direction
                # fallthrough if direction not found (shouldn't happen)
            q.append(np)
    # No path found (perhaps boxed in by walls); try a greedy nudge
    sx, sy = start
    tx, ty = goal
    greedy: List[str] = []
    if tx > sx:
        greedy.append("E")
    if tx < sx:
        greedy.append("W")
    if ty > sy:
        greedy.append("S")
    if ty < sy:
        greedy.append("N")
    # Append the other directions as fallbacks to avoid being stuck
    for D in DIR_ORDER:
        if D not in greedy:
            greedy.append(D)
    for D in greedy:
        dx, dy = DIRS[D]
        nx, ny = sx + dx, sy + dy
        if within_bounds(nx, ny, L) and known.get((nx, ny)) != "W":
            return D
    return None


def decide_action(state: Dict) -> Dict:
    """
    Decide the next action based on the current state.  The strategy is to
    follow a precomputed list of scanning centres to cover the entire grid.
    Each crow is assigned to the nearest unvisited scanning centre.  When a
    crow arrives at a centre and scanning there would reveal something, it
    scans.  Otherwise, the crow moves one step along a path to its target.
    Once all walls have been discovered or all centres are complete, the
    strategy submits the wall positions.
    """
    L = state["grid_size"]
    known = state["known"]
    crows = state["crows"]
    num_walls = state["num_walls"]
    centres_to_scan: Set[Tuple[int, int]] = state["scan_centres_to_scan"]
    centres_done: Set[Tuple[int, int]] = state["scan_centres_done"]
    targets = state["targets"]
    last_failed_edge: Set[Tuple[int, int, str]] = state["last_failed_edge"]

    # If we already discovered all walls, submit immediately
    if len([1 for v in known.values() if v == "W"]) >= num_walls:
        return {"action_type": "submit"}

    # Remove centres that no longer need scanning (no unknown cells in 5×5 area)
    # We iterate over a copy to avoid modifying set during iteration
    for centre in list(centres_to_scan):
        if not scan_would_reveal(centre, L, known):
            centres_to_scan.remove(centre)
            centres_done.add(centre)

    # If all centres are processed and we haven't found all walls, fall back to
    # exploring unknown cells with local scans (same as previous strategy)
    if not centres_to_scan:
        # If there are still unknown cells, use the original frontier‑based scanning
        # strategy to clear them; otherwise, submit whatever walls we have.
        # Build frontier cells: unknown cells
        frontiers: Set[Tuple[int, int]] = set()
        for y in range(L):
            for x in range(L):
                if (x, y) not in known and known.get((x, y)) != "W":
                    frontiers.add((x, y))
        # If no unknown cells remain, submit
        if not frontiers:
            return {"action_type": "submit"}
        # Otherwise behave like previous algorithm: scan if standing on a cell
        # whose scan would reveal new info; else move to nearest frontier
        # 1) Scan if any crow can reveal new info at current location
        for cid, pos in crows.items():
            if scan_would_reveal(pos, L, known):
                return {"action_type": "scan", "crow_id": cid}
        # 2) Assign nearest frontier to each crow and move toward it
        # Simple greedy assignment per crow
        best_cid = None
        best_goal = None
        best_dist = 10**9
        for cid, pos in crows.items():
            # find nearest frontier
            nearest = None
            nearest_d = 10**9
            for f in frontiers:
                d = abs(pos[0] - f[0]) + abs(pos[1] - f[1])
                if d < nearest_d:
                    nearest = f
                    nearest_d = d
            if nearest is not None and nearest_d < best_dist:
                best_cid = cid
                best_goal = nearest
                best_dist = nearest_d
        # If we found a target, move toward it
        if best_cid is not None:
            step = bfs_next_step(crows[best_cid], best_goal, L, known)
            # If BFS fails, just scan to learn more
            if step is None:
                return {"action_type": "scan", "crow_id": best_cid}
            return {"action_type": "move", "crow_id": best_cid, "direction": step}
        # Fallback: scan with any crow
        any_cid = next(iter(crows.keys()))
        return {"action_type": "scan", "crow_id": any_cid}

    # Primary scanning plan: check if any crow is standing on a centre needing scanning
    for cid, pos in crows.items():
        if pos in centres_to_scan:
            return {"action_type": "scan", "crow_id": cid}

    # Assign targets to crows if they don't have one or their target is complete
    assigned: Set[Tuple[int, int]] = set(targets.values())
    for cid, pos in crows.items():
        tgt = targets.get(cid)
        if tgt not in centres_to_scan:
            # find nearest centre not taken
            nearest = None
            nearest_d = 10**9
            for centre in centres_to_scan:
                if centre in assigned:
                    continue
                d = abs(pos[0] - centre[0]) + abs(pos[1] - centre[1])
                if d < nearest_d:
                    nearest = centre
                    nearest_d = d
            if nearest is not None:
                targets[cid] = nearest
                assigned.add(nearest)

    # Choose the crow that can make progress to its target fastest (smallest
    # Manhattan distance).  Compute BFS step and issue a move.
    selected_cid = None
    selected_goal = None
    best_dist = 10**9
    for cid, pos in crows.items():
        tgt = targets.get(cid)
        if tgt is None:
            continue
        d = abs(pos[0] - tgt[0]) + abs(pos[1] - tgt[1])
        if d < best_dist:
            best_dist = d
            selected_cid = cid
            selected_goal = tgt

    # There should always be a selected crow here because centres_to_scan is not empty
    if selected_cid is None or selected_goal is None:
        # Fallback to scanning with any crow
        any_cid = next(iter(crows.keys()))
        return {"action_type": "scan", "crow_id": any_cid}

    # Determine next step via BFS
    step = bfs_next_step(crows[selected_cid], selected_goal, L, known)
    # If no step (already there or path blocked), we scan or pick another direction
    if step is None:
        # At the goal but not scanning centre? or path blocked; just scan to learn more
        return {"action_type": "scan", "crow_id": selected_cid}

    # Avoid repeating known failed edge (loop protection)
    cx, cy = crows[selected_cid]
    if (cx, cy, step) in last_failed_edge:
        # Try alternate directions
        for alt in DIR_ORDER:
            if alt == step:
                continue
            dx, dy = DIRS[alt]
            nx, ny = cx + dx, cy + dy
            if within_bounds(nx, ny, L) and known.get((nx, ny)) != "W":
                step = alt
                break
    return {"action_type": "move", "crow_id": selected_cid, "direction": step}


def walls_as_strings(known: Dict[Tuple[int, int], str]) -> List[str]:
    """Convert positions of known walls into the required 'x-y' string format."""
    return [f"{x}-{y}" for (x, y), tag in known.items() if tag == "W"]


@app.route('/fog-of-wall', methods=['POST'])
def fog_of_wall():  # pragma: no cover - entry point for server
    """
    Core endpoint for the Fog‑of‑Wall game.  It manages per‑game state,
    updates knowledge from the previous action's result, and selects the next
    action using the scanning pattern strategy.  Returns JSON with one of
    three actions: ``move``, ``scan``, or ``submit``.
    """
    payload = request.get_json(force=True, silent=True) or {}
    logger.info("Fog‑of‑Wall request: %s", payload)

    challenger_id = str(payload.get("challenger_id", ""))
    game_id = str(payload.get("game_id", ""))
    if not challenger_id or not game_id:
        return json.dumps({"error": "challenger_id and game_id are required"}), 400
    key = (challenger_id, game_id)

    # Initialise a new game when ``test_case`` is present
    if key not in STATE and payload.get("test_case"):
        tc = payload["test_case"] or {}
        L = int(tc.get("length_of_grid", 0))
        num_walls = int(tc.get("num_of_walls", 0))
        crows_list = tc.get("crows", [])
        crows: Dict[str, Tuple[int, int]] = {str(c["id"]): (int(c["x"]), int(c["y"]))
                                             for c in crows_list}
        # Known map: mark initial crow positions as empty
        known: Dict[Tuple[int, int], str] = {}
        for pos in crows.values():
            add_cell(known, pos, "E")
        # Compute scanning centres on a lattice covering the grid
        axis_positions = compute_scan_positions(L)
        scan_centres: Set[Tuple[int, int]] = set((x, y) for x in axis_positions for y in axis_positions)
        STATE[key] = {
            "grid_size": L,
            "num_walls": num_walls,
            "crows": crows,
            "known": known,
            "scan_centres_to_scan": set(scan_centres),
            "scan_centres_done": set(),
            "targets": {},  # crow_id -> centre
            "last_failed_edge": set(),  # track bump edges to avoid loops
        }

    # If the game has not been initialised properly, reject
    if key not in STATE:
        return json.dumps({"error": "Unknown game. Send initial test_case first."}), 400

    state = STATE[key]
    L = state["grid_size"]
    known = state["known"]
    crows = state["crows"]
    # -------------------------------------------------------------------------
    # Ingest previous action result
    prev = payload.get("previous_action") or {}
    if prev:
        act = prev.get("your_action")
        cid = str(prev.get("crow_id")) if prev.get("crow_id") is not None else None
        # Handle move action
        if act == "move" and cid in crows:
            direction = prev.get("direction")
            mr = parse_move_result(prev.get("move_result"))
            old_pos = crows[cid]
            if mr is not None:
                new_pos = mr
                # If we didn't move, we bumped into a wall; record it
                if new_pos == old_pos:
                    if direction in DIRS:
                        dx, dy = DIRS[direction]
                        wx, wy = old_pos[0] + dx, old_pos[1] + dy
                        if within_bounds(wx, wy, L):
                            add_cell(known, (wx, wy), "W")
                        # Record failed edge to avoid repeating
                        state["last_failed_edge"].add((old_pos[0], old_pos[1], direction))
                else:
                    # Successful move
                    crows[cid] = new_pos
                    add_cell(known, new_pos, "E")
        # Handle scan action
        elif act == "scan" and cid in crows:
            scan = prev.get("scan_result")
            update_known_from_scan(crows[cid], scan, L, known)
            # If the crow was standing on a planned centre, mark it done
            pos = crows[cid]
            if pos in state["scan_centres_to_scan"]:
                state["scan_centres_to_scan"].remove(pos)
                state["scan_centres_done"].add(pos)

    # -------------------------------------------------------------------------
    # Decide next action based on updated state
    decision = decide_action(state)

    # Submit action
    if decision.get("action_type") == "submit":
        submission = walls_as_strings(known)
        resp = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "action_type": "submit",
            "submission": submission,
        }
        logger.info("Submitting %d/%d walls", len(submission), state["num_walls"])
        # Remove state for this game to free memory
        STATE.pop(key, None)
        return json.dumps(resp)

    # Scan action
    if decision.get("action_type") == "scan":
        resp = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": decision["crow_id"],
            "action_type": "scan",
        }
        logger.info("Action: scan with crow %s", decision["crow_id"])
        return json.dumps(resp)

    # Move action
    if decision.get("action_type") == "move":
        resp = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": decision["crow_id"],
            "action_type": "move",
            "direction": decision["direction"],
        }
        logger.info("Action: move crow %s %s", decision["crow_id"], decision["direction"])
        return json.dumps(resp)

    # Fallback (shouldn't happen) – default to scanning with the first crow
    any_cid = next(iter(crows.keys()))
    return json.dumps({
        "challenger_id": challenger_id,
        "game_id": game_id,
        "crow_id": any_cid,
        "action_type": "scan",
    })