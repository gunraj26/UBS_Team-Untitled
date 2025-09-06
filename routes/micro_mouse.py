# import json
# import logging
# from flask import request
# from routes import app
# logger = logging.getLogger(__name__)

"""
Micromouse controller implementing on‑the‑fly maze exploration.

This endpoint provides instructions for a micromouse navigating an
unknown 16×16 maze to the centre goal.  It maintains an internal
representation of the maze, updates it with sensor data at each step,
computes a path to the goal using breadth‑first search (BFS) on
discovered passages, and issues a sequence of tokens to move the mouse
one cell at a time while always keeping momentum at zero.  By using
`F1` twice to translate a full cell and only performing in‑place
rotations at zero momentum, the algorithm avoids illegal moves and
simplifies momentum management.
"""

import json
import logging
from typing import Dict, Tuple, List, Optional

from flask import request
from routes import app

logger = logging.getLogger(__name__)

# Directions encoded as integers: 0=North, 1=East, 2=South, 3=West
DIRECTIONS = [0, 1, 2, 3]
DX = {0: 0, 1: 1, 2: 0, 3: -1}
DY = {0: 1, 1: 0, 2: -1, 3: 0}
DIR_NAMES = {0: "N", 1: "E", 2: "S", 3: "W"}

# Target goal cells (0-indexed coordinates)
GOAL_CELLS = {(7, 7), (7, 8), (8, 7), (8, 8)}


class Session:
    """State for a single game UUID."""
    def __init__(self) -> None:
        # Maze representation: for each cell (x,y) store walls on N,E,S,W.
        # True means wall, False means open, None means unknown.
        self.maze: Dict[Tuple[int, int], Dict[str, Optional[bool]]] = {}
        # Mark cells that have been visited.
        self.visited: Dict[Tuple[int, int], bool] = {}
        # Current cell coordinates and heading
        self.x: int = 0
        self.y: int = 0
        self.heading: int = 0  # 0=north,1=east,2=south,3=west
        # Pending instructions to be sent in the next response.
        self.pending: List[str] = []
        # Track current run number to reset on new runs
        self.run: int = 0

    def reset(self) -> None:
        """Reset state for a new run."""
        self.maze.clear()
        self.visited.clear()
        self.x, self.y = 0, 0
        self.heading = 0
        self.pending.clear()

    def update_maze_with_sensors(self, sensor_data: List[int]) -> None:
        """Update the internal maze map based on sensor readings.

        Sensor data ordering: [L90°, L45°, front, R45°, R90°].  We use
        only the 90° sensors (indices 0 and 4) and the front sensor (index 2)
        to infer walls on the left, front and right sides of the current
        cell.  Positive distances mean open; zero means a wall is adjacent.
        """
        pos = (self.x, self.y)
        # Ensure cell exists in map
        if pos not in self.maze:
            self.maze[pos] = {"N": None, "E": None, "S": None, "W": None}
        # Left wall
        left_dir = (self.heading - 1) % 4
        if sensor_data[0] == 0:
            self.set_wall(pos, left_dir, True)
        else:
            self.set_wall(pos, left_dir, False)
        # Front wall
        if sensor_data[2] == 0:
            self.set_wall(pos, self.heading, True)
        else:
            self.set_wall(pos, self.heading, False)
        # Right wall
        right_dir = (self.heading + 1) % 4
        if sensor_data[4] == 0:
            self.set_wall(pos, right_dir, True)
        else:
            self.set_wall(pos, right_dir, False)
        # Mark current cell as visited
        self.visited[pos] = True

    def set_wall(self, pos: Tuple[int, int], direction: int, is_wall: bool) -> None:
        """Set a wall status for pos in the given direction and update adjacent cell."""
        cell = self.maze.setdefault(pos, {"N": None, "E": None, "S": None, "W": None})
        dir_name = DIR_NAMES[direction]
        if cell[dir_name] is None:
            cell[dir_name] = is_wall
        # Also update neighbour cell if within bounds
        nx, ny = pos[0] + DX[direction], pos[1] + DY[direction]
        if 0 <= nx < 16 and 0 <= ny < 16:
            ncell = self.maze.setdefault((nx, ny), {"N": None, "E": None, "S": None, "W": None})
            opposite = DIR_NAMES[(direction + 2) % 4]
            if ncell[opposite] is None:
                ncell[opposite] = is_wall

    def bfs_to_goal(self) -> Optional[List[Tuple[int, int]]]:
        """Perform BFS on known and unknown cells to find a path to the goal.

        Unknown edges are treated as open to encourage exploration.  Returns
        a list of cell coordinates from the current position to a goal cell,
        inclusive.  If no path is found (should not happen in an open maze),
        returns None.
        """
        from collections import deque
        start = (self.x, self.y)
        if start in GOAL_CELLS:
            return [start]
        queue: deque[Tuple[int, int]] = deque([start])
        parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
        visited: set[Tuple[int, int]] = {start}
        while queue:
            cx, cy = queue.popleft()
            # Expand neighbours in four cardinal directions
            for direction in DIRECTIONS:
                nx, ny = cx + DX[direction], cy + DY[direction]
                if not (0 <= nx < 16 and 0 <= ny < 16):
                    continue
                # Check if there is a wall blocking this edge; unknown treated as open
                cell = self.maze.get((cx, cy), None)
                blocked = False
                if cell:
                    wall_state = cell[DIR_NAMES[direction]]
                    if wall_state is True:
                        blocked = True
                if blocked:
                    continue
                npos = (nx, ny)
                if npos in visited:
                    continue
                visited.add(npos)
                parents[npos] = (cx, cy)
                if npos in GOAL_CELLS:
                    # Build path
                    path = [npos]
                    while path[-1] != start:
                        path.append(parents[path[-1]])
                    path.reverse()
                    return path
                queue.append(npos)
        return None

    def compute_tokens_to_move(self, target: Tuple[int, int]) -> List[str]:
        """Compute a sequence of tokens to rotate and move one cell to target.

        Assumes target is one of the four neighbours of the current cell.
        Always returns a sequence of in‑place rotations (if necessary)
        followed by two 'F1' commands to translate one full cell.  The
        heading and position are updated accordingly.
        """
        tokens: List[str] = []
        # Determine desired direction to target
        dx = target[0] - self.x
        dy = target[1] - self.y
        if dx == 1 and dy == 0:
            desired = 1  # east
        elif dx == -1 and dy == 0:
            desired = 3  # west
        elif dx == 0 and dy == 1:
            desired = 0  # north
        elif dx == 0 and dy == -1:
            desired = 2  # south
        else:
            raise ValueError("Target is not adjacent")
        # Compute rotation needed (0, ±1, ±2 steps of 90°)
        diff = (desired - self.heading) % 4
        if diff == 1:
            tokens.extend(["R", "R"])
            self.heading = (self.heading + 1) % 4
        elif diff == 2:
            # 180° turn: two right turns
            tokens.extend(["R", "R", "R", "R"])
            self.heading = (self.heading + 2) % 4
        elif diff == 3:
            # left 90° turn
            tokens.extend(["L", "L"])
            self.heading = (self.heading - 1) % 4
        # Now move forward one cell with two half‑steps
        tokens.extend(["F1", "F1"])
        # Update position after moving
        self.x, self.y = target
        return tokens


# Maintain sessions per game UUID
_sessions: Dict[str, Session] = {}


@app.route('/micro-mouse', methods=['POST'])
def micro_mouse() -> str:
    data = request.get_json() or {}
    logger.info("data sent for evaluation %s", data)
    game_uuid = data.get("game_uuid")
    # Create or retrieve session
    session = _sessions.setdefault(game_uuid, Session())
    # Reset on new run
    current_run = data.get("run", 0)
    if current_run != session.run:
        session.run = current_run
        session.reset()
    # If goal reached, end the challenge
    if data.get("goal_reached"):
        result = {"instructions": [], "end": True}
        logger.info("My result :%s", result)
        return json.dumps(result)
    # Update map with sensor data when no pending actions from previous step
    if not session.pending and data.get("sensor_data"):
        try:
            sensor_data = [int(v) for v in data.get("sensor_data", [])]
        except Exception:
            sensor_data = [0, 0, 0, 0, 0]
        session.update_maze_with_sensors(sensor_data)
    # If there are pending instructions, return the next batch
    if session.pending:
        instructions = session.pending
        session.pending = []
        result = {"instructions": instructions, "end": False}
        logger.info("My result :%s", result)
        return json.dumps(result)
    # Compute a path to the goal using BFS
    path = session.bfs_to_goal()
    if not path:
        # No path found; return end with failure
        result = {"instructions": [], "end": True}
        logger.info("My result :%s", result)
        return json.dumps(result)
    # If we are already at goal, request end next time
    if len(path) <= 1:
        result = {"instructions": [], "end": True}
        logger.info("My result :%s", result)
        return json.dumps(result)
    # Otherwise, plan tokens to move to the next cell on the path
    next_cell = path[1]
    tokens = session.compute_tokens_to_move(next_cell)
    result = {"instructions": tokens, "end": False}
    logger.info("My result :%s", result)
    return json.dumps(result)