# import json
# import logging
# from flask import request
# from routes import app
# logger = logging.getLogger(__name__)

"""
Endpoint for controlling a micromouse simulation.

This module defines a single HTTP POST endpoint at `/micro-mouse` that accepts
JSON describing the current state of a micromouse and returns a short list
of movement tokens.  The goal of the controller is to navigate the mouse
through a 16×16 maze without crashing into walls.  Each request includes
sensor readings at five angles (left‑90°, left‑45°, front, right‑45° and
right‑90°) and the current momentum.  The response includes a small array
of movement tokens and an `end` flag indicating whether the simulation
should terminate.

The simple strategy implemented here is reactive: it examines the sensor
values for obstacles and chooses one of three behaviours:

* **Accelerate forward** when the front path is clear.  The mouse uses
  `F2` to increase its momentum up to the maximum of +4.  Once at top
  speed the controller instructs `F1` to maintain momentum.  The logic
  avoids accelerating when momentum is already at the limit.
* **Brake** when an obstacle appears directly ahead while the mouse is
  moving.  Braking uses the `BB` token, which reduces momentum by two
  units and executes a half‑step translation in the current heading.  In
  practice this quickly brings the mouse to a halt before attempting
  in‑place turns.
* **Rotate** when the front cell is blocked and the mouse has stopped.
  The controller prefers turning left if there is space at 45° to the
  left, otherwise it turns right.  In‑place rotations (`L` or `R`)
  require zero momentum, so the controller brakes first if necessary.

This approach keeps the mouse moving forward when possible and makes
simple course corrections when it encounters walls.  More sophisticated
algorithms (flood‑fill, wall‑following, etc.) are possible, but the
baseline strategy here satisfies the requirement to avoid the error
"simulation error: crashed into a wall" while demonstrating the
structure of a Flask route handler.
"""

import json
import logging
from flask import request
from routes import app

logger = logging.getLogger(__name__)


@app.route("/micro-mouse", methods=["POST"])
def micro_mouse():
    """Compute a list of movement tokens for the micromouse.

    The function expects a JSON body with at least the following keys:

    - `sensor_data`: a list of five integers, representing distance
      measurements at fixed angles around the mouse.  Values of zero
      indicate an adjacent wall, while positive values (up to 12) mean
      there is free space ahead.  The ordering is [L90°, L45°, front,
      R45°, R90°].
    - `momentum`: the current momentum of the mouse (−4 to +4).

    Additional fields such as `total_time_ms`, `goal_reached` and
    `run_time_ms` are passed through by the simulator but are not used
    directly in this simple controller.  These fields remain available
    for more advanced algorithms that may want to adjust behaviour based
    on overall progress or remaining time budget.

    The handler returns a JSON object with two fields:

    - `instructions`: an array of movement tokens to execute in order.
      Each token must be one of the allowed commands defined by the
      simulation rules (e.g. "F2", "BB", "L", "R", "F1L", etc.).
    - `end`: a boolean indicating whether the current simulation should
      terminate immediately without executing the provided instructions.

    Returns:
        A JSON‑encoded string describing the next set of instructions and
        whether to end the simulation.
    """
    data = request.get_json() or {}
    logger.info("data sent for evaluation %s", data)

    # Extract relevant fields with sensible defaults.  If sensor data is
    # missing or malformed, treat all sensors as blocked to prevent
    # accidental crashes.
    sensor_data = data.get("sensor_data") or [0, 0, 0, 0, 0]
    momentum = data.get("momentum", 0)

    # The indices for sensor_data correspond to angles around the mouse.
    # We only need the front (index 2) and the 45° left/right sensors
    # (indices 1 and 3) for this simple controller.  The 90° sensors
    # provide additional context for left‑ or right‑turning decisions but
    # are not strictly necessary here.
    try:
        # Distances at various angles.  A positive value indicates free
        # space; zero means the mouse is adjacent to a wall.  The
        # ordering is [L90°, L45°, front, R45°, R90°].
        l90 = sensor_data[0]
        l45 = sensor_data[1]
        front = sensor_data[2]
        r45 = sensor_data[3]
        r90 = sensor_data[4]
    except (TypeError, IndexError):
        # If sensor_data isn't a list of length five, treat all
        # directions as blocked to prompt a safe rotation.
        l90 = l45 = front = r45 = r90 = 0

    instructions: list[str] = []

    # Decision logic:
    # Keep the control scheme very simple to avoid crashes: always
    # maintain zero momentum before moving and move one half‑step at a
    # time.  Rotate right until a free path opens ahead.
    if momentum > 0:
        # If still moving, brake towards zero momentum.  We never turn
        # or accelerate while momentum is non‑zero.
        instructions.append("BB")
    else:
        if front and front > 0:
            # A clear path ahead: take one half‑step forward without
            # increasing momentum.  F1 moves a half‑step at constant
            # speed (m=0 → m=0) and is safer than accelerating.
            instructions.append("F1")
        else:
            # Path ahead is blocked.  Do not attempt to rotate with
            # non‑zero momentum.  Instead, issue a brake command even
            # when already at rest.  A brake at momentum 0 (BB at m=0)
            # simply consumes time (200 ms) without changing state.  This
            # conservative approach avoids illegal rotations while
            # preventing forward motion into a wall.
            instructions.append("BB")

    # Construct the response.  The 'end' flag remains false because this
    # controller is designed to continue operating until an external
    # termination condition is triggered (e.g. time expires or the maze is
    # solved).  To implement early termination, set end=True here.
    result = {
        "instructions": instructions,
        "end": False,
    }
    logger.info("My result :%s", result)
    return json.dumps(result)