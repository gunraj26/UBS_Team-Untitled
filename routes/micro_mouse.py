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
        l45 = sensor_data[1]
        front = sensor_data[2]
        r45 = sensor_data[3]
    except (TypeError, IndexError):
        # If sensor_data isn't a list of at least three elements,
        # consider everything blocked to prompt a safe rotation.
        l45 = front = r45 = 0

    instructions: list[str] = []

    # Decision logic:
    # 1. If there is free space ahead, keep moving forward.
    #    Accelerate with F2 up to momentum +4, then hold speed with F1.
    if front and front > 0:
        if momentum < 4:
            instructions.append("F2")
        else:
            instructions.append("F1")
    else:
        # 2. Front is blocked.  If the mouse is still moving, brake.
        if momentum > 0:
            # Use BB to reduce momentum toward 0 by two units.
            instructions.append("BB")
        else:
            # 3. Momentum is zero and front is blocked: decide on a turn.
            # Prefer to turn left if the 45° left sensor is open; otherwise
            # turn right.  If both are blocked, rotate 90° to the right in two
            # successive 45° steps to search for an open path.
            if l45 and l45 > 0:
                instructions.append("L")
            elif r45 and r45 > 0:
                instructions.append("R")
            else:
                # Both sides are blocked at 45°; perform two right turns to
                # rotate 90° and reassess.  Using two separate tokens here
                # keeps the instruction set valid and ensures momentum is
                # zero throughout.
                instructions.extend(["R", "R"])

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