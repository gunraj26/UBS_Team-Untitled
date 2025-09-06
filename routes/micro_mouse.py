import json
import logging

from flask import request

from routes import app

logger = logging.getLogger(__name__)


@app.route('/micro-mouse', methods=['POST'])
def micromouse_controller():
    data = request.get_json()
    logging.info("Micromouse data received: {}".format(data))
    
    # Extract data from request
    game_uuid = data.get("game_uuid")
    sensor_data = data.get("sensor_data", [0, 0, 0, 0, 0])  # Default to no walls detected
    total_time_ms = data.get("total_time_ms", 0)
    goal_reached = data.get("goal_reached", False)
    best_time_ms = data.get("best_time_ms")
    run_time_ms = data.get("run_time_ms", 0)
    run = data.get("run", 0)
    momentum = data.get("momentum", 0)
    
    # Simple micromouse logic - this is a basic implementation
    # You should replace this with your actual maze-solving algorithm
    instructions = []
    end = False
    
    # Basic wall-following algorithm example
    if not goal_reached:
        # Check sensors: [L90°, L45°, Front, R45°, R90°]
        front_clear = sensor_data[2] == 0
        left_clear = sensor_data[0] == 0 or sensor_data[1] == 0
        right_clear = sensor_data[3] == 0 or sensor_data[4] == 0
        
        # Simple decision making
        if momentum == 0:
            # Starting from rest
            if front_clear:
                instructions = ["F2"]  # Accelerate forward
            elif right_clear:
                instructions = ["R"]   # Turn right
            elif left_clear:
                instructions = ["L"]   # Turn left
            else:
                instructions = ["R", "R"]  # Turn around
        else:
            # Already moving
            if front_clear and momentum > 0:
                instructions = ["F1"]  # Continue forward
            elif momentum > 0:
                # Need to slow down and turn
                if right_clear:
                    instructions = ["BB", "R"]
                elif left_clear:
                    instructions = ["BB", "L"]
                else:
                    instructions = ["BB", "R", "R"]
            else:
                # Handle reverse momentum (if any)
                instructions = ["V0"]  # Decelerate from reverse
    else:
        # Goal reached, end the challenge
        end = True
    
    # Timeout check - end if we've used too much time
    if total_time_ms > 290000:  # Leave some buffer before 300,000 ms limit
        end = True
    
    response = {
        "instructions": instructions,
        "end": end
    }
    
    logging.info("Micromouse response: {}".format(response))
    return json.dumps(response)


# Alternative endpoint following the exact pattern from your example
@app.route('/square', methods=['POST'])
def evaluate():
    data = request.get_json()
    logging.info("data sent for evaluation {}".format(data))
    input_value = data.get("input")
    result = input_value * input_value
    logging.info("My result :{}".format(result))
    return json.dumps(result)