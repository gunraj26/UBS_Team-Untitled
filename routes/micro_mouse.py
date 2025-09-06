from flask import request, jsonify
from typing import List, Optional, Dict, Any
import uuid

from routes import app

# Valid movement tokens as defined in the specification
VALID_TOKENS = {
    # Longitudinal
    'F0', 'F1', 'F2', 'BB', 'V0', 'V1', 'V2',
    # In-place rotations
    'L', 'R',
    # Moving rotations (translation + rotation)
    'F0L', 'F0R', 'F1L', 'F1R', 'F2L', 'F2R',
    'V0L', 'V0R', 'V1L', 'V1R', 'V2L', 'V2R',
    'BBL', 'BBR',
    # Corner turns (many combinations possible)
    # Format: (F0|F1|F2|V0|V1|V2)(L|R)(T|W)[(L|R)]
    # Examples: F1LT, F2RT, F1LW, V1RTL, etc.
}

# Add corner turn tokens programmatically
for accel in ['F0', 'F1', 'F2', 'V0', 'V1', 'V2']:
    for direction in ['L', 'R']:
        for radius in ['T', 'W']:  # Tight or Wide
            VALID_TOKENS.add(f'{accel}{direction}{radius}')
            # With optional end rotation
            for end_rot in ['L', 'R']:
                VALID_TOKENS.add(f'{accel}{direction}{radius}{end_rot}')

class MicromouseController:
    """
    Base class for implementing micromouse control logic.
    Extend this class and implement the get_instructions method.
    """
    
    def get_instructions(self, sensor_data: List[int], game_state: Dict[str, Any]) -> List[str]:
        """
        Override this method to implement your micromouse strategy.
        
        Args:
            sensor_data: List of 5 integers representing sensor readings at -90°, -45°, 0°, +45°, +90°
            game_state: Dictionary containing current game state
            
        Returns:
            List of movement tokens to execute
        """
        raise NotImplementedError("Implement your micromouse control logic here")
    
    def should_end(self, game_state: Dict[str, Any]) -> bool:
        """
        Override this method to determine when to end the challenge.
        
        Args:
            game_state: Dictionary containing current game state
            
        Returns:
            True if the challenge should end, False otherwise
        """
        return False

# Example controller implementation
class ExampleController(MicromouseController):
    """Example micromouse controller - replace with your own logic."""
    
    def get_instructions(self, sensor_data: List[int], game_state: Dict[str, Any]) -> List[str]:
        """
        Simple example: move forward if no obstacle ahead, otherwise turn right.
        Replace this with your own maze-solving algorithm.
        """
        # sensor_data[2] is the front sensor (0° direction)
        if sensor_data[2] == 0:  # No wall ahead
            return ['F1']  # Move forward at constant speed
        else:  # Wall ahead
            return ['R']  # Turn right
    
    def should_end(self, game_state: Dict[str, Any]) -> bool:
        """End if we've reached the goal or used too much time."""
        # End if we've used more than 250 seconds (leaving buffer)
        if game_state['total_time_ms'] > 250000:
            return True
        
        # End if we've reached the goal and have a good time
        if game_state['goal_reached'] and game_state['best_time_ms'] is not None:
            return True
        
        return False

# Initialize the controller immediately
controller = ExampleController()

def set_controller(new_controller: MicromouseController):
    """Set the micromouse controller implementation."""
    global controller
    controller = new_controller

def validate_request_body(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate the incoming request body according to API specification."""
    required_fields = [
        'game_uuid', 'sensor_data', 'total_time_ms', 'goal_reached',
        'best_time_ms', 'run_time_ms', 'run', 'momentum'
    ]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate sensor_data
    if not isinstance(data['sensor_data'], list) or len(data['sensor_data']) != 5:
        return False, "sensor_data must be a list of 5 integers"
    
    if not all(isinstance(x, int) and 0 <= x <= 1 for x in data['sensor_data']):
        return False, "sensor_data values must be 0 or 1"
    
    # Validate momentum range
    if not isinstance(data['momentum'], int) or not (-4 <= data['momentum'] <= 4):
        return False, "momentum must be an integer between -4 and 4"
    
    # Validate boolean fields
    if not isinstance(data['goal_reached'], bool):
        return False, "goal_reached must be a boolean"
    
    # Validate numeric fields
    numeric_fields = ['total_time_ms', 'run_time_ms', 'run']
    for field in numeric_fields:
        if not isinstance(data[field], (int, float)) or data[field] < 0:
            return False, f"{field} must be a non-negative number"
    
    # best_time_ms can be null or a non-negative number
    if data['best_time_ms'] is not None:
        if not isinstance(data['best_time_ms'], (int, float)) or data['best_time_ms'] < 0:
            return False, "best_time_ms must be null or a non-negative number"
    
    return True, ""

def validate_instructions(instructions: List[str]) -> tuple[bool, str]:
    """Validate movement tokens according to specification."""
    if not isinstance(instructions, list):
        return False, "instructions must be a list"
    
    for instruction in instructions:
        if not isinstance(instruction, str):
            return False, "all instructions must be strings"
        
        # Check if it's a valid token (this is a simplified check)
        # In a full implementation, you'd want more sophisticated validation
        if instruction not in VALID_TOKENS:
            # Check if it might be a corner turn we didn't enumerate
            if not _is_valid_corner_turn(instruction):
                return False, f"Invalid movement token: {instruction}"
    
    return True, ""

def _is_valid_corner_turn(token: str) -> bool:
    """Check if a token is a valid corner turn format."""
    if len(token) < 4:
        return False
    
    # Check format: (F0|F1|F2|V0|V1|V2)(L|R)(T|W)[(L|R)]
    accel_part = token[:2]  # F0, F1, F2, V0, V1, V2
    if accel_part not in ['F0', 'F1', 'F2', 'V0', 'V1', 'V2']:
        return False
    
    direction = token[2]  # L or R
    if direction not in ['L', 'R']:
        return False
    
    if len(token) < 4:
        return False
    
    radius = token[3]  # T or W
    if radius not in ['T', 'W']:
        return False
    
    # Optional end rotation
    if len(token) == 5:
        end_rot = token[4]
        if end_rot not in ['L', 'R']:
            return False
    elif len(token) > 5:
        return False
    
    return True

@app.route('/micro-mouse', methods=['POST'])
def micro_mouse():
    """
    Micromouse API endpoint as specified in the documentation.
    
    Expected request body:
    {
        "game_uuid": "xxxx",
        "sensor_data": [1, 1, 0, 1, 1],
        "total_time_ms": 0,
        "goal_reached": false,
        "best_time_ms": null,
        "run_time_ms": 0,
        "run": 0,
        "momentum": 0
    }
    
    Response body:
    {
        "instructions": ["F2", "F2", "BB"],
        "end": false
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'Invalid JSON in request body'}), 400
        
        # Validate request body
        valid, error_msg = validate_request_body(data)
        if not valid:
            return jsonify({'error': error_msg}), 400
        
        # Check if controller is set
        if controller is None:
            return jsonify({'error': 'No controller implementation set'}), 500
        
        # Extract game state
        game_state = {
            'game_uuid': data['game_uuid'],
            'total_time_ms': data['total_time_ms'],
            'goal_reached': data['goal_reached'],
            'best_time_ms': data['best_time_ms'],
            'run_time_ms': data['run_time_ms'],
            'run': data['run'],
            'momentum': data['momentum']
        }
        
        # Check if controller wants to end the challenge
        should_end = controller.should_end(game_state)
        if should_end:
            return jsonify({
                'instructions': [],
                'end': True
            })
        
        # Get instructions from controller
        instructions = controller.get_instructions(data['sensor_data'], game_state)
        
        # Validate instructions
        valid, error_msg = validate_instructions(instructions)
        if not valid:
            return jsonify({'error': error_msg}), 400
        
        # Return response
        return jsonify({
            'instructions': instructions,
            'end': False
        })
        
    except Exception as e:
        # More detailed error logging
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in micro_mouse endpoint: {error_details}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
