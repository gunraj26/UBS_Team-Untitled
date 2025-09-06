import os
import sys
import logging
from flask import request, jsonify
import json
from collections import deque, defaultdict
import heapq
import math

# Add parent directory to path to allow importing routes module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging

from flask import request

from routes import app

logger = logging.getLogger(__name__)

class FogOfWallGame:
    """
    Game state manager for Fog of Wall maze exploration
    """
    
    def __init__(self):
        self.games = {}  # Store game states by game_id
        
    def start_new_game(self, game_id, test_case):
        """Initialize a new game with test case data"""
        # Handle None or missing test_case
        if not test_case:
            logger.error(f"Empty or None test_case provided for game {game_id}")
            raise ValueError("test_case cannot be None or empty")
            
        # Validate test_case structure
        if not isinstance(test_case, dict):
            logger.error(f"Invalid test_case type for game {game_id}: {type(test_case)}")
            raise ValueError(f"test_case must be a dictionary, got {type(test_case)}")
            
        # Handle None or missing crows
        crows_data = test_case.get('crows', [])
        if not crows_data:
            logger.warning(f"No crows data in test_case for game {game_id}")
            crows_data = []
        elif not isinstance(crows_data, list):
            logger.error(f"Invalid crows data type for game {game_id}: {type(crows_data)}")
            raise ValueError(f"crows must be a list, got {type(crows_data)}")
            
        # Safely process crows, filtering out None values
        crows = {}
        for i, crow in enumerate(crows_data):
            if crow is None:
                logger.warning(f"Skipping None crow at index {i} in game {game_id}")
                continue
            if not isinstance(crow, dict):
                logger.warning(f"Skipping invalid crow at index {i} in game {game_id}: {type(crow)}")
                continue
            if 'id' not in crow or 'x' not in crow or 'y' not in crow:
                logger.warning(f"Skipping crow at index {i} in game {game_id} missing required fields: {crow}")
                continue
            try:
                x, y = int(crow['x']), int(crow['y'])
                crows[crow['id']] = {'x': x, 'y': y}
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping crow at index {i} in game {game_id} with invalid coordinates: {e}")
                continue
        
        if not crows:
            logger.error(f"No valid crows found in test_case for game {game_id}")
            raise ValueError("No valid crows found in test_case")
        
        # Validate grid size
        grid_size = test_case.get('length_of_grid', 10)
        try:
            grid_size = int(grid_size)
            if grid_size <= 0:
                logger.warning(f"Invalid grid_size {grid_size} for game {game_id}, using default 10")
                grid_size = 10
        except (ValueError, TypeError):
            logger.warning(f"Invalid grid_size type for game {game_id}, using default 10")
            grid_size = 10
            
        # Validate number of walls
        num_walls = test_case.get('num_of_walls', 0)
        try:
            num_walls = int(num_walls)
            if num_walls < 0:
                logger.warning(f"Invalid num_walls {num_walls} for game {game_id}, using 0")
                num_walls = 0
        except (ValueError, TypeError):
            logger.warning(f"Invalid num_walls type for game {game_id}, using 0")
            num_walls = 0
        
        self.games[game_id] = {
            'crows': crows,
            'grid_size': grid_size,
            'num_walls': num_walls,
            'discovered_walls': set(),
            'explored_cells': set(),
            'scan_results': {},  # Store scan results for each position
            'move_count': 0,
            'game_complete': False,
            'max_moves': min(grid_size * grid_size, 200)  # Limit moves to prevent timeout
        }
        logger.info(f"Started new game {game_id} with {len(crows)} crows, grid_size={grid_size}, num_walls={num_walls}")
        
    def get_crow_position(self, game_id, crow_id):
        """Get current position of a crow"""
        if game_id not in self.games:
            return None
        return self.games[game_id]['crows'].get(crow_id)
        
    def update_crow_position(self, game_id, crow_id, new_x, new_y):
        """Update crow position after a move"""
        if game_id in self.games and crow_id in self.games[game_id]['crows']:
            self.games[game_id]['crows'][crow_id] = {'x': new_x, 'y': new_y}
            self.games[game_id]['move_count'] += 1
            
    def add_scan_result(self, game_id, crow_id, x, y, scan_data):
        """Process and store scan results"""
        if game_id not in self.games:
            return
            
        if not scan_data or not isinstance(scan_data, list):
            return
            
        game = self.games[game_id]
        game['move_count'] += 1
        
        # Mark the center cell as explored
        game['explored_cells'].add((x, y))
        
        # Process the 5x5 scan grid
        for i, row in enumerate(scan_data):
            if not isinstance(row, list):
                continue
            for j, cell in enumerate(row):
                if cell == 'W':  # Wall found
                    # Convert relative position to absolute
                    wall_x = x + (j - 2)  # j-2 because center is at [2][2]
                    wall_y = y + (i - 2)  # i-2 because center is at [2][2]
                    
                    # Check bounds
                    if 0 <= wall_x < game['grid_size'] and 0 <= wall_y < game['grid_size']:
                        game['discovered_walls'].add((wall_x, wall_y))
                        
        # Store scan result for this position
        game['scan_results'][(x, y)] = scan_data
        
    def get_discovered_walls(self, game_id):
        """Get all discovered walls in submission format"""
        if game_id not in self.games:
            return []
        return [f"{x}-{y}" for x, y in self.games[game_id]['discovered_walls']]
        
    def is_game_complete(self, game_id):
        """Check if all walls have been discovered"""
        if game_id not in self.games:
            return False
        game = self.games[game_id]
        return len(game['discovered_walls']) >= game['num_walls']
        
    def get_game_stats(self, game_id):
        """Get current game statistics"""
        if game_id not in self.games:
            return None
        game = self.games[game_id]
        return {
            'walls_discovered': len(game['discovered_walls']),
            'total_walls': game['num_walls'],
            'cells_explored': len(game['explored_cells']),
            'move_count': game['move_count'],
            'completion_percentage': len(game['discovered_walls']) / game['num_walls'] * 100
        }

class MazeExplorer:
    """
    Intelligent maze exploration strategy with multi-crow coordination
    """
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.explored = set()
        self.walls = set()
        self.frontier = set()
        self.crow_assignments = {}  # Track which areas each crow is exploring
        
    def get_next_action(self, game_state, crows):
        """
        Determine the best next action for any crow using optimized exploration
        Returns (crow_id, action_type, direction_or_none)
        """
        # Check if we should submit early (found enough walls or running out of moves)
        walls_found = len(game_state['discovered_walls'])
        total_walls = game_state['num_walls']
        moves_used = game_state['move_count']
        max_moves = game_state['max_moves']
        
        # Submit early if we found all walls or are running out of moves
        # Be more conservative about early submission to find more walls
        if (walls_found >= total_walls or 
            moves_used >= max_moves * 0.95):  # Submit at 95% of max moves
            logger.info(f"Submitting: walls={walls_found}/{total_walls}, moves={moves_used}/{max_moves}")
            return None, 'submit', None
        
        # Strategy: Prioritize scanning unexplored positions, then move to new areas
        
        # First, find any crow that can scan an unexplored position
        # Prioritize crows in areas with more potential for wall discovery
        best_scan_crow = None
        best_scan_score = -1
        
        for crow_id, crow_pos in crows.items():
            if not crow_pos or not isinstance(crow_pos, dict):
                continue
            x, y = crow_pos['x'], crow_pos['y']
            
            # Skip if already scanned this position
            if (x, y) in game_state['scan_results']:
                continue
            
            # Calculate scan value for this position
            scan_score = self._calculate_scan_value(x, y, game_state)
            if scan_score > best_scan_score:
                best_scan_score = scan_score
                best_scan_crow = crow_id
                
        if best_scan_crow:
            logger.info(f"Scanning with crow {best_scan_crow} at position {crows[best_scan_crow]}")
            return best_scan_crow, 'scan', None
        
        # If all crows have scanned their positions, find the best move
        best_crow = None
        best_direction = None
        best_score = -1
        
        for crow_id, crow_pos in crows.items():
            if not crow_pos or not isinstance(crow_pos, dict):
                continue
                
            x = crow_pos.get('x')
            y = crow_pos.get('y')
            
            if x is None or y is None:
                continue
            
            # Try each direction and score the move
            for direction in ['N', 'S', 'E', 'W']:
                if not self._is_valid_move(crow_pos, direction, game_state):
                    continue
                    
                new_x, new_y = self._get_new_position(x, y, direction)
                
                # Skip if we've already explored this position
                if (new_x, new_y) in game_state['explored_cells']:
                    continue
                
                # Calculate score for this move
                score = self._calculate_move_score(new_x, new_y, game_state)
                
                if score > best_score:
                    best_score = score
                    best_crow = crow_id
                    best_direction = direction
        
        if best_crow and best_direction:
            logger.info(f"Moving crow {best_crow} {best_direction} from {crows[best_crow]}")
            return best_crow, 'move', best_direction
        
        # If no good moves found, try any valid move (even to explored areas)
        for crow_id, crow_pos in crows.items():
            if not crow_pos or not isinstance(crow_pos, dict):
                continue
                
            x = crow_pos.get('x')
            y = crow_pos.get('y')
            
            if x is None or y is None:
                continue
            
            # Try each direction for any valid move
            for direction in ['N', 'S', 'E', 'W']:
                if not self._is_valid_move(crow_pos, direction, game_state):
                    continue
                    
                new_x, new_y = self._get_new_position(x, y, direction)
                
                # Even if explored, try to move there if it's not a wall
                return crow_id, 'move', direction
        
        # If absolutely no moves possible, submit what we have
        logger.warning("No valid moves found, submitting current results")
        return None, 'submit', None
    
    def _calculate_move_score(self, x, y, game_state):
        """Calculate how valuable it would be to move to position (x, y)"""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 0
            
        # Check if position is already explored
        if (x, y) in game_state['explored_cells']:
            return 0
            
        # Check if position is a known wall
        if (x, y) in game_state['discovered_walls']:
            return 0
            
        # Base score for unexplored position
        score = 20  # Higher base score to encourage exploration
        
        # Bonus for being near unexplored areas (potential for scanning)
        unexplored_nearby = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                    if (check_x, check_y) not in game_state['explored_cells']:
                        unexplored_nearby += 1
        
        # Higher score for positions with more unexplored nearby cells
        score += min(unexplored_nearby, 15)  # Increased cap
        
        # Bonus for being far from explored areas (frontier exploration)
        min_distance = float('inf')
        for explored_x, explored_y in game_state['explored_cells']:
            distance = abs(x - explored_x) + abs(y - explored_y)
            min_distance = min(min_distance, distance)
            
        if min_distance != float('inf'):
            # Prefer positions that are far from explored areas
            score += min(min_distance, 10)  # Increased bonus
        else:
            # Unexplored area - very high priority
            score += 20
        
        return score
        
        
                
        
        
        
        
        
            
    def _calculate_scan_value(self, x, y, game_state):
        """Calculate how valuable it would be to scan at position (x, y)"""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 0
            
        # Skip if already scanned
        if (x, y) in game_state['scan_results']:
            return 0
            
        value = 0
        
        # Count unexplored cells in 5x5 area that could be walls
        unexplored_count = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                    if (check_x, check_y) not in game_state['explored_cells']:
                        unexplored_count += 1
        
        # Base value on unexplored cells in scan area
        value += unexplored_count * 3  # Increased multiplier
        
        # Bonus for being near known walls (might discover more walls nearby)
        wall_proximity = 0
        for wall_x, wall_y in game_state['discovered_walls']:
            distance = abs(x - wall_x) + abs(y - wall_y)
            if distance <= 4:  # Within 4 cells of a known wall
                wall_proximity += 1
                
        if wall_proximity > 0:
            value += wall_proximity * 5  # Increased bonus
            
        return value
        
    def _is_valid_move(self, crow_pos, direction, game_state):
        """Check if a move in the given direction is valid"""
        if not crow_pos or not isinstance(crow_pos, dict):
            return False
            
        x = crow_pos.get('x')
        y = crow_pos.get('y')
        
        if x is None or y is None:
            return False
            
        new_x, new_y = self._get_new_position(x, y, direction)
        
        if new_x is None or new_y is None:
            return False
        
        # Check bounds
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            return False
        
        # Check if the destination is a known wall
        if (new_x, new_y) in game_state['discovered_walls']:
            return False
            
        return True
        
    def _get_new_position(self, x, y, direction):
        """Get new position after moving in given direction"""
        if x is None or y is None:
            return None, None
            
        if direction == 'N':
            return x, y - 1
        elif direction == 'S':
            return x, y + 1
        elif direction == 'E':
            return x + 1, y
        elif direction == 'W':
            return x - 1, y
        return x, y

# Global game manager
game_manager = FogOfWallGame()

@app.route('/fog-of-wall', methods=['POST'])
def fog_of_wall():
    """
    Main endpoint for Fog of Wall game
    Handles initial setup, move results, scan results, and submissions
    """
    try:
        # Get raw data first for debugging
        raw_data = request.get_data()
        logger.info(f"Raw request data: {raw_data}")
        
        try:
            payload = request.get_json(force=True)
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            return jsonify({'error': 'Invalid JSON in request body'}), 400
            
        if not payload:
            logger.error("Empty payload received")
            return jsonify({'error': 'Empty request body'}), 400
            
        challenger_id = payload.get('challenger_id')
        game_id = payload.get('game_id')
        
        # Determine request type for better logging
        has_test_case = 'test_case' in payload and payload['test_case'] is not None and payload['test_case'] != 'null'
        has_previous_action = 'previous_action' in payload and payload['previous_action'] is not None
        
        request_type = "initial" if has_test_case else "follow-up" if has_previous_action else "invalid"
        logger.info(f"Received {request_type} request: challenger_id={challenger_id}, game_id={game_id}, has_test_case={has_test_case}, has_previous_action={has_previous_action}")
        
        # Log full payload for debugging
        logger.info(f"Full payload: {payload}")
        
        # Log test_case structure for debugging
        if 'test_case' in payload:
            test_case = payload['test_case']
            logger.info(f"Test case data: {test_case} (type: {type(test_case)})")
            if isinstance(test_case, dict) and 'crows' in test_case:
                logger.info(f"Crows data: {test_case['crows']}")
            elif test_case is None or test_case == 'null':
                logger.info("Test case is None/null - this is expected for follow-up requests")
        
        if not challenger_id or not game_id:
            return jsonify({'error': 'Missing challenger_id or game_id'}), 400
            
        # Check if this is an initial request with valid test_case data
        if 'test_case' in payload and payload['test_case'] is not None and payload['test_case'] != 'null':
            test_case = payload['test_case']
            
            # Add validation for test_case
            if not isinstance(test_case, dict):
                logger.error(f"Invalid test_case type: {type(test_case)}, value: {test_case}")
                return jsonify({'error': 'Invalid test_case data - must be a dictionary'}), 400
                
            # Validate crows data before starting game
            crows_data = test_case.get('crows', [])
            if not crows_data or not isinstance(crows_data, list):
                logger.error(f"Invalid crows data: {crows_data}")
                return jsonify({'error': 'No valid crows data found in test_case'}), 400
                
            # Check if game already exists
            if game_id in game_manager.games:
                logger.warning(f"Game {game_id} already exists, restarting with new test case")
                # Restart the game with the new test case
                try:
                    game_manager.start_new_game(game_id, test_case)
                except Exception as e:
                    logger.error(f"Failed to restart game: {str(e)}")
                    return jsonify({'error': f'Failed to restart game: {str(e)}'}), 400
            else:
                # Initialize new game
                try:
                    game_manager.start_new_game(game_id, test_case)
                except Exception as e:
                    logger.error(f"Failed to start new game: {str(e)}")
                    return jsonify({'error': f'Failed to start new game: {str(e)}'}), 400
            
            # Get initial action
            game_state = game_manager.games[game_id]
            crows = game_state['crows']
            
            # Check if we have any crows
            if not crows:
                return jsonify({'error': 'No crows available'}), 400
                
            # Start with scanning at initial positions
            first_crow_id = list(crows.keys())[0]
            return jsonify({
                'challenger_id': challenger_id,
                'game_id': game_id,
                'crow_id': first_crow_id,
                'action_type': 'scan'
            })
            
        # Handle previous action result
        previous_action = payload.get('previous_action')
        if not previous_action:
            # If we don't have a test_case and no previous_action, this is an invalid request
            logger.error(f"Invalid request: no test_case and no previous_action for game {game_id}")
            return jsonify({'error': 'Invalid request: must provide either test_case or previous_action'}), 400
            
        action_type = previous_action.get('your_action')
        crow_id = previous_action.get('crow_id')
        
        # Validate that we have the required fields
        if not action_type or not crow_id:
            return jsonify({'error': 'Missing action_type or crow_id in previous_action'}), 400
            
        # Check if game exists
        if game_id not in game_manager.games:
            logger.error(f"Game {game_id} not found when processing previous action")
            return jsonify({'error': 'Game not found'}), 404
        
        if action_type == 'move':
            # Process move result
            move_result = previous_action.get('move_result')
            if move_result:
                new_x, new_y = None, None
                
                # Handle list format [x, y]
                if isinstance(move_result, list) and len(move_result) == 2:
                    new_x, new_y = move_result
                # Handle dict format {x: x, y: y} or {crow_id: id, x: x, y: y}
                elif isinstance(move_result, dict):
                    new_x = move_result.get('x')
                    new_y = move_result.get('y')
                
                # Validate coordinates are numbers
                if (new_x is not None and new_y is not None and 
                    isinstance(new_x, (int, float)) and isinstance(new_y, (int, float))):
                    game_manager.update_crow_position(game_id, crow_id, int(new_x), int(new_y))
                    logger.info(f"Updated crow {crow_id} position to ({int(new_x)}, {int(new_y)})")
                else:
                    logger.warning(f"Invalid move result coordinates: {move_result}")
            else:
                logger.warning(f"Invalid move result format: {move_result}")
                
        elif action_type == 'scan':
            # Process scan result
            scan_result = previous_action.get('scan_result')
            crow_pos = game_manager.get_crow_position(game_id, crow_id)
            if crow_pos and scan_result and isinstance(scan_result, list):
                # Validate scan result is 5x5 grid
                if len(scan_result) == 5 and all(isinstance(row, list) and len(row) == 5 for row in scan_result):
                    game_manager.add_scan_result(game_id, crow_id, crow_pos['x'], crow_pos['y'], scan_result)
                else:
                    logger.warning(f"Invalid scan result format: {scan_result}")
            else:
                logger.warning(f"Invalid scan result or crow position: scan_result={scan_result}, crow_pos={crow_pos}")
                
        # Check if game is complete or move limit reached
        game_state = game_manager.games[game_id]
        if game_manager.is_game_complete(game_id) or game_state['move_count'] >= game_state['max_moves']:
            # Submit the discovered walls
            discovered_walls = game_manager.get_discovered_walls(game_id)
            logger.info(f"Game {game_id} completed: walls={len(discovered_walls)}, moves={game_state['move_count']}, max_moves={game_state['max_moves']}")
            return jsonify({
                'challenger_id': challenger_id,
                'game_id': game_id,
                'action_type': 'submit',
                'submission': discovered_walls
            })
            
        # Get next action
        if game_id not in game_manager.games:
            return jsonify({'error': 'Game not found'}), 404
            
        game_state = game_manager.games[game_id]
        if not game_state:
            return jsonify({'error': 'Game state is None'}), 500
            
        crows = game_state.get('crows', {})
        if not crows:
            return jsonify({'error': 'No crows available'}), 400
            
        grid_size = game_state.get('grid_size')
        if not grid_size:
            return jsonify({'error': 'Grid size not found'}), 500
            
        try:
            explorer = MazeExplorer(grid_size)
            next_crow, next_action, direction = explorer.get_next_action(game_state, crows)
        except Exception as e:
            logger.error(f"Error in get_next_action: {str(e)}")
            # Fallback: submit what we have
            discovered_walls = game_manager.get_discovered_walls(game_id)
            return jsonify({
                'challenger_id': challenger_id,
                'game_id': game_id,
                'action_type': 'submit',
                'submission': discovered_walls
            })
        
        if not next_crow or not next_action or next_action == 'submit':
            # No valid actions or submit requested, submit what we have
            discovered_walls = game_manager.get_discovered_walls(game_id)
            logger.info(f"Submitting game {game_id} with {len(discovered_walls)} walls found")
            return jsonify({
                'challenger_id': challenger_id,
                'game_id': game_id,
                'action_type': 'submit',
                'submission': discovered_walls
            })
            
        if next_action == 'scan':
            return jsonify({
                'challenger_id': challenger_id,
                'game_id': game_id,
                'crow_id': next_crow,
                'action_type': 'scan'
            })
        elif next_action == 'move':
            # Validate direction
            if direction not in ['N', 'S', 'E', 'W']:
                logger.warning(f"Invalid direction: {direction}, submitting instead")
                discovered_walls = game_manager.get_discovered_walls(game_id)
                return jsonify({
                    'challenger_id': challenger_id,
                    'game_id': game_id,
                    'action_type': 'submit',
                    'submission': discovered_walls
                })
            return jsonify({
                'challenger_id': challenger_id,
                'game_id': game_id,
                'crow_id': next_crow,
                'action_type': 'move',
                'direction': direction
            })
        else:
            # Fallback: submit current results
            discovered_walls = game_manager.get_discovered_walls(game_id)
            return jsonify({
                'challenger_id': challenger_id,
                'game_id': game_id,
                'action_type': 'submit',
                'submission': discovered_walls
            })
            
    except Exception as e:
        logger.error(f"Error in fog_of_wall endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/fog-of-wall/stats/<game_id>', methods=['GET'])
def get_game_stats(game_id):
    """Get statistics for a specific game"""
    stats = game_manager.get_game_stats(game_id)
    if stats is None:
        return jsonify({'error': 'Game not found'}), 404
    return jsonify(stats)

if __name__ == "__main__":
    # Only run the app if this file is executed directly
    app.run(debug=True, port=5001)
