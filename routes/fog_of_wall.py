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
            logger.warning(f"Empty test_case provided for game {game_id}")
            test_case = {}
            
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
            'game_complete': False
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
        Determine the best next action for any crow using coordinated exploration
        Returns (crow_id, action_type, direction_or_none)
        """
        # Strategy: Multi-crow coordinated exploration with intelligent scanning
        
        # First, check if any crow should scan using strategic priority
        scan_priorities = self._get_scanning_priority(game_state, crows)
        if scan_priorities:
            # Return the highest priority scan
            return scan_priorities[0][0], 'scan', None
                
        # If all crows have scanned their positions, find the best coordinated move
        best_crow, best_direction = self._find_best_coordinated_move(game_state, crows)
        if best_crow and best_direction:
            return best_crow, 'move', best_direction
            
        # Fallback: move any crow in a random direction
        for crow_id, crow_pos in crows.items():
            for direction in ['N', 'S', 'E', 'W']:
                if self._is_valid_move(crow_pos, direction):
                    return crow_id, 'move', direction
                    
        return None, None, None
        
    def _find_best_coordinated_move(self, game_state, crows):
        """Find the best move using coordinated multi-crow strategy"""
        # Assign exploration areas to crows if not already assigned
        if not self.crow_assignments:
            self._assign_exploration_areas(game_state, crows)
            
        best_score = -1
        best_crow = None
        best_direction = None
        
        for crow_id, crow_pos in crows.items():
            if not crow_pos or not isinstance(crow_pos, dict):
                continue
                
            x = crow_pos.get('x')
            y = crow_pos.get('y')
            
            if x is None or y is None:
                continue
            
            # Get assigned area for this crow
            assigned_area = self.crow_assignments.get(crow_id, None)
            
            # Try each direction
            for direction in ['N', 'S', 'E', 'W']:
                if not self._is_valid_move(crow_pos, direction):
                    continue
                    
                new_x, new_y = self._get_new_position(x, y, direction)
                
                # Skip if we've already explored this position
                if (new_x, new_y) in game_state['explored_cells']:
                    continue
                    
                # Calculate exploration value with coordination
                score = self._calculate_coordinated_value(new_x, new_y, game_state, crow_id, assigned_area)
                
                if score > best_score:
                    best_score = score
                    best_crow = crow_id
                    best_direction = direction
                    
        return best_crow, best_direction
        
    def _assign_exploration_areas(self, game_state, crows):
        """Assign different areas of the grid to different crows for efficient exploration"""
        crow_ids = list(crows.keys())
        grid_size = game_state['grid_size']
        
        if len(crow_ids) == 1:
            # Single crow explores everything
            self.crow_assignments[crow_ids[0]] = (0, 0, grid_size, grid_size)
        elif len(crow_ids) == 2:
            # Two crows: split vertically
            mid_x = grid_size // 2
            self.crow_assignments[crow_ids[0]] = (0, 0, mid_x, grid_size)
            self.crow_assignments[crow_ids[1]] = (mid_x, 0, grid_size, grid_size)
        elif len(crow_ids) == 3:
            # Three crows: split into quadrants with overlap
            mid_x = grid_size // 2
            mid_y = grid_size // 2
            self.crow_assignments[crow_ids[0]] = (0, 0, mid_x + 2, mid_y + 2)
            self.crow_assignments[crow_ids[1]] = (mid_x - 2, 0, grid_size, mid_y + 2)
            self.crow_assignments[crow_ids[2]] = (0, mid_y - 2, grid_size, grid_size)
        else:
            # More than 3 crows: assign based on current positions
            for i, crow_id in enumerate(crow_ids):
                crow_pos = crows[crow_id]
                if not crow_pos or not isinstance(crow_pos, dict):
                    continue
                x = crow_pos.get('x')
                y = crow_pos.get('y')
                if x is None or y is None:
                    continue
                # Assign a small area around each crow's starting position
                self.crow_assignments[crow_id] = (max(0, x-5), max(0, y-5), 
                                                min(grid_size, x+6), min(grid_size, y+6))
                
    def _calculate_coordinated_value(self, x, y, game_state, crow_id, assigned_area):
        """Calculate exploration value considering crow's assigned area"""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 0
            
        # Check if position is already explored
        if (x, y) in game_state['explored_cells']:
            return 0
            
        # Check if position is a known wall
        if (x, y) in game_state['discovered_walls']:
            return 0
            
        base_value = 10
        
        # Bonus for being in assigned area
        if assigned_area:
            min_x, min_y, max_x, max_y = assigned_area
            if min_x <= x < max_x and min_y <= y < max_y:
                base_value += 5
                
        # Calculate distance from explored areas
        min_distance = float('inf')
        for explored_x, explored_y in game_state['explored_cells']:
            distance = abs(x - explored_x) + abs(y - explored_y)
            min_distance = min(min_distance, distance)
            
        # Higher value for positions further from explored areas
        if min_distance == float('inf'):
            return base_value + 10  # Unexplored area
        else:
            return base_value + max(0, 10 - min_distance)
        
    def _find_best_move(self, game_state, crows):
        """Find the best move for any crow using exploration strategy"""
        best_score = -1
        best_crow = None
        best_direction = None
        
        for crow_id, crow_pos in crows.items():
            if not crow_pos or not isinstance(crow_pos, dict):
                continue
                
            x = crow_pos.get('x')
            y = crow_pos.get('y')
            
            if x is None or y is None:
                continue
            
            # Try each direction
            for direction in ['N', 'S', 'E', 'W']:
                if not self._is_valid_move(crow_pos, direction):
                    continue
                    
                new_x, new_y = self._get_new_position(x, y, direction)
                
                # Skip if we've already explored this position
                if (new_x, new_y) in game_state['explored_cells']:
                    continue
                    
                # Calculate exploration value
                score = self._calculate_exploration_value(new_x, new_y, game_state)
                
                if score > best_score:
                    best_score = score
                    best_crow = crow_id
                    best_direction = direction
                    
        return best_crow, best_direction
        
    def _find_optimal_scanning_positions(self, game_state, crows):
        """Find optimal positions for scanning to maximize wall discovery"""
        optimal_positions = []
        
        # Find unexplored areas that could benefit from scanning
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) in game_state['explored_cells']:
                    continue
                    
                # Calculate potential wall discovery value
                potential_walls = self._estimate_potential_walls(x, y, game_state)
                if potential_walls > 0:
                    # Find closest crow to this position
                    closest_crow = self._find_closest_crow(x, y, crows)
                    if closest_crow:
                        distance = self._calculate_distance(
                            crows[closest_crow]['x'], crows[closest_crow]['y'], x, y
                        )
                        optimal_positions.append((x, y, closest_crow, potential_walls, distance))
                        
        # Sort by potential walls (descending) and distance (ascending)
        optimal_positions.sort(key=lambda x: (-x[3], x[4]))
        return optimal_positions
        
    def _estimate_potential_walls(self, x, y, game_state):
        """Estimate how many walls might be discovered by scanning at position (x, y)"""
        potential_count = 0
        
        # Check 5x5 area around the position
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                    # If this cell is unexplored, it could potentially be a wall
                    if (check_x, check_y) not in game_state['explored_cells']:
                        potential_count += 1
                        
        return potential_count
        
    def _find_closest_crow(self, target_x, target_y, crows):
        """Find the crow closest to the target position"""
        closest_crow = None
        min_distance = float('inf')
        
        for crow_id, crow_pos in crows.items():
            distance = self._calculate_distance(
                crow_pos['x'], crow_pos['y'], target_x, target_y
            )
            if distance < min_distance:
                min_distance = distance
                closest_crow = crow_id
                
        return closest_crow
        
    def _calculate_distance(self, x1, y1, x2, y2):
        """Calculate Manhattan distance between two positions"""
        return abs(x1 - x2) + abs(y1 - y2)
        
    def _get_scanning_priority(self, game_state, crows):
        """Determine which crow should scan next based on strategic value"""
        scan_priorities = []
        
        for crow_id, crow_pos in crows.items():
            x, y = crow_pos['x'], crow_pos['y']
            
            # Skip if already scanned this position
            if (x, y) in game_state['scan_results']:
                continue
                
            # Calculate strategic value of scanning here
            strategic_value = self._calculate_scanning_value(x, y, game_state)
            scan_priorities.append((crow_id, strategic_value))
            
        # Sort by strategic value (descending)
        scan_priorities.sort(key=lambda x: x[1], reverse=True)
        return scan_priorities
        
    def _calculate_scanning_value(self, x, y, game_state):
        """Calculate the strategic value of scanning at position (x, y)"""
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
        value += unexplored_count * 2
        
        # Bonus for being in a strategic position (center of unexplored areas)
        if unexplored_count > 10:  # High density of unexplored cells
            value += 5
            
        # Check if this position is near known walls (might discover more walls nearby)
        wall_proximity = 0
        for wall_x, wall_y in game_state['discovered_walls']:
            distance = self._calculate_distance(x, y, wall_x, wall_y)
            if distance <= 3:  # Within 3 cells of a known wall
                wall_proximity += 1
                
        if wall_proximity > 0:
            value += wall_proximity * 3
            
        return value
        
    def _calculate_exploration_value(self, x, y, game_state):
        """Calculate how valuable it would be to explore position (x, y)"""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 0
            
        # Check if position is already explored
        if (x, y) in game_state['explored_cells']:
            return 0
            
        # Check if position is a known wall
        if (x, y) in game_state['discovered_walls']:
            return 0
            
        # Calculate distance from explored areas (prefer unexplored frontiers)
        min_distance = float('inf')
        for explored_x, explored_y in game_state['explored_cells']:
            distance = abs(x - explored_x) + abs(y - explored_y)
            min_distance = min(min_distance, distance)
            
        # Higher value for positions further from explored areas
        # But not too far (to avoid getting lost)
        if min_distance == float('inf'):
            return 10  # Unexplored area
        else:
            return max(0, 10 - min_distance)
            
    def _is_valid_move(self, crow_pos, direction):
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
        payload = request.get_json(force=True)
        challenger_id = payload.get('challenger_id')
        game_id = payload.get('game_id')
        
        logger.info(f"Received request: challenger_id={challenger_id}, game_id={game_id}, has_test_case={'test_case' in payload}, has_previous_action={'previous_action' in payload}")
        
        # Log test_case structure for debugging
        if 'test_case' in payload:
            test_case = payload['test_case']
            logger.info(f"Test case data: {test_case}")
            if isinstance(test_case, dict) and 'crows' in test_case:
                logger.info(f"Crows data: {test_case['crows']}")
        
        if not challenger_id or not game_id:
            return jsonify({'error': 'Missing challenger_id or game_id'}), 400
            
        # Check if this is an initial request
        if 'test_case' in payload:
            # Check if game already exists
            if game_id in game_manager.games:
                logger.warning(f"Game {game_id} already exists, returning current state")
                # Return the existing game state instead of overwriting
                game_state = game_manager.games[game_id]
                crows = game_state['crows']
                
                if not crows:
                    logger.error(f"Game {game_id} exists but has no crows. This indicates a corrupted game state.")
                    # Try to restart the game with the new test case
                    logger.info(f"Attempting to restart game {game_id} with new test case")
                    try:
                        game_manager.start_new_game(game_id, payload['test_case'])
                        game_state = game_manager.games[game_id]
                        crows = game_state['crows']
                        
                        if not crows:
                            return jsonify({'error': 'No crows available even after restart'}), 400
                    except Exception as e:
                        logger.error(f"Failed to restart game: {str(e)}")
                        return jsonify({'error': f'Failed to restart game: {str(e)}'}), 400
                    
                # Return the first available crow for scanning
                first_crow_id = list(crows.keys())[0]
                return jsonify({
                    'challenger_id': challenger_id,
                    'game_id': game_id,
                    'crow_id': first_crow_id,
                    'action_type': 'scan'
                })
            
            # Initialize new game only if it doesn't exist
            test_case = payload['test_case']
            
            # Add validation for test_case
            if not test_case or not isinstance(test_case, dict):
                logger.error(f"Invalid test_case: {test_case}")
                return jsonify({'error': 'Invalid test_case data'}), 400
                
            # Validate crows data before starting game
            crows_data = test_case.get('crows', [])
            if not crows_data or not isinstance(crows_data, list):
                logger.error(f"Invalid crows data: {crows_data}")
                return jsonify({'error': 'No valid crows data found in test_case'}), 400
                
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
                
            explorer = MazeExplorer(game_state['grid_size'])
            
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
            return jsonify({'error': 'No previous action provided'}), 400
            
        action_type = previous_action.get('your_action')
        crow_id = previous_action.get('crow_id')
        
        # Validate that we have the required fields
        if not action_type or not crow_id:
            return jsonify({'error': 'Missing action_type or crow_id in previous_action'}), 400
        
        if action_type == 'move':
            # Process move result
            move_result = previous_action.get('move_result')
            if move_result and isinstance(move_result, list) and len(move_result) == 2:
                new_x, new_y = move_result
                # Validate coordinates are numbers
                if isinstance(new_x, (int, float)) and isinstance(new_y, (int, float)):
                    game_manager.update_crow_position(game_id, crow_id, int(new_x), int(new_y))
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
                
        # Check if game is complete
        if game_manager.is_game_complete(game_id):
            # Submit the discovered walls
            discovered_walls = game_manager.get_discovered_walls(game_id)
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
        
        if not next_crow or not next_action:
            # No valid actions, submit what we have
            discovered_walls = game_manager.get_discovered_walls(game_id)
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
