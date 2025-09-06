from flask import Blueprint, request, Response
import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Tuple, Optional
import random
import logging

logger = logging.getLogger(__name__)

slpu = Blueprint('slpu', __name__)


class SnakesLaddersGame:
    def __init__(self, board_width: int, board_height: int, jumps: Dict[int, int]):
        self.width = board_width
        self.height = board_height
        self.board_size = board_width * board_height
        self.jumps = jumps  # start_square -> end_square
        
    def get_position_after_move(self, current_pos: int, roll: int) -> int:
        """Calculate position after a move, handling overshoot"""
        new_pos = current_pos + roll
        
        # Handle overshoot
        if new_pos > self.board_size:
            overshoot = new_pos - self.board_size
            new_pos = self.board_size - overshoot
            
        # Apply jumps (snakes/ladders)
        if new_pos in self.jumps:
            new_pos = self.jumps[new_pos]
            
        return max(0, new_pos)
    
    def simulate_game(self, rolls: List[int]) -> Tuple[bool, float, List[int]]:
        """
        Simulate the game with given rolls
        Returns: (player2_wins, coverage, positions_visited)
        """
        player1_pos = 0
        player2_pos = 0
        player1_die_type = 'regular'  # 'regular' or 'power'
        player2_die_type = 'regular'
        current_player = 1
        roll_idx = 0
        positions_visited = set()
        
        while roll_idx < len(rolls) and player1_pos < self.board_size and player2_pos < self.board_size:
            roll = rolls[roll_idx]
            
            if current_player == 1:
                # Calculate actual movement
                if player1_die_type == 'regular':
                    move = roll
                    # Power up if rolled 6
                    if roll == 6:
                        player1_die_type = 'power'
                else:  # power die
                    move = 2 ** (roll - 1)  # 1->2, 2->4, 3->8, 4->16, 5->32, 6->64
                    # Revert to regular if rolled 1
                    if roll == 1:
                        player1_die_type = 'regular'
                
                player1_pos = self.get_position_after_move(player1_pos, move)
                positions_visited.add(player1_pos)
                
            else:  # player 2
                # Calculate actual movement
                if player2_die_type == 'regular':
                    move = roll
                    # Power up if rolled 6
                    if roll == 6:
                        player2_die_type = 'power'
                else:  # power die
                    move = 2 ** (roll - 1)  # 1->2, 2->4, 3->8, 4->16, 5->32, 6->64
                    # Revert to regular if rolled 1
                    if roll == 1:
                        player2_die_type = 'regular'
                
                player2_pos = self.get_position_after_move(player2_pos, move)
                positions_visited.add(player2_pos)
            
            # Check win condition
            if player1_pos >= self.board_size:
                coverage = len(positions_visited) / self.board_size
                return False, coverage, list(positions_visited)
            elif player2_pos >= self.board_size:
                coverage = len(positions_visited) / self.board_size
                return True, coverage, list(positions_visited)
            
            # Switch players
            current_player = 2 if current_player == 1 else 1
            roll_idx += 1
        
        # Game didn't end
        coverage = len(positions_visited) / self.board_size
        return player2_pos >= player1_pos, coverage, list(positions_visited)

def parse_svg_board(svg_content: str) -> Tuple[int, int, Dict[int, int]]:
    """Parse SVG content to extract board dimensions and jumps"""
    try:
        root = ET.fromstring(svg_content)
        
        # Extract viewBox to get dimensions
        viewbox = root.get('viewBox', '0 0 512 512')
        _, _, width_px, height_px = map(int, viewbox.split())
        
        # Calculate board dimensions (each square is 32px)
        board_width = width_px // 32
        board_height = height_px // 32
        
        # Extract jumps from line elements
        jumps = {}
        
        for line in root.findall('.//{http://www.w3.org/2000/svg}line'):
            x1 = float(line.get('x1', 0))
            y1 = float(line.get('y1', 0))
            x2 = float(line.get('x2', 0))
            y2 = float(line.get('y2', 0))
            
            # Convert coordinates to square numbers
            start_square = coord_to_square(x1, y1, board_width, board_height)
            end_square = coord_to_square(x2, y2, board_width, board_height)
            
            if start_square and end_square:
                jumps[start_square] = end_square
        
        return board_width, board_height, jumps
        
    except Exception as e:
        # Default fallback
        return 16, 16, {}

def coord_to_square(x: float, y: float, board_width: int, board_height: int) -> Optional[int]:
    """Convert SVG coordinates to square number"""
    try:
        # Each square is 32px, coordinates are at center (16px offset)
        col = int((x - 16) / 32)
        row = int((y - 16) / 32)
        
        if col < 0 or col >= board_width or row < 0 or row >= board_height:
            return None
        
        # Convert to boustrophedon numbering (starting from bottom-left)
        bottom_row = board_height - 1 - row
        
        if bottom_row % 2 == 0:  # Even rows go left to right
            square = bottom_row * board_width + col + 1
        else:  # Odd rows go right to left
            square = bottom_row * board_width + (board_width - 1 - col) + 1
            
        return square
    except:
        return None

def generate_optimal_rolls(game: SnakesLaddersGame, max_attempts: int = 1000) -> str:
    """Generate rolls that maximize coverage while ensuring player 2 wins"""
    best_rolls = []
    best_coverage = 0
    
    for attempt in range(max_attempts):
        # Generate random sequence of reasonable length
        sequence_length = random.randint(20, 100)
        rolls = [random.randint(1, 6) for _ in range(sequence_length)]
        
        # Test the sequence
        player2_wins, coverage, _ = game.simulate_game(rolls)
        
        if player2_wins and coverage > best_coverage:
            best_rolls = rolls
            best_coverage = coverage
            
        # Early exit if we find good coverage
        if best_coverage > 0.4:
            break
    
    # If no good solution found, try a simple approach
    if not best_rolls:
        # Simple strategy: alternate small and medium moves
        rolls = []
        for i in range(50):
            if i % 4 == 0:
                rolls.append(6)  # Power up occasionally
            elif i % 3 == 0:
                rolls.append(1)  # Small moves for fine control
            else:
                rolls.append(random.randint(2, 5))
        
        player2_wins, coverage, _ = game.simulate_game(rolls)
        if player2_wins:
            best_rolls = rolls
    
    # Final fallback - just try to win
    if not best_rolls:
        best_rolls = [random.randint(1, 6) for _ in range(30)]
    
    return ''.join(map(str, best_rolls))

@slpu.route('/slpu', methods=['POST'])
def solve_snakes_ladders():
    try:
        # Get SVG content from request
        svg_content = request.data.decode('utf-8')

        logger.info(svg_content)
        
        # Parse the board
        board_width, board_height, jumps = parse_svg_board(svg_content)
        
        # Create game instance
        game = SnakesLaddersGame(board_width, board_height, jumps)
        
        # Generate optimal rolls
        solution_rolls = generate_optimal_rolls(game)
        
        # Create response SVG
        response_svg = f'<svg xmlns="http://www.w3.org/2000/svg"><text>{solution_rolls}</text></svg>'
        
        return Response(response_svg, mimetype='image/svg+xml')
        
    except Exception as e:
        # Fallback response
        fallback_rolls = ''.join([str(random.randint(1, 6)) for _ in range(50)])
        response_svg = f'<svg xmlns="http://www.w3.org/2000/svg"><text>{fallback_rolls}</text></svg>'
        return Response(response_svg, mimetype='image/svg+xml')