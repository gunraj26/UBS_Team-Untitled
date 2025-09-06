from flask import Blueprint, request, Response
import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Tuple, Optional
import random

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
        
        # Handle overshoot - bounce back from the end
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
        
        while roll_idx < len(rolls):
            if player1_pos >= self.board_size or player2_pos >= self.board_size:
                break
                
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
                if player1_pos > 0:
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
                if player2_pos > 0:
                    positions_visited.add(player2_pos)
            
            # Check win condition
            if player1_pos >= self.board_size:
                coverage = len(positions_visited) / self.board_size
                return False, coverage, list(positions_visited)  # Player 1 wins
            elif player2_pos >= self.board_size:
                coverage = len(positions_visited) / self.board_size
                return True, coverage, list(positions_visited)   # Player 2 wins
            
            # Switch players
            current_player = 2 if current_player == 1 else 1
            roll_idx += 1
        
        # Game didn't end within roll limit
        coverage = len(positions_visited) / self.board_size
        return player2_pos > player1_pos, coverage, list(positions_visited)

def parse_svg_board(svg_content: str) -> Tuple[int, int, Dict[int, int]]:
    """Parse SVG content to extract board dimensions and jumps"""
    try:
        # Remove XML namespaces for easier parsing
        svg_content = svg_content.replace('xmlns="http://www.w3.org/2000/svg"', '')
        root = ET.fromstring(svg_content)
        
        # Extract viewBox to get dimensions
        viewbox = root.get('viewBox', '0 0 512 512')
        parts = viewbox.strip().split()
        if len(parts) >= 4:
            width_px = int(parts[2])
            height_px = int(parts[3])
        else:
            width_px, height_px = 512, 512
        
        # Calculate board dimensions (each square is 32px)
        board_width = width_px // 32
        board_height = height_px // 32
        
        # Extract jumps from line elements
        jumps = {}
        
        # Look for line elements in the SVG
        for line in root.iter():
            if line.tag == 'line' or line.tag.endswith('}line'):
                try:
                    x1 = float(line.get('x1', 0))
                    y1 = float(line.get('y1', 0))
                    x2 = float(line.get('x2', 0))
                    y2 = float(line.get('y2', 0))
                    
                    # Convert coordinates to square numbers
                    start_square = coord_to_square(x1, y1, board_width, board_height)
                    end_square = coord_to_square(x2, y2, board_width, board_height)
                    
                    if start_square and end_square and start_square != end_square:
                        jumps[start_square] = end_square
                except (ValueError, TypeError):
                    continue
        
        return board_width, board_height, jumps
        
    except Exception as e:
        # Default fallback - try to extract dimensions from viewBox string
        try:
            viewbox_match = re.search(r'viewBox="([^"]*)"', svg_content)
            if viewbox_match:
                parts = viewbox_match.group(1).split()
                if len(parts) >= 4:
                    width_px = int(parts[2])
                    height_px = int(parts[3])
                    return width_px // 32, height_px // 32, {}
        except:
            pass
        return 16, 16, {}

def coord_to_square(x: float, y: float, board_width: int, board_height: int) -> Optional[int]:
    """Convert SVG coordinates to square number using boustrophedon pattern"""
    try:
        # Each square is 32px, coordinates are at center (16px offset)
        col = int((x - 16) / 32)
        row = int((y - 16) / 32)
        
        if col < 0 or col >= board_width or row < 0 or row >= board_height:
            return None
        
        # Convert to boustrophedon numbering (starting from bottom-left)
        # Bottom row is row 0, top row is row (board_height-1)
        bottom_row = board_height - 1 - row
        
        if bottom_row % 2 == 0:  # Even rows (0, 2, 4...) go left to right
            square = bottom_row * board_width + col + 1
        else:  # Odd rows (1, 3, 5...) go right to left
            square = bottom_row * board_width + (board_width - 1 - col) + 1
            
        return square if 1 <= square <= board_width * board_height else None
    except:
        return None

def generate_strategic_rolls(game: SnakesLaddersGame, max_attempts: int = 2000) -> str:
    """Generate rolls using multiple strategies to maximize coverage"""
    best_rolls = []
    best_coverage = 0
    
    strategies = [
        lambda: generate_coverage_focused_rolls(game),
        lambda: generate_balanced_rolls(game),
        lambda: generate_power_die_rolls(game),
        lambda: generate_random_rolls(game, 60),
        lambda: generate_random_rolls(game, 80),
        lambda: generate_random_rolls(game, 100),
    ]
    
    for attempt in range(max_attempts):
        # Choose strategy
        strategy = strategies[attempt % len(strategies)]
        rolls = strategy()
        
        # Test the sequence
        try:
            player2_wins, coverage, _ = game.simulate_game(rolls)
            
            if player2_wins and coverage > best_coverage:
                best_rolls = rolls
                best_coverage = coverage
                
            # Early exit if we find very good coverage
            if best_coverage > 0.6:
                break
                
        except Exception:
            continue
    
    # If no solution found, return a simple fallback
    if not best_rolls:
        best_rolls = [random.randint(1, 6) for _ in range(50)]
    
    return ''.join(map(str, best_rolls))

def generate_coverage_focused_rolls(game: SnakesLaddersGame) -> List[int]:
    """Generate rolls focused on maximizing board coverage"""
    rolls = []
    
    # Start with small moves to visit early squares
    for i in range(15):
        if i % 3 == 0:
            rolls.append(1)  # Small steps
        elif i % 3 == 1:
            rolls.append(2)  # Medium steps
        else:
            rolls.append(random.randint(3, 4))  # Medium-large steps
    
    # Mix in some power-up opportunities
    for i in range(10):
        if random.random() < 0.3:
            rolls.append(6)  # Power up
        else:
            rolls.append(random.randint(1, 5))
    
    # Add more varied moves
    for i in range(25):
        rolls.append(random.randint(1, 6))
    
    return rolls

def generate_balanced_rolls(game: SnakesLaddersGame) -> List[int]:
    """Generate balanced rolls with good mix of all die faces"""
    rolls = []
    
    # Ensure we have a good distribution
    for face in range(1, 7):
        count = random.randint(3, 8)
        for _ in range(count):
            rolls.append(face)
    
    # Shuffle for randomness
    random.shuffle(rolls)
    
    # Add some extra random rolls
    for _ in range(20):
        rolls.append(random.randint(1, 6))
    
    return rolls

def generate_power_die_rolls(game: SnakesLaddersGame) -> List[int]:
    """Generate rolls that utilize power die mechanics"""
    rolls = []
    
    # Start with regular moves
    for _ in range(8):
        rolls.append(random.randint(1, 5))
    
    # Power up sequence
    rolls.append(6)  # Power up
    
    # Use power die strategically
    power_sequence = [2, 3, 4, 1]  # Large moves then revert
    rolls.extend(power_sequence)
    
    # Continue with regular play
    for _ in range(30):
        if random.random() < 0.15:  # Occasional power ups
            rolls.append(6)
        elif random.random() < 0.1:  # Some small moves for control
            rolls.append(1)
        else:
            rolls.append(random.randint(2, 5))
    
    return rolls

def generate_random_rolls(game: SnakesLaddersGame, length: int) -> List[int]:
    """Generate completely random rolls of given length"""
    return [random.randint(1, 6) for _ in range(length)]

@slpu.route('/slpu', methods=['POST'])
def solve_snakes_ladders():
    try:
        # Get SVG content from request
        svg_content = request.data.decode('utf-8')
        
        # Parse the board
        board_width, board_height, jumps = parse_svg_board(svg_content)
        
        # Create game instance
        game = SnakesLaddersGame(board_width, board_height, jumps)
        
        # Generate optimal rolls
        solution_rolls = generate_strategic_rolls(game)
        
        # Create response SVG
        response_svg = f'<svg xmlns="http://www.w3.org/2000/svg"><text>{solution_rolls}</text></svg>'
        
        return Response(response_svg, mimetype='image/svg+xml')
        
    except Exception as e:
        # Fallback response with reasonable length
        fallback_rolls = ''.join([str(random.randint(1, 6)) for _ in range(60)])
        response_svg = f'<svg xmlns="http://www.w3.org/2000/svg"><text>{fallback_rolls}</text></svg>'
        return Response(response_svg, mimetype='image/svg+xml')