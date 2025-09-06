from flask import Blueprint, request, jsonify

mages_gambit = Blueprint('mages_gambit', __name__)

def solve_mages_gambit(intel, reserve, fronts, stamina):
    """
    Solve the mage's gambit problem to find minimum time to defeat all undead
    
    Args:
        intel: List of [front, mana_cost] pairs representing undead attacks
        reserve: Maximum mana capacity
        fronts: Number of fronts (not directly used in calculation)
        stamina: Maximum number of spells before requiring cooldown
    
    Returns:
        Minimum time in minutes to defeat all undead and be ready for expedition
    """
    if not intel:
        return 0
    
    current_mana = reserve
    current_stamina = stamina
    total_time = 0
    last_front = None
    
    for i, (front, mana_cost) in enumerate(intel):
        # Check if we need cooldown before this attack
        needs_cooldown = False
        
        # Need cooldown if not enough mana or no stamina left
        if current_mana < mana_cost or current_stamina == 0:
            needs_cooldown = True
        
        if needs_cooldown:
            # Perform cooldown
            total_time += 10
            current_mana = reserve
            current_stamina = stamina
        
        # Perform the attack
        current_mana -= mana_cost
        current_stamina -= 1
        
        # Add time for the attack (10 minutes unless extending AOE at same front)
        if last_front == front:
            # Extending AOE at same location - no extra time needed
            pass
        else:
            # New target location - takes 10 minutes
            total_time += 10
        
        last_front = front
    
    # After all attacks, Klein must be in cooldown state to join expedition
    # If he's not already in cooldown, he needs one more cooldown
    total_time += 10  # Final cooldown to be ready for expedition
    
    return total_time

@mages_gambit.route('/the-mages-gambit', methods=['POST'])
def the_mages_gambit():
    """
    POST endpoint for The Mage's Gambit challenge
    
    Expected input: List of scenarios with intel, reserve, fronts, stamina
    Returns: List of results with minimum time for each scenario
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a list of scenarios"}), 400
        
        results = []
        
        for scenario in data:
            # Validate required fields
            required_fields = ['intel', 'reserve', 'fronts', 'stamina']
            if not all(field in scenario for field in required_fields):
                return jsonify({"error": f"Missing required fields: {required_fields}"}), 400
            
            intel = scenario['intel']
            reserve = scenario['reserve']
            fronts = scenario['fronts']
            stamina = scenario['stamina']
            
            # Validate data types and ranges
            if not isinstance(intel, list):
                return jsonify({"error": "intel must be a list"}), 400
            
            if not all(isinstance(item, list) and len(item) == 2 for item in intel):
                return jsonify({"error": "Each intel item must be [front, mana_cost]"}), 400
            
            if not all(isinstance(reserve, int) and isinstance(fronts, int) and isinstance(stamina, int)):
                return jsonify({"error": "reserve, fronts, and stamina must be integers"}), 400
            
            if reserve <= 0 or fronts <= 0 or stamina <= 0:
                return jsonify({"error": "reserve, fronts, and stamina must be positive"}), 400
            
            # Validate intel ranges
            for front, mana_cost in intel:
                if not (1 <= front <= fronts):
                    return jsonify({"error": f"Front {front} out of range [1, {fronts}]"}), 400
                if not (1 <= mana_cost <= reserve):
                    return jsonify({"error": f"Mana cost {mana_cost} out of range [1, {reserve}]"}), 400
            
            # Solve the scenario
            time_needed = solve_mages_gambit(intel, reserve, fronts, stamina)
            results.append({"time": time_needed})
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500