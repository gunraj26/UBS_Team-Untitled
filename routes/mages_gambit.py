from flask import Flask, jsonify, request
from typing import Any, Dict, List, Tuple

def compute_time_for_scenario(intel: List[Tuple[int, int]], reserve: int, stamina: int) -> int:
    """
    Compute the minimal time (in minutes) for Klein to clear all undead waves in a scenario.
    Follows the rules specified in the challenge description.
    """
    current_mana = reserve
    current_stamina = stamina
    total_time = 0
    last_front = None
    last_action_was_attack = False

    for entry in intel:
        front, cost = entry
        # Cool down if resources insufficient
        if current_mana < cost or current_stamina < 1:
            total_time += 10  # cooldown time
            current_mana = reserve
            current_stamina = stamina
            last_action_was_attack = False
            last_front = None

        # Time to cast depends on whether the front is the same as the last one
        if last_action_was_attack and last_front == front:
            total_time += 0  # extension spell
        else:
            total_time += 10  # new target

        current_mana -= cost
        current_stamina -= 1
        last_action_was_attack = True
        last_front = front

    # Add final cooldown if there were any spells cast
    if intel:
        total_time += 10
    return total_time

app = Flask(__name__)

@app.route("/the-mages-gambit", methods=["POST"])
def the_mages_gambit() -> Any:
    """
    Handle POST requests containing an array of scenarios. Each scenario must include
    'intel', 'reserve', 'fronts', and 'stamina'. Returns a JSON array of objects with 'time'.
    """
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Request body must be a JSON array of scenarios."}), 400

    results: List[Dict[str, int]] = []
    for idx, scenario in enumerate(data):
        required = {"intel", "reserve", "fronts", "stamina"}
        if not required.issubset(scenario.keys()):
            missing = required.difference(scenario.keys())
            return jsonify({"error": f"Scenario at index {idx} is missing keys: {', '.join(sorted(missing))}."}), 400

        intel = scenario["intel"]
        reserve = int(scenario["reserve"])
        stamina = int(scenario["stamina"])

        # Compute and append the time
        time_required = compute_time_for_scenario(intel, reserve, stamina)
        results.append({"time": time_required})

    return jsonify(results), 200

