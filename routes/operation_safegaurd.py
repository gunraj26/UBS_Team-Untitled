from flask import Flask, jsonify

from routes import app
@app.route("/operation-safeguard", methods=["POST"])
def operation_safeguard():
    response = {
        "challenge_one": "YkwgmzE mzxd",
        "challenge_two": "73",
        "challenge_three": "DATAVAULT",
        "challenge_four": "Threat group: SHADOW. Objective: Data Vault 73 at Meridian International Bank."
    }
    return jsonify(response), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
