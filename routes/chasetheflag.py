from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/chasetheflag", methods=["POST"])
def chasetheflag():
    return jsonify({
        "challenge1": "your_flag_1",
        "challenge2": "your_flag_2",
        "challenge3": "your_flag_3",
        "challenge4": "your_flag_4",
        "challenge5": "your_flag_5"
    })