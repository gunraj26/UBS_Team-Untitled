from flask import Flask, jsonify, request

from routes import app

@app.route("/trivia", methods=["POST"])
def chasetheflag():
    return jsonify({
       "answers": [
        2,
        1,
        2,
        2,
        3,
        4,
        2,
        5,
        4, 
      ]
    })