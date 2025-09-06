import logging
import socket

from routes import app


logger = logging.getLogger(__name__)


@app.route('/', methods=['GET'])
def default_route():
    return 'Python Template'

# app.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/trivia")
def trivia():
    return jsonify({
        "answers": [3, 1, 2, 2, 3, 4, 4, 5, 4, 3, 1, 2, 4, 2, 2, 2, 1, 2, 2, 2, 4, 2, 2, 3, 2]
    })

@app.post("/micro-mouse")
def micro_mouse():
    return jsonify(
        {
  "instructions": ["F2", "F2", "BB"],
  "end": False
})



logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logging.info("Starting application ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 8080))
    port = sock.getsockname()[1]
    sock.close()
    app.run(port=port)
