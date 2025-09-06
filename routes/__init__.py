from flask import Flask

app = Flask(__name__)
import routes.square
import routes.ticketing_agent
import routes.operation_safeguard
import routes.investigate
import routes.blankety
import routes.princess
import routes.fog_of_wall
