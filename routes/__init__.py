from flask import Flask

app = Flask(__name__)
import routes.square
import routes.ticketing_agent
import routes.operation_safeguard
import routes.investigate
import routes.blankety
import routes.princess
import routes.fog_of_wall
import routes.trading_formula

# --- import blueprints ---
from .trading_formula import trading_formula_bp
# (if you have these, import them as well)
# from .square import square_bp
# from .investigate import investigate_bp

# --- register blueprints ---
app.register_blueprint(trading_formula_bp)        # exposes POST /trading-formula        
# app.register_blueprint(square_bp)
# app.register_blueprint(investigate_bp)

