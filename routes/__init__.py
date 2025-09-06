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
import routes.mstCalc
import routes.duolingo

# --- import blueprints ---
# from .trading_formula import trading_formula_bp

# # --- register blueprints ---
# app.register_blueprint(trading_formula_bp)              



from routes.ink_archive import bp as ink_archive_bp
app.register_blueprint(ink_archive_bp)
