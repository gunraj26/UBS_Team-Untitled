from flask import Flask

app = Flask(__name__)
import routes.square
import routes.trading_formula
import routes.trading_bot


# --- import blueprints ---
# from .trading_formula import trading_formula_bp

# # --- register blueprints ---
# app.register_blueprint(trading_formula_bp)    

# from .trading_bot import bp as trading_bot_bp  # noqa: F401

# __all__ = ["trading_bot_bp"]          


from routes.ink_archive import bp as ink_archive_bp
app.register_blueprint(ink_archive_bp)

from routes.slpu import slpu
app.register_blueprint(slpu)

from routes.mages_gambit import mages_gambit

app.register_blueprint(mages_gambit)
