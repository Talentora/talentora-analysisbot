from flask import Flask
from .evaluation import bp as evaluation_bp
from .hume_callback import bp_hume as hume_callback_bp
from app.utils import handle_success

"""A function to register all the blueprints."""
def register_blueprints(app: Flask):
    app.register_blueprint(evaluation_bp, url_prefix='/evaluation')
    app.register_blueprint(hume_callback_bp, url_prefix='/hume-callback')
    
    @app.route('/', methods=['GET'])
    def heartbeat():
        return handle_success("It's running")