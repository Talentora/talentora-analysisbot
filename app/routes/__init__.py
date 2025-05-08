from flask import Flask
from .merge_routes import merge_bp as merge_bp
from .evaluation import evaluation_bp
from .hume_callback import bp_hume
from .health import health_bp

from app.utils import handle_success

"""A function to register all the blueprints."""
def register_blueprints(app: Flask):
    app.register_blueprint(evaluation_bp, url_prefix='/evaluation')
    app.register_blueprint(merge_bp, url_prefix='/merge')
    app.register_blueprint(bp_hume, url_prefix='/hume-callback')
    app.register_blueprint(health_bp, url_prefix='/health')

    @app.route('/', methods=['GET'])
    def heartbeat():
        return handle_success("It's running")