from flask import Flask
from .scores import bp as scores_bp

"""A function to register all the blueprints."""
def register_blueprints(app: Flask):
    app.register_blueprint(scores_bp, url_prefix='/scores')