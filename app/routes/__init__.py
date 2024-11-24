from flask import Flask
from .evaluation import bp as evaluation_bp

"""A function to register all the blueprints."""
def register_blueprints(app: Flask):
    app.register_blueprint(evaluation_bp, url_prefix='/evaluation')