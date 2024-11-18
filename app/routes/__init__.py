from flask import Flask
from app.routes.scores import bp as scores_bp
from app.routes.summarize import bp as summarize_bp
from app.routes.evaluation import bp as eval_bp

"""A function to register all the blueprints."""
def register_blueprints(app: Flask):
    app.register_blueprint(scores_bp, url_prefix='/scores')
    app.register_blueprint(summarize_bp, url_prefix='/summarize')
    app.register_blueprint(eval_bp, url_prefix='/eval')