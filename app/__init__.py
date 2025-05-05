from flask import Flask
from flask_cors import CORS
from app.routes.resume_routes import resume_bp  # Import the resume routes
from app.routes.health import health_bp  # Import the health routes
from app.routes.evaluation import evaluation_bp
from app.routes.hume_callback import bp_hume

def create_app(Test=False) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = Test

    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register the blueprints
    app.register_blueprint(resume_bp)
    app.register_blueprint(health_bp)  # Register the health blueprint
    app.register_blueprint(evaluation_bp, url_prefix="/evaluation")
    app.register_blueprint(bp_hume, url_prefix="/hume-callback")

    return app

# This file can be empty, it just marks the directory as a Python package