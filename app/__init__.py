from flask import Flask
from flask_cors import CORS
from app.routes.resume_routes import resume_bp  # Import the resume routes
from app.routes.health import health_bp  # Import the health routes

def create_app(Test=False) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = Test

    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register the blueprints
    app.register_blueprint(resume_bp)
    app.register_blueprint(health_bp)  # Register the health blueprint

    return app

# This file can be empty, it just marks the directory as a Python package