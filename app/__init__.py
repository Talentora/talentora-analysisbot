from flask import Flask, request
from flask_cors import CORS
from app.routes import register_blueprints
import logging

def create_app(Test=False) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = Test

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Add request logging middleware
    @app.before_request
    def log_request_info():
        logging.info(f'Request: {request.method} {request.url}')

    @app.after_request
    def log_response_info(response):
        logging.info(f'Response: {response.status}')
        return response

    # Register the blueprints
    # app.register_blueprint(resume_bp)
    # app.register_blueprint(health_bp)  # Register the health blueprint
    # app.register_blueprint(evaluation_bp, url_prefix="/evaluation")
    # app.register_blueprint(hume_callback_bp, url_prefix="/hume-callback")
    # app.register_blueprint(merge_bp, url_prefix="/merge")
    register_blueprints(app)

    return app

# This file can be empty, it just marks the directory as a Python package