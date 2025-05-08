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
    register_blueprints(app)

    return app

# This file can be empty, it just marks the directory as a Python package