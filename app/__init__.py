from flask import Flask
from flask_cors import CORS
from app.routes import register_blueprints

def create_app(Test=False) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = Test

    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register the blueprints
    # app.register_blueprint(resume_bp)
    # app.register_blueprint(health_bp)  # Register the health blueprint
    # app.register_blueprint(evaluation_bp, url_prefix="/evaluation")
    # app.register_blueprint(hume_callback_bp, url_prefix="/hume-callback")
    # app.register_blueprint(merge_bp, url_prefix="/merge")
    register_blueprints(app)


    return app

# This file can be empty, it just marks the directory as a Python package