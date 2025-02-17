from flask import Flask
from flask_cors import CORS
from .routes import *

def create_app(Test=False) -> Flask:

    app = Flask(__name__)
    app.config["TESTING"] = Test

    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.route("/", methods=["GET"])
    def health_check():
        return "<p>Hello, World!</p>"
    
    routes.register_blueprints(app)
   

    return app

# This file can be empty, it just marks the directory as a Python package