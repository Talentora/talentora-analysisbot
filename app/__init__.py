from flask import Flask
from flask_cors import CORS
from .routes import *

def create_app(Test=False) -> Flask:

    app = Flask(__name__)
    app.config["TESTING"] = Test

    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.route("/")
    def health_check():
        return "<p>Hello, World!</p>"

    routes.register_blueprints(app)

    return app