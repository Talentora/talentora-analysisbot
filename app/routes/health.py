from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__)

@health_bp.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "Healthy", "message": "Hello, World!"}), 200
