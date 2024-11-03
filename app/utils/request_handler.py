from flask import jsonify

# Handles API requests/returns


def handle_success(data:str, addition = ""):
    """handle success requests"""
    if(addition != ""):
        return jsonify({'message': data, 'data': addition}), 200
    return jsonify({'message': data}), 200


def handle_bad_request(message:str):
    """handle bad requests"""
    return jsonify({'message': message}), 400


def handle_server_error(err:str):
    """handle server errors"""
    return jsonify({'message': str(err)}), 500


def handle_not_found():
    """handle not found error"""
    return jsonify({'message': "not found"}), 404