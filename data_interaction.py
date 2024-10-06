import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/idk", methods=['POST'])
def summarized_data():
    return "<p>Hello, World!</p>"