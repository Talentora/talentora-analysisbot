from flask import Blueprint, request, jsonify
from ..utils import *
from flask_cors import cross_origin, CORS
from app.services import *


bp = Blueprint('scores', __name__)
CORS(bp)