from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
from ..utils import *
from app.services import score_calculation
from controllers.supabase_db import insert_supabase_data, get_supabase_data
from app.controllers.daily_db import get_dailydb_data


bp = Blueprint('scores', __name__)
CORS(bp)

@bp.route("/url", methods=['GET','POST'])
@cross_origin
def interview_evaluation():
    try:
        #get text_raw
        #get interview questions
        #get min_qual
        #get preferred_qual
        #get table to store

        #{"total":total,"text":lex,"audio":audio,"video":video}
        interview_eval = score_calculation.eval_result(text_raw, questions, min_qual, preferred_qual)

        #send evaluation
        result = insert_supabase_data(table,interview_eval)

        return jsonify({'result': result}), 200
    except Exception as e:
         return jsonify({'error': str(e)}), 500