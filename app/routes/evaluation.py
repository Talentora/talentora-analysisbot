from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
from ..utils import *
from app.services import score_calculation
from controllers.supabase_db import insert_supabase_data, get_supabase_data, update_supabase_data
from app.controllers.daily_db import get_dailydb_data


bp = Blueprint('scores', __name__)
CORS(bp)

@bp.route("/url", methods=['GET','POST'])
@cross_origin
def interview_evaluation():
    try:
        text_raw = get_dailydb_data()
        #job_interview_config table interview_questions column
        job_id = ""
        applicant_id = ""

        get_conditions = ["job_id",job_id]
        questions = get_supabase_data("job_interview_config","interview_questions",get_conditions)
        #job_interview_config table 
        min_qual = get_supabase_data("job_interview_config","column_name",get_conditions)
        #job_interview_config table 
        preferred_qual = get_supabase_data("job_interview_config","column_name",get_conditions)

        #{"total":total,"text":lex,"audio":audio,"video":video}
        interview_eval = score_calculation.eval_result(text_raw, questions, min_qual, preferred_qual)

        #send evaluation
        update_conditions = ["applicant_id",applicant_id]
        data = {'interview_eval':interview_eval}
        result = update_supabase_data("applicants",data, update_conditions)

        return handle_success(result)
    except Exception as e:
        return handle_server_error(e)