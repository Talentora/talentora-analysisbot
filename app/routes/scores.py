from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
from ..utils import *
import uuid
from app.services import score_calculation
from app.controllers.supabase_db import insert_supabase_data, get_supabase_data, update_supabase_data
# from app.controllers.daily_db import get_dailydb_data
from app.services.summarize import dialogue_processing


bp = Blueprint('scores', __name__)
CORS(bp)

@bp.route("/interview", methods=['GET'])
@cross_origin()
def interview_evaluation():
    try:
        # daily_data = get_dailydb_data()
        # text_raw = daily_data["text"]
        # job_id = daily_data["job_id"]
        # applicat_id = daily_data["applicant_id"]

        # get_conditions = ["job_id",job_id]
        # questions = get_supabase_data("job_interview_config","interview_questions",get_conditions)
        # min_qual = get_supabase_data("job_interview_config","column_name",get_conditions)
        # preferred_qual = get_supabase_data("job_interview_config","column_name",get_conditions)

        # #{"total":total,"text":lex,"audio":audio,"video":video}
        # interview_eval = score_calculation.eval_result(text_raw, questions, min_qual, preferred_qual)
        # emotion_eval = {} #vincent
        # interview_summary = dialogue_processing(text_raw)

        # #send evaluation
        # ai_summary_id = uuid.uuid4()
        # data_send = {'id':ai_summary_id,'text_eval':interview_eval,'emotion_eval':emotion_eval,'interview_summary':interview_summary}
        # insert_supabase_data("AI_summary",data_send)

        # #update
        # update_conditions = ["applicat_id",applicat_id,'job_id',job_id]
        # data_update = {"AI_summary":ai_summary_id}
        # result = update_supabase_data("application",data_update,update_conditions)

        # return handle_success(result)
        return handle_success("not implemented")
    except Exception as e:
        return handle_server_error(e)
