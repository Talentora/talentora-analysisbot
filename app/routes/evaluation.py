from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import os
from dotenv import load_dotenv
from ..utils import *
from app.services import score_calculation
from app.controllers.supabase_db import insert_supabase_data, get_supabase_data
# from app.controllers.daily_db import get_dailydb_data

from app.controllers.dailybatchprocessor import DailyBatchProcessor, process_transcription_job

load_dotenv()

api_key=os.environ.get("DAILY_API_KEY")
if not api_key:
    # raise EnvironmentError("DAILY_API_KEY environment variable is not set!")
    api_key = "none"

bp = Blueprint('eval', __name__)
CORS(bp)

@bp.route("/url", methods=['GET','POST'])
@cross_origin()
def interview_evaluation():
    try:
        batch_processor = DailyBatchProcessor(api_key)
        text_raw = process_transcription_job(batch_processor, "recording_id")
        
        questions = get_supabase_data()
        min_qual = get_supabase_data()
        preferred_qual = get_supabase_data()
        table = get_supabase_data()

        #{"total":total,"text":lex,"audio":audio,"video":video}
        interview_eval = score_calculation.eval_result(text_raw, questions, min_qual, preferred_qual)

        #send evaluation
        result = insert_supabase_data(table,interview_eval)

        return handle_success(result)
    except Exception as e:
         return handle_server_error(e)