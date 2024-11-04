from flask import Blueprint
from ..utils import *
from flask_cors import cross_origin, CORS
import app.services.supabase as supabase
# from dailyDB import get_dailydb_data
from app.services.summarize import dialogue_processing
from app.services.audio import process_audio
# from app.services.sentiment_analysis import analyze

supabase_client = supabase()
bp = Blueprint('summarize', __name__)
CORS(bp)


#call the functions from controllers file
#to be updated
@bp.route("/url", methods=['GET','POST'])
@cross_origin
def interview_response_summarization():
    def text_summarization():
        try:
            #get data from DailyDB
            # dialogue = get_dailydb_data()
            dialogue = ""
            # Summarization
            summarized_dialogue = dialogue_processing(dialogue)
            result = supabase_client.insert_supabase_data({"summarized_dialogue": summarized_dialogue})
            
            # Return summarized data to superbase
            return handle_success(result)
        except Exception as e:
            return handle_server_error(e)


@bp.route("/url", methods=['GET','POST'])
@cross_origin
def interview_video_summarization():
    # analyze()
    return None


@bp.route("/url", methods=['GET','POST'])
@cross_origin
def interview_audio_summarization():
    process_audio()
    return None    