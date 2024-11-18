from flask import Blueprint
from ..utils import *
from flask_cors import cross_origin, CORS
#import app.services.supabase as supabase
#from app.services.summarize import dialogue_processing
#from app.controllers.daily_db import get_dailydb_data
#from controllers.supabase_db import insert_supabase_data
# from app.services.sentiment_analysis import analyze

# supabase_client = supabase()
bp = Blueprint('summarize', __name__)
CORS(bp)



@bp.route("/url", methods=['GET','POST'])
@cross_origin
def interview_response_summarization():
    pass
"""
def interview_response_summarization():
    try:
        #get data from DailyDB
        dialogue = get_dailydb_data()
        # Summarization
        summarized_dialogue = dialogue_processing(dialogue)
        result = insert_supabase_data({"summarized_dialogue": summarized_dialogue})
            
        # Return summarized data to superbase
        return handle_success(result)
    except Exception as e:
        return handle_server_error(e)


@bp.route("/video", methods=['GET','POST'])
@cross_origin
def interview_video_summarization():
    # analyze()
    return None


@bp.route("/audio", methods=['GET','POST'])
@cross_origin
def interview_audio_summarization():
    #process_audio()
    return None    

"""