from flask import Blueprint
from ..utils import *
from flask_cors import cross_origin, CORS
from app.services import *
from app.controllers.functions import text_summarization, audio_summarization, video_summarization
import app.services.supabase as supabase

supabase_client = supabase()
bp = Blueprint('summarize', __name__)
CORS(bp)


#call the functions from controllers file
#to be updated
@app.route("/url", methods=['GET','POST'])
@cross_origin
def interview_response_summarization():
    def text_summarization():
        return None
        try:
            #get data from DailyDB
            dialogue = get_dailydb_data()
            
            # Summarization
            summarized_dialogue = dialogue_processing(dialogue)
            result = supabase_client.insert_supabase_data({"summarized_dialogue": summarized_dialogue})
            
            # Return summarized data to superbase
            return handle_success(result)
        except Exception as e:
            return handle_server_error(e)


@app.route("/url", methods=['GET','POST'])
@cross_origin
def interview_video_summarization():
    video_summarization()
    pass


@app.route("/url", methods=['GET','POST'])
@cross_origin
def interview_audio_summarization():
    audio_summarization()
    pass
    