from flask import Flask, request, jsonify

# from app.services.sentiment_analysis import analyze
from app.services.summarize import dialogue_processing
from app.services.audio import process_audio
# from app.services.evaluation_calculation import evaluation

import app.services.supabase as supabase
from dailyDB import get_dailydb_data


supabase_client = supabase()

def interview_evaluation():
    pass
    # try:
    #     interview_eval = evaluation() #interview result. would call function from services file
    #     table = ""
    #     result = insert_supabase_data(table,interview_eval)

    #     return jsonify({'result': result}), 200
    
    # except Exception as e:
    #      return jsonify({'error': str(e)}), 500


def text_summarization():
    try:
        #get data from DailyDB
        dialogue = get_dailydb_data()
        
        # Summarization
        summarized_dialogue = dialogue_processing(dialogue)
        result = supabase_client.insert_supabase_data({"summarized_dialogue": summarized_dialogue})
        
        # Return summarized data to superbase
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def video_summarization():
    pass
    # analyze()
    return None

def audio_summarization():
    process_audio()
    return None
