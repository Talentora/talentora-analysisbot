from flask import Flask, request, jsonify

from services.video import process_video
from services.summarize import dialogue_processing
from services.audio import process_audio
from services.evaluation_calculation import evaluation
from supabase_db import insert_supabase_data
from dailyDB import get_dailydb_data

def interview_evaluation():
    try:
        interview_eval = evaluation() #interview result. would call function from services file
        table = ""
        result = insert_supabase_data(table,interview_eval)

        return jsonify({'result': result}), 200
    
    except Exception as e:
         return jsonify({'error': str(e)}), 500


def text_summarization():
    try:
        #get data from DailyDB
        dialogue = get_dailydb_data()
        
        # Summarization
        summarized_dialogue = dialogue_processing(dialogue)
        result = insert_supabase_data({"summarized_dialogue": summarized_dialogue})
        
        # Return summarized data to superbase
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def video_summarization():
    process_video()
    return None

def audio_summarization():
    process_audio()
    return None