from flask import Flask, request, jsonify

# fix package imports
from services.video import process_video
from services.summarize import dialogue_processing
from services.audio import process_audio
from services.score_calculation import score
from supabase import insert_supabase_data
from dailyDB import get_dailydb_data

def interview_score():
    try:
        interview_score = score() #interview result. would call function from services file
        result = insert_supabase_data({"interview_score": interview_score})

        return jsonify({'result': result}), 200
    
    except Exception as e:
         return jsonify({'error': str(e)}), 500

    return None

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
