import os
from flask import Flask, request, jsonify

# fix package imports
from video import process_video
from summarize import dialogue_processing
from audio import process_audio



def hello_world():
    return "<p>Hello, World!</p>"


def text_summarization():
    try:
        #get data from DailyDB
        data = request.get_json()
        if not data or 'dialogue' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        dialogue = data['dialogue']
        
        # Summarization
        summarized_dialogue = dialogue_processing(dialogue)
        
        # Return summarized data to superbase
        return jsonify({'summarized_dialogue': summarized_dialogue}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def video_summarization():
    process_video()
    return None

def audio_summarization():
    process_audio()
    return None
