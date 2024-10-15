import os
from flask import Flask, request, jsonify
from summarize import dialogue_processing

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/url", methods=['GET','POST'])
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


@app.route("/url", methods=['GET','POST'])
def video_summarization():

    return None

@app.route("/url", methods=['GET','POST'])
def audio_summarization():
    
    return None

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    
#fly.io https://fly.io/docs/launch/deploy/

"""
Deployment Backend
https://railway.app/
"""