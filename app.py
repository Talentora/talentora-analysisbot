import os
from flask import Flask, request, jsonify
from services.summarize import dialogue_processing


'''
Current implementation: After interview concludes, the interviee will be taken to a thank you page. 
This page URL will be the trigger urls to activate the AI model processing functions
'''



app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

#call the functions from controllers file
#to be updated
@app.route("/url", methods=['GET','POST'])
def text_summarization():
    pass

@app.route("/url", methods=['GET','POST'])
def video_summarization():
    pass
@app.route("/url", methods=['GET','POST'])
def audio_summarization():
    pass

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    
#fly.io https://fly.io/docs/launch/deploy/

"""
Deployment Backend
https://railway.app/
"""