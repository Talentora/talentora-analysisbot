from flask import Flask
from flask_cors import CORS
from app.controllers.functions import text_summarization, audio_summarization, video_summarization, interview_score_calculation

def create_app(Test=False) -> Flask:

    app = Flask(__name__)
    app.config["TESTING"] = Test

    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.route("/")
    def hello_world():
        return "<p>Hello, World!</p>"

    @app.route("/url", methods=['GET','POST'])
    def interview_score():
        interview_score_calculation()
        pass

    #call the functions from controllers file
    #to be updated
    @app.route("/url", methods=['GET','POST'])
    def interview_response_summarization():
        text_summarization()
        pass

    @app.route("/url", methods=['GET','POST'])
    def interview_video_summarization():
        video_summarization()
        pass
    @app.route("/url", methods=['GET','POST'])
    def interview_audio_summarization():
        audio_summarization()
        pass
        

    return app