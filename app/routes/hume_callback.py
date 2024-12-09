import json
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
from hume import HumeClient
from app.configs.hume_config import HUME_API_KEY
from app.controllers.hume_job_manager import JobManager
from app.services.sentiment_analysis import EmotionAnalyzer
from app.controllers.supabase_db import SupabaseDB
from app.services.summarize import ai_summary

bp_hume = Blueprint('hume_callback', __name__)
CORS(bp_hume)

@bp_hume.route('/hume', methods=['POST'])
@cross_origin()
def hume_callback():
    try:
        data = request.get_json()

        # Extract job_id from the callback data
        job_id = data.get('job_id')
        recording_id = request.args.get('recording_id')
        if not job_id:
            return jsonify({'error': 'job_id not provided in callback'}), 400

        # Initialize components
        client = HumeClient(api_key=HUME_API_KEY)
        job_manager = JobManager(client)
        emotion_analyzer = EmotionAnalyzer(HUME_API_KEY)
        database = SupabaseDB()

        # Get predictions
        predictions = job_manager.get_job_predictions(job_id)
        if not predictions:
            print(f"No predictions found for job {job_id}")
            return jsonify({'status': 'no predictions'}), 200

        # Process predictions
        emotion_results = emotion_analyzer.process_predictions(predictions)
        transcript_summary = database.get_supabase_data("AI_summary", "transcript_summary", ["recording_id", recording_id]).data[0]['transcript_summary']
        text_eval = database.get_supabase_data("AI_summary", "text_eval", ["recording_id", recording_id]).data[0]['text_eval']
        job_description = request.args.get('job_description')
        summary = json.loads(ai_summary(transcript_summary, text_eval, job_description, emotion_results))
        
        update_data = {
            "emotion_eval": emotion_results,
            "overall_summary": summary
        }
        
        # Update the existing record using the analysis_id
        condition = ["recording_id", recording_id]
        database.update_supabase_data("AI_summary", update_data, condition)

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        print(f"Error processing Hume callback: {e}")
        return jsonify({'error': str(e)}), 500

def save_emotion_analysis_results(job_id, results):
    # Implement your logic to save results to the database or perform other actions
    pass