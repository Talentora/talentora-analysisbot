from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import json
from hume import HumeClient

from app.configs.hume_config import HUME_API_KEY
from app.controllers.hume_job_manager import JobManager
from app.services.sentiment_analysis import EmotionAnalyzer
from app.controllers.supabase_db import SupabaseDB
from app.services.summarize import ai_summary

class HumeCallbackHandler:
    """Handles processing of Hume API callbacks and data management."""
    
    def __init__(self):
        """Initialize components needed for processing Hume callbacks."""
        self.client = HumeClient(api_key=HUME_API_KEY)
        self.job_manager = JobManager(self.client)
        self.emotion_analyzer = EmotionAnalyzer(HUME_API_KEY)
        self.database = SupabaseDB()

    def validate_request(self, data, recording_id):
        """Validate incoming request data."""
        job_id = data.get('job_id')
        if not job_id:
            raise ValueError('job_id not provided in callback')
        if not recording_id:
            raise ValueError('recording_id not provided in query parameters')
        return job_id

    def get_existing_data(self, recording_id):
        """Retrieve existing data from database."""
        transcript_data = self.database.get_supabase_data(
            "AI_summary",
            "transcript_summary",
            ["recording_id", recording_id]
        ).data[0]['transcript_summary']

        text_eval_data = self.database.get_supabase_data(
            "AI_summary",
            "text_eval",
            ["recording_id", recording_id]
        ).data[0]['text_eval']

        return transcript_data, text_eval_data

    def process_emotions(self, job_id):
        """Process emotion predictions from Hume."""
        predictions = self.job_manager.get_job_predictions(job_id)
        if not predictions:
            raise ValueError(f"No predictions found for job {job_id}")
        
        return self.emotion_analyzer.process_predictions(predictions)

    def generate_summary(self, transcript_summary, text_eval, job_description, emotion_results):
        """Generate overall summary from all available data."""
        return json.loads(ai_summary(
            transcript_summary,
            text_eval,
            job_description,
            emotion_results
        ))

    def update_database(self, recording_id, emotion_results, summary):
        """Update database with processed results."""
        update_data = {
            "emotion_eval": emotion_results,
            "overall_summary": summary
        }
        self.database.update_supabase_data(
            "AI_summary",
            update_data,
            ["recording_id", recording_id]
        )

    def process_callback(self, data, recording_id, job_description):
        """Process the complete Hume callback workflow."""
        # Validate request
        job_id = self.validate_request(data, recording_id)

        # Get existing data
        transcript_summary, text_eval = self.get_existing_data(recording_id)

        # Process emotions
        emotion_results = self.process_emotions(job_id)

        # Generate summary
        summary = self.generate_summary(
            transcript_summary,
            text_eval,
            job_description,
            emotion_results
        )

        # Update database
        self.update_database(recording_id, emotion_results, summary)

# Blueprint setup
bp_hume = Blueprint('hume_callback', __name__)
CORS(bp_hume)
callback_handler = HumeCallbackHandler()

@bp_hume.route('/hume', methods=['POST'])
@cross_origin()
def hume_callback():
    """Handle incoming Hume API callbacks."""
    try:
        data = request.get_json()
        recording_id = request.args.get('recording_id')
        job_description = request.args.get('job_description')

        callback_handler.process_callback(data, recording_id, job_description)
        return jsonify({'status': 'success'}), 200

    except ValueError as e:
        print(f"Validation error in Hume callback: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error processing Hume callback: {e}")
        return jsonify({'error': str(e)}), 500