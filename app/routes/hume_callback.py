import os
from pathlib import Path
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import json
from hume import HumeClient

from app.configs.hume_config import HUME_API_KEY
from app.controllers.hume_job_manager import JobManager
from app.controllers.supabase_db import SupabaseDB
from app.services.summarize import ai_summary
from app.services.mmr.data_preprocessor import DataPreprocessor
from app.services.mmr.multi_modal_regressor import MultiModalRegressor

BASE_DIR = Path(__file__).resolve().parent.parent  # Gets the app directory
MODEL_PATH = os.path.join(BASE_DIR, "services", "mmr", "mmr_model.pkl")
class HumeCallbackHandler:
    """Handles processing of Hume API callbacks and data management."""
    
    def __init__(self):
        """Initialize components needed for processing Hume callbacks."""
        self.client = HumeClient(api_key=HUME_API_KEY)
        self. job_manager = JobManager(self.client)
        # self.emotion_analyzer = EmotionAnalyzer(HUME_API_KEY)
        self.database = SupabaseDB()
        self.mmr_preprocessor = DataPreprocessor()
        self.mmr = MultiModalRegressor.load_model(MODEL_PATH)

    
    def validate_request(self, data, recording_id):
        """Validate incoming request data."""
        print("[DEBUG] validate_request called")
        job_id = data.get('job_id')
        if not job_id:
            raise ValueError('job_id not provided in callback')
        if not recording_id:
            raise ValueError('recording_id not provided in query parameters')
        return job_id

    def process_emotions(self, job_id):
        """Process emotion predictions from Hume."""
        #NOTE: This is where the MMR model is used to process the predictions
        print(f"[DEBUG] process_emotions called for job_id={job_id}")
        predictions = self.job_manager.get_job_predictions(job_id) 
        if not predictions:
            raise ValueError(f"No predictions found for job {job_id}")
        print("[DEBUG] Predictions retrieved from Hume")
        
        
        # Pre-process the Hume data, generate the aggregates & timelines, and prepare the dataframes for the SVR models
        json_resp = self.mmr_preprocessor.process_single_prediction_from_obj(predictions)
        prepared_input = self.mmr_preprocessor.prepare_single_prediction(self.mmr_preprocessor.get_model_input_dataframes())
        
        # Get predictions from individual SVR models and meta model        
        overall_score, svr_predictions = self.mmr.predict(new_data=prepared_input)
        
        # feed predictions into json_resp to complete the final results
        final_results = self.mmr_preprocessor.update_prediction_results(json_resp, overall_score, svr_predictions)
        
        return final_results


    async def process_callback(self, recording_url):
        """Process the complete Hume callback workflow."""
        print("[DEBUG] process_callback called")
        #Prep, upload and wait for Hume job to complete
        models = {
            "face":"",
            "prosody": {"granularity": "conversational_turn"},
            "language": {"granularity": "conversational_turn"},
        }
        
        hume_job_id = self.job_manager.start_job(urls = [recording_url], models = models)
        await self.job_manager.wait_for_job_completion(hume_job_id)
        # Process emotions from the Hume job
        emotion_results = self.process_emotions(hume_job_id)
        print(f"[DEBUG] emotion_results: {emotion_results}")
        return emotion_results


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
        print(f"[DEBUG] request data: {data}")
        recording_id = request.args.get('recording_id')
        job_description = request.args.get('job_description')
        print(f"[DEBUG] query params => recording_id={recording_id}, job_description={job_description}")

        callback_handler.process_callback(data, recording_id, job_description)
        return jsonify({'status': 'success'}), 200

    except ValueError as e:
        print(f"Validation error in Hume callback: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error processing Hume callback: {e}")
        return jsonify({'error': str(e)}), 500