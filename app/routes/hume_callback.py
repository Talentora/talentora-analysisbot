from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
from hume import HumeClient
from app.configs.hume_config import HUME_API_KEY
from app.controllers.hume_job_manager import JobManager
from app.services.sentiment_analysis import EmotionAnalyzer

bp_hume = Blueprint('hume_callback', __name__)
CORS(bp_hume)

@bp_hume.route('/hume', methods=['POST'])
@cross_origin()
def hume_callback():
    try:
        data = request.get_json()
        print(f"Received Hume callback data: {data}")

        # Extract job_id from the callback data
        job_id = data.get('job_id')
        if not job_id:
            return jsonify({'error': 'job_id not provided in callback'}), 400

        # Initialize components
        client = HumeClient(api_key=HUME_API_KEY)
        job_manager = JobManager(client)
        emotion_analyzer = EmotionAnalyzer(HUME_API_KEY)

        # Get predictions
        predictions = job_manager.get_job_predictions(job_id)
        if not predictions:
            print(f"No predictions found for job {job_id}")
            return jsonify({'status': 'no predictions'}), 200

        # Process predictions
        results = emotion_analyzer.process_predictions(predictions)

        # Save or process the results as needed
        # save_emotion_analysis_results(job_id, results)  # Implement this function
        print(results)

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        print(f"Error processing Hume callback: {e}")
        return jsonify({'error': str(e)}), 500

def save_emotion_analysis_results(job_id, results):
    # Implement your logic to save results to the database or perform other actions
    pass