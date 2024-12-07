from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import hmac
import uuid
import base64
import hashlib
from app.configs.daily_config import DAILY_API_KEY, DAILY_WEBHOOK_SECRET
from app.configs.merge_config import MERGE_API_KEY, MERGE_ACCOUNT_TOKEN
from ..utils import *
from app.services import score_calculation
from app.services import summarize
from app.services.sentiment import run_emotion_analysis
from app.services.score_calculation import response_eval
from app.controllers.supabase_db import SupabaseDB
from app.controllers.merge import MergeAPIClient
from app.controllers.dailybatchprocessor import DailyBatchProcessor
from app.controllers.recording_link import DailyVideoDownloader

bp = Blueprint('eval', __name__)
CORS(bp)

def verify_webhook_signature(timestamp, raw_body, signature):
    if not DAILY_WEBHOOK_SECRET:
        return True  # Skip verification if secret not set
        
    try:
        # Decode the base64 secret
        decoded_secret = base64.b64decode(DAILY_WEBHOOK_SECRET)
        
        # Create signature string
        signature_string = f"{timestamp}.{raw_body}"
        
        # Calculate HMAC
        computed_hmac = hmac.new(
            decoded_secret,
            signature_string.encode('utf-8'),
            hashlib.sha256
        )
        computed_signature = base64.b64encode(computed_hmac.digest()).decode()
        
        return hmac.compare_digest(computed_signature, signature)
    except Exception as e:
        print(f"Signature verification failed: {e}")
        return False


@bp.route("/webhook", methods=['POST'])
@cross_origin()
def handle_webhook():
    try:
        # Get raw body for signature verification
        raw_body = request.get_data(as_text=True)
        
        # Get headers for verification
        timestamp = request.headers.get('X-Webhook-Timestamp')
        signature = request.headers.get('X-Webhook-Signature')
                
        # Verify signature if present
        if timestamp and signature:
            if not verify_webhook_signature(timestamp, raw_body, signature):
                return jsonify({'error': 'Invalid signature'}), 401
        
        
        batch_processor = DailyBatchProcessor(DAILY_API_KEY)
        downloader = DailyVideoDownloader(DAILY_API_KEY)
        merge_client = MergeAPIClient(MERGE_ACCOUNT_TOKEN, MERGE_API_KEY)
        database = SupabaseDB()

        # Parse JSON data
        data = request.get_json()
        
        # Handle initial webhook verification
        if data.get('test') == 'test':
            return jsonify({'status': 'success'}), 200
        
        event_type = data.get('type')
        if event_type == 'recording.started':
            # Just acknowledge the start
            return jsonify({'status': 'recording started'}), 200
        
        # Handle actual webhook event
        elif event_type == 'recording.ready-to-download':
            # Extract the recording ID from the payload
            recording_id = data['payload']['recording_id']
            room_name = data['payload']['room_name']
            
            insert_recording_id = {
                'recording_id': recording_id
            }
            
            database.update_supabase_data("AI_summary", insert_recording_id, ['room_name', room_name])
            
            # Initialize the batch processor
            job_response = batch_processor.submit_batch_processor_job(recording_id)
            job_id = job_response["id"]
            
            return jsonify({'status': f'batch processor job started with id: {job_id}'}), 200
         
        elif event_type == 'batch-processor.job-finished':
            # Extract the job ID from the payload
            print('Batch processor job finished')
            job_id = data['payload']['id']
            
            text_raw = batch_processor.process_transcription_job(job_id)
            
            recording_id = data['payload']['input']['recordingId']
            # Process the transcription with the recording ID
            result = downloader.get_download_link(recording_id)
            
            media_urls = [result['download_link']]
            models = {
                "face": {},
                "language": {},
                "prosody": {}
            }
                        
            summary = batch_processor.process_summary_job(job_id)
            
             # insert merge information
            application_id = database.get_supabase_data("AI_summary", "application_id", ["recording_id", recording_id])
            merge_job_id = database.get_supabase_data("applications", "job_id", ["id", application_id])
            
            merge_job = merge_client.process_job_data(merge_job_id)
            merge_job_description = merge_job.data.get("name") + merge_job.data.get("description")
            
            data_to_insert = response_eval(text_raw, merge_job_description)

            initial_data = {
                "transcript_summary": summary,
                'text_eval': data_to_insert,
                'batch-processor_transcript_id': job_id
            }
            database.update_supabase_data("AI_summary", initial_data, ['recording_id', recording_id])

            # publicly accessible callback URL
            callback_url = f'https://roborecruiter-analysisbot-production.up.railway.app/hume-callback/hume?recording_id={recording_id}&job_description={merge_job_description}' 

            # start emotion analysis job with callback
            emotion_job_id = run_emotion_analysis(media_urls, text_raw, models, callback_url)
            
            # return handle_success(result)
            print(emotion_job_id)
            return jsonify({'status': f'Emotion analysis job started with ID: {emotion_job_id}'}), 200
        
        return jsonify({'error': 'Unsupported event type'}), 400
        
    except Exception as e:
        return handle_server_error(e)
