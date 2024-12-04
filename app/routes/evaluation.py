from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import os
import hmac
import base64
import hashlib
import uuid
from dotenv import load_dotenv
from ..utils import *
from app.services import score_calculation
from app.services import summarize
from app.services.sentiment import run_emotion_analysis
from app.controllers.supabase_db import SupabaseDB
from app.controllers.dailybatchprocessor import DailyBatchProcessor, process_transcription_job
from app.controllers.recording_link import DailyVideoDownloader

load_dotenv()

api_key=os.environ.get("DAILY_API_KEY")
webhook_secret = os.environ.get("DAILY_WEBHOOK_SECRET")

if not api_key:
    # raise EnvironmentError("DAILY_API_KEY environment variable is not set!")
    api_key = "none"

bp = Blueprint('eval', __name__)
CORS(bp)

def verify_webhook_signature(timestamp, raw_body, signature):
    if not webhook_secret:
        return True  # Skip verification if secret not set
        
    try:
        # Decode the base64 secret
        decoded_secret = base64.b64decode(webhook_secret)
        
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
        
        
        batch_processor = DailyBatchProcessor(api_key)
        downloader = DailyVideoDownloader(api_key)

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
            
            # Initialize the batch processor
            job_response = batch_processor.submit_transcript_job(recording_id)
            job_id = job_response["id"]
            
            return jsonify({'status': f'batch processor job started with id: {job_id}'}), 200
         
        elif event_type == 'batch-processor.job-finished':
            # Extract the job ID from the payload
            print('Batch processor job finished')
            job_id = data['payload']['id']
            print(job_id)
            text_raw = process_transcription_job(batch_processor, job_id)
            print(text_raw)
            
            recording_id = "cf6bcc01-14ac-48d5-9473-bbc516522e1c"
            
            # Process the transcription with the recording ID
            result = downloader.get_download_link(recording_id)
            print(result['download_link'])
            supabase_condition = ["id",recording_id]
            # job_id = SupabaseDB.get_supabase_data("applications","job_id",supabase_condition)
                                    
            # Get necessary data from Supabase
            # job_condition = ["job_id",job_id]
            # questions = SupabaseDB.get_supabase_data("job_interview_config","interview_questions",job_condition)
            # min_qual = SupabaseDB.get_supabase_data("job_interview_config","min_qual",job_condition)
            # preferred_qual = SupabaseDB.get_supabase_data("job_interview_config","preferred_qual",job_condition)
            # media_url = SupabaseDB.get_supabase_data("job_interview_config","preferred_qual",job_condition)
            
            media_urls = [result['download_link']]
            models = {
                "face": {},
                "language": {},
                "prosody": {}
            }

            # Calculate interview evaluation
            # evaluation_id = str(uuid())
            # text_eval = score_calculation.eval_result(text_raw, questions, min_qual, preferred_qual)
            emotion_eval = run_emotion_analysis(media_urls, text_raw, models)
            # interview_summary = summarize.dialogue_processing(text_raw, questions)

            # Send evaluation to Supabase
            # data_to_insert = {"id":evaluation_id,"text_eval":text_eval,"emotion_eval":emotion_eval,"interview_summary":interview_summary}
            # result = SupabaseDB.insert_supabase_data("AI_summary", data_to_insert)

            #update
            # update = {"AI_Summary":evaluation_id}
            # result = SupabaseDB.update_supabase_data("applications", update, supabase_condition)
            
            # return handle_success(result)
            print(emotion_eval)
            return jsonify({'status': f'batch processor job finished with id: {job_id}'}), 200
        
        return jsonify({'error': 'Unsupported event type'}), 400
        
    except Exception as e:
        return handle_server_error(e)

