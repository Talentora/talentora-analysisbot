from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import os
import hmac
import base64
import hashlib
from dotenv import load_dotenv
from ..utils import *
from app.services import score_calculation
from app.controllers.supabase_db import insert_supabase_data, get_supabase_data
# from app.controllers.daily_db import get_dailydb_data

from app.controllers.dailybatchprocessor import DailyBatchProcessor, process_transcription_job

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
        
        # Parse JSON data
        data = request.get_json()
        
        # Handle initial webhook verification
        if data.get('test') == 'test':
            return jsonify({'status': 'success'}), 200
            
        # Handle actual webhook event
        if data.get('type') == 'recording.ready-to-download':
            # Extract the recording ID from the payload
            recording_id = data['payload']['recording_id']
            
            # Initialize the batch processor
            batch_processor = DailyBatchProcessor(api_key)
            
            # Process the transcription with the recording ID
            text_raw = process_transcription_job(batch_processor, recording_id)
            
            # Get necessary data from Supabase
            questions = get_supabase_data()
            min_qual = get_supabase_data()
            preferred_qual = get_supabase_data()
            table = get_supabase_data()

            # Calculate interview evaluation
            interview_eval = score_calculation.eval_result(text_raw, questions, min_qual, preferred_qual)

            # Send evaluation to Supabase
            result = insert_supabase_data(table, interview_eval)
            
            return handle_success(result)
        
        return jsonify({'error': 'Unsupported event type'}), 400
        
    except Exception as e:
        return handle_server_error(e)

