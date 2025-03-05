from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import hmac
import uuid
import base64
import hashlib

from app.configs.daily_config import DAILY_API_KEY, DAILY_WEBHOOK_SECRET
from app.configs.merge_config import MERGE_API_KEY, MERGE_ACCOUNT_TOKEN
from app.services import score_calculation, summarize
from app.services.sentiment import run_emotion_analysis
from app.services.response_analysis import evaluate_candidate
from app.controllers.supabase_db import SupabaseDB
from app.controllers.merge import MergeAPIClient
from app.controllers.dailybatchprocessor import DailyBatchProcessor
from app.controllers.recording_link import DailyVideoDownloader
from app.utils import handle_server_error


class WebhookHandler:
    def __init__(self):
        self.batch_processor = DailyBatchProcessor(DAILY_API_KEY)
        self.downloader = DailyVideoDownloader(DAILY_API_KEY)
        self.merge_client = MergeAPIClient(MERGE_ACCOUNT_TOKEN, MERGE_API_KEY)
        self.database = SupabaseDB()

    def verify_signature(self, timestamp, raw_body, signature):
        """Verify the webhook signature from Daily."""
        if not DAILY_WEBHOOK_SECRET:
            return True

        try:
            decoded_secret = base64.b64decode(DAILY_WEBHOOK_SECRET)
            signature_string = f"{timestamp}.{raw_body}"
            
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

    def handle_recording_started(self):
        """Handle recording.started event."""
        print("[DEBUG] handle_recording_started called")
        return jsonify({'status': 'recording started'}), 200

    def handle_recording_ready(self, payload):
        """Handle recording.ready-to-download event."""
        print("[DEBUG] handle_recording_ready called")
        recording_id = payload['recording_id']
        room_name = payload['room_name']
        
        print(f"[DEBUG] recording_id: {recording_id}, room_name: {room_name}")
        
        # Update database with recording ID
        self.database.update_supabase_data(
            "AI_summary",
            {'recording_id': recording_id},
            ['room_name', room_name]
        )
        
        # Start batch processor job
        job_response = self.batch_processor.submit_batch_processor_job(recording_id)
        print(f"[DEBUG] batch processor job started, job_response: {job_response}")
        return jsonify({'status': f'batch processor job started with id: {job_response["id"]}'}), 200

    def handle_job_finished(self, payload):
        """Handle batch-processor.job-finished event."""
        print("[DEBUG] handle_job_finished called")
        job_id = payload['id']
        recording_id = payload['input']['recordingId']
        
        print(f"[DEBUG] job_id: {job_id}, recording_id: {recording_id}")

        # Process transcription
        text_raw = self.batch_processor.process_transcription_job(job_id)
        print(f"[DEBUG] Transcription text length: {len(text_raw)} chars")
        download_result = self.downloader.get_download_link(recording_id)
        print(f"[DEBUG] Download link: {download_result.get('download_link')}")
        
        # Set up models and media URLs
        media_urls = [download_result['download_link']]
        models = {
            "face": {},
            "language": {},
            "prosody": {}
        }

        # Get summary and process merge data
        summary = self.batch_processor.process_summary_job(job_id)
        print(f"[DEBUG] Summary text length: {len(summary)} chars")
        
        application_id = self.database.get_supabase_data(
            "AI_summary", 
            "application_id", 
            ["recording_id", recording_id]
        ).data[0]['application_id']
        print(f"[DEBUG] application_id: {application_id}")
        
        merge_job_id = self.database.get_supabase_data(
            "applications", 
            "job_id", 
            ["id", application_id]
        ).data[0]['job_id']
        print(f"[DEBUG] merge_job_id: {merge_job_id}")
        
        # TODO: 
        # https://docs.merge.dev/ats/candidates/
        # candidate_resume = ResumeParser.parse_pdf(candidate_resume_url)
        # retrieve from merge using candidate key
        # 

        # Process merge job data
        merge_job = self.merge_client.process_job_data(merge_job_id)
        merge_job_description = merge_job.data.get("name") + merge_job.data.get("description")
        print(f"[DEBUG] merge_job_description length: {len(merge_job_description)}")
        
        technical_evaluation = evaluate_candidate(
            interview_transcript=text_raw,
            resume=candidate_resume,
            job_description=merge_job_description
        )
        
        print(f"[DEBUG] response_eval output: {technical_evaluation}")
        
        self.database.update_supabase_data(
            "AI_summary",
            {
                "transcript_summary": summary,
                "technical_eval": technical_evaluation,
                'batch-processor_transcript_id': job_id
            },
            ['recording_id', recording_id]
        )

        # Start emotion analysis
        callback_url = (
            f'https://roborecruiter-analysisbot-production.up.railway.app/hume-callback/hume'
            f'?recording_id={recording_id}&job_description={merge_job_description}'
        )
        emotion_job_id = run_emotion_analysis(media_urls, text_raw, models, callback_url)
        print(f"[DEBUG] Emotion analysis started, job_id: {emotion_job_id}")
        
        return jsonify({'status': f'Emotion analysis job started with ID: {emotion_job_id}'}), 200

# Blueprint setup.
bp = Blueprint('eval', __name__)
CORS(bp)
webhook_handler = WebhookHandler()

@bp.route("/webhook", methods=['POST'])
@cross_origin()
def handle_webhook():
    try:
        # Get raw body and headers for verification
        raw_body = request.get_data(as_text=True)
        timestamp = request.headers.get('X-Webhook-Timestamp')
        signature = request.headers.get('X-Webhook-Signature')
        
        # Verify signature if present
        if timestamp and signature:
            if not webhook_handler.verify_signature(timestamp, raw_body, signature):
                return jsonify({'error': 'Invalid signature'}), 401
        
        # Parse JSON data
        data = request.get_json()
        
        # Handle webhook verification
        if data.get('test') == 'test':
            return jsonify({'status': 'success'}), 200
        
        # Route events to appropriate handlers
        event_type = data.get('type')
        if event_type == 'recording.started':
            return webhook_handler.handle_recording_started()
        elif event_type == 'recording.ready-to-download':
            return webhook_handler.handle_recording_ready(data['payload'])
        elif event_type == 'batch-processor.job-finished':
            return webhook_handler.handle_job_finished(data['payload'])
        
        print("[DEBUG] Unsupported event type")
        return jsonify({'error': 'Unsupported event type'}), 400
        
    except Exception as e:
        print(f"[DEBUG] Exception in handle_webhook: {str(e)}")
        return handle_server_error(e)