from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import requests
import json
from app.controllers.aws_s3 import fetch_from_s3
from app.controllers.supabase_db import SupabaseDB
from app.services.text_analysis import analyze_interview_parallel

# class WebhookHandler:
#     def __init__(self):
#         self.batch_processor = DailyBatchProcessor(DAILY_API_KEY)
#         self.downloader = DailyVideoDownloader(DAILY_API_KEY)
#         self.merge_client = MergeAPIClient(MERGE_ACCOUNT_TOKEN, MERGE_API_KEY)
#         self.database = SupabaseDB()

#     def verify_signature(self, timestamp, raw_body, signature):
#         """Verify the webhook signature from Daily."""
#         if not DAILY_WEBHOOK_SECRET:
#             return True

#         try:
#             decoded_secret = base64.b64decode(DAILY_WEBHOOK_SECRET)
#             signature_string = f"{timestamp}.{raw_body}"
            
#             computed_hmac = hmac.new(
#                 decoded_secret,
#                 signature_string.encode('utf-8'),
#                 hashlib.sha256
#             )
#             computed_signature = base64.b64encode(computed_hmac.digest()).decode()
            
#             return hmac.compare_digest(computed_signature, signature)
#         except Exception as e:
#             print(f"Signature verification failed: {e}")
#             return False

#     def handle_recording_started(self):
#         """Handle recording.started event."""
#         print("[DEBUG] handle_recording_started called")
#         return jsonify({'status': 'recording started'}), 200

#     def handle_recording_ready(self, payload):
#         """Handle recording.ready-to-download event."""
#         print("[DEBUG] handle_recording_ready called")
#         recording_id = payload['recording_id']
#         room_name = payload['room_name']
        
#         print(f"[DEBUG] recording_id: {recording_id}, room_name: {room_name}")
        
#         # Update database with recording ID
#         self.database.update_supabase_data(
#             "AI_summary",
#             {'recording_id': recording_id},
#             ['room_name', room_name]
#         )
        
#         # Start batch processor job
#         job_response = self.batch_processor.submit_batch_processor_job(recording_id)
#         print(f"[DEBUG] batch processor job started, job_response: {job_response}")
#         return jsonify({'status': f'batch processor job started with id: {job_response["id"]}'}), 200

#     def handle_job_finished(self, payload):
#         """Handle batch-processor.job-finished event."""
#         print("[DEBUG] handle_job_finished called")
#         job_id = payload['id']
#         recording_id = payload['input']['recordingId']
        
#         print(f"[DEBUG] job_id: {job_id}, recording_id: {recording_id}")

#         # Process transcription
#         text_raw = self.batch_processor.process_transcription_job(job_id)
#         print(f"[DEBUG] Transcription text length: {len(text_raw)} chars")
#         download_result = self.downloader.get_download_link(recording_id)
#         print(f"[DEBUG] Download link: {download_result.get('download_link')}")
        
#         # Set up models and media URLs
#         media_urls = [download_result['download_link']]
#         models = {
#             "face": {},
#             "language": {},
#             "prosody": {}
#         }

#         # Get summary and process merge data
#         summary = self.batch_processor.process_summary_job(job_id)
#         print(f"[DEBUG] Summary text length: {len(summary)} chars")
        
#         application_id = self.database.get_supabase_data(
#             "AI_summary", 
#             "application_id", 
#             ["recording_id", recording_id]
#         ).data[0]['application_id']
#         print(f"[DEBUG] application_id: {application_id}")
        
#         merge_job_id = self.database.get_supabase_data(
#             "applications", 
#             "job_id", 
#             ["id", application_id]
#         ).data[0]['job_id']
#         print(f"[DEBUG] merge_job_id: {merge_job_id}")

#         # Process merge job data
#         merge_job = self.merge_client.process_job_data(merge_job_id)
#         merge_job_description = merge_job.data.get("name") + merge_job.data.get("description")
#         print(f"[DEBUG] merge_job_description length: {len(merge_job_description)}")
        
#         # Update database with evaluation data
#         # data_to_insert = response_eval(text_raw, merge_job_description)
#         data_to_insert = analyze_interview_parallel(text_raw)     

#         print(f"[DEBUG] response_eval output: {data_to_insert}")
        
#         self.database.update_supabase_data(
#             "AI_summary",
#             {
#                 "transcript_summary": summary,
#                 'text_eval': data_to_insert,
#                 'batch-processor_transcript_id': job_id
#             },
#             ['recording_id', recording_id]
#         )

#         # Start emotion analysis
#         callback_url = (
#             f'https://roborecruiter-analysisbot-production.up.railway.app/hume-callback/hume'
#             f'?recording_id={recording_id}&job_description={merge_job_description}'
#         )
#         emotion_job_id = run_emotion_analysis(media_urls, text_raw, models, callback_url)
#         print(f"[DEBUG] Emotion analysis started, job_id: {emotion_job_id}")
        
#         return jsonify({'status': f'Emotion analysis job started with ID: {emotion_job_id}'}), 200

# # Blueprint setup.
# bp = Blueprint('eval', __name__)
# CORS(bp)
# webhook_handler = WebhookHandler()

# New endpoint to handle notifications from the analysis bot
@bp.route('/analysis-bot', methods=['POST'])
@cross_origin()
def analysis_bot():
    database = SupabaseDB()

    try:
        data                  = request.get_json()
        recording_id          = data['recording_id']
        application_id        = data['application_id']
        job_id                = data['job_id']
        user_id               = data['user_id']

        bucket                = 'talentorarecordings'
        file_path             = f"{user_id}/{job_id}/interview.mp4"
        transcript_path       = f"{user_id}/{job_id}/transcript.json"
        local_path            = f"./tmp/{file_path}"
        transcript_local_path = f"./tmp/{transcript_path}"
        
        fetch_from_s3(bucket, file_path, local_path) #recording
        fetch_from_s3(bucket, transcript_path, transcript_local_path) #transcript

        with open(transcript_local_path, 'r') as f: 
            transcript = json.dumps(json.load(f))


        # analyze the interview
        text_eval = analyze_interview_parallel(transcript)
        summary   = ""

        # update the database
        database.update_supabase_data(
            "AI_summary",
            {
                "application_id": application_id,
                "recording_id": recording_id,
                "transcript_summary": summary,
                'text_eval': text_eval,
                'batch-processor_transcript_id': job_id,
                'emotion_eval': emotional_eval
            },
            ['application_id', application_id]
        )

        # grab the S3‐hosted transcript JSON
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        resp           = requests.get(transcript_uri)
        text           = resp.json()['results']['transcripts'][0]['transcript']

        # now feed that into your existing services *in memory*
        text_lines     = text.splitlines()
        eval_data      = analyze_interview_parallel(text_lines)
        # or if you want the old response_eval:
        # eval_data    = response_eval(text_lines, merge_job_description)

        return jsonify({
                'transcript': text,
                'evaluation': eval_data
            }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500