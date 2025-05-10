from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import requests
import json
from app.controllers.aws_s3 import fetch_from_s3, generate_s3_presigned_download_url
from app.controllers.supabase_db import SupabaseDB
from app.services.text_analysis import analyze_interview_parallel
from app.utils.request_handler import handle_success, handle_server_error
from app.services.summarize import summarize_interview_json
import os
import logging
from app.routes.hume_callback import HumeCallbackHandler

evaluation_bp = Blueprint('evaluation', __name__)

# New endpoint to handle notifications from the analysis bot
@evaluation_bp.route('/analysis-bot', methods=['POST'])
@cross_origin()
def analysis_bot():
    database = SupabaseDB()
    hume_callback = HumeCallbackHandler()
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis bot")
    logger.info(f"Request: {request.get_json()}")


    try:
        # Fetch data
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
        
        # fetch_from_s3(bucket, file_path, local_path) #recording
        recording_url = generate_s3_presigned_download_url(bucket, file_path)
        fetch_from_s3(bucket, transcript_path, transcript_local_path) #transcript

        with open(transcript_local_path, 'r') as f: 
            transcript = json.dumps(json.load(f))

        # Analyze the interview
        text_eval      = analyze_interview_parallel(transcript)
        summary        = summarize_interview_json(transcript)
        emotional_eval = hume_callback.process_callback(recording_url)


        # Update the database
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


        # delete the tmp files
        os.remove(local_path)
        os.remove(transcript_local_path)

        return handle_success("Successfully updated the database")
    except Exception as e:
        return handle_server_error(e)