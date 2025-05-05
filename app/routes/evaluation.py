from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import requests
import json
from app.controllers.aws_s3 import fetch_from_s3
from app.controllers.supabase_db import SupabaseDB
from app.services.text_analysis import analyze_interview_parallel
from app.utils.request_handler import handle_success, handle_server_error
from app.routes import bp

# New endpoint to handle notifications from the analysis bot
@bp.route('/analysis-bot', methods=['POST'])
@cross_origin()
def analysis_bot():
    database = SupabaseDB()

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
        
        fetch_from_s3(bucket, file_path, local_path) #recording
        fetch_from_s3(bucket, transcript_path, transcript_local_path) #transcript

        with open(transcript_local_path, 'r') as f: 
            transcript = json.dumps(json.load(f))


        # Analyze the interview
        text_eval      = analyze_interview_parallel(transcript)
        summary        = "summary" #TODO: add summary
        emotional_eval = "emotional_eval" #TODO: add emotional eval


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

        return handle_success("Successfully updated the database")
    except Exception as e:
        return handle_server_error(e)