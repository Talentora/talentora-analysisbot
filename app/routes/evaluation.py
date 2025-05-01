from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import requests
import json
from app.controllers.aws_s3 import fetch_from_s3
from app.controllers.supabase_db import SupabaseDB
from app.services.text_analysis import analyze_interview_parallel


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
        text_eval      = analyze_interview_parallel(transcript)
        summary        = ""
        emotional_eval = ""

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