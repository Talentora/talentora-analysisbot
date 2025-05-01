from flask import Blueprint, request, jsonify
from flask_cors import cross_origin, CORS
import requests

from app.configs.merge_config import MERGE_API_KEY, MERGE_ACCOUNT_TOKEN
from app.controllers.supabase_db import SupabaseDB
from app.controllers.merge import MergeAPIClient
from app.services.score_calculation import response_eval
from app.services.sentiment import run_emotion_analysis
from app.utils import handle_server_error

bp = Blueprint('eval', __name__)
CORS(bp)

class WebhookHandler:
    def __init__(self):
        self.database     = SupabaseDB()
        self.merge_client = MergeAPIClient(MERGE_ACCOUNT_TOKEN, MERGE_API_KEY)

    def handle_recording_insert(self, record):
        """
        Supabase will POST:
          {
            "type":"INSERT",
            "table":"recordings",
            "schema":"public",
            "record":{
              "id":"…",
              "room_name":"…",
              "video_path":"…",
              "transcript_path":"…",
              "created_at":"…"
            },
            "old_record": null
          }
        """
        room_name      = record["room_name"]
        video_key      = record["video_path"]
        transcript_key = record["transcript_path"]

        # generate signed URLs for download
        video_url      = self.database.create_signed_url("recordings",  [video_key])[0]
        transcript_url = self.database.create_signed_url("transcripts", [transcript_key])[0]

        # fetch and split transcript
        resp = requests.get(transcript_url)
        resp.raise_for_status()
        transcript_lines = [
            line.strip() for line in resp.text.splitlines() if line.strip()
        ]

        #  fetch your merge job description
        #    — adjust this to whatever lookup needed
        merge_job = self.merge_client.process_job_data()
        desc = (
            merge_job.data.get("name","") +
            merge_job.data.get("description","")
        )

        # run text evaluation
        text_eval = response_eval(transcript_lines, desc)

        # store transcript_summary + text_eval back in your AI_summary table
        self.database.update_supabase_data(
            "AI_summary",
            {
                "transcript_summary": "\n".join(transcript_lines),
                "text_eval": text_eval
            },
            ["room_name", room_name]
        )

        # emotion analysis on the video URL
        models = {"face":{}, "language":{}, "prosody":{}}
        callback_url = (
            f"https://your-domain.com/hume-callback/hume"
            f"?recording_id={record['id']}&job_description={desc}"
        )
        emotion_job_id = run_emotion_analysis(
            media_urls=[video_url],
            text=transcript_lines,
            models=models,
            callback_url=callback_url
        )

        return jsonify({
            "status":          "analysis started",
            "emotion_job_id":  emotion_job_id
        }), 200

webhook_handler = WebhookHandler()

@bp.route("/webhook", methods=["POST"])
@cross_origin()
def handle_webhook():
    try:
        data = request.get_json()

        # Supabase DB webhook on public.recordings INSERT
        if data.get("table") == "recordings" and data.get("type") == "INSERT":
            return webhook_handler.handle_recording_insert(data["record"])

        return jsonify({"error":"unsupported event"}), 400
    except Exception as e:
        return handle_server_error(e)
