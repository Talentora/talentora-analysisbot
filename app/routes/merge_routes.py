from flask import Blueprint, request, jsonify
from app.controllers.merge_handler import MergeHandler
import hmac, hashlib, base64
from dotenv import load_dotenv
import os

from app.controllers.supabase_db import SupabaseDB

load_dotenv()

MERGE_WEBHOOK_SECRET = os.getenv("MERGE_WEBHOOK_SECRET")

merge_bp = Blueprint("merge", __name__)

def verify_signature(raw_body : bytes, webhook_signature : str):
    """
    Verify the signature of the request
    """
    hmac_digest = hmac.new(MERGE_WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256).digest()
    b64_encoded = base64.urlsafe_b64encode(hmac_digest).decode()
    return hmac.compare_digest(b64_encoded, webhook_signature)



@merge_bp.route("/new-job", methods=["POST"])
def new_job():
    sig = request.headers.get("X-Merge-Webhook-Signature", "")
    raw = request.get_data() # returns bytes, no need to decode
    if not verify_signature(raw, sig):
        return jsonify({"error": "Invalid signature"}), 401 
    payload = request.get_json(force=True)

    if not payload:
        raise ValueError("No data found in payload")

    try:
        handler = MergeHandler()
        handler.handle_new_job(payload)
        
        return jsonify({"message": "New job created successfully"}), 200
    except Exception as e:
        print(f"Error in merge_webhook - new_job: {str(e)}")
        return jsonify({"error": str(e)}), 500


@merge_bp.route("/new-application", methods=["POST"])
def new_application():
    sig = request.headers.get("X-Merge-Webhook-Signature", "")
    raw = request.get_data()  # returns bytes
    if not verify_signature(raw, sig):
        return jsonify({"error": "Invalid signature"}), 401
    try:
        payload = request.get_json(force=True)

        handler = MergeHandler()

        if not handler.handle_new_application(payload):
            return jsonify({"error": "Failed to process application"}), 500
        
        
        return jsonify({"message": "Application processed"}), 200
    except Exception as e:
        print(f"Error in merge_webhook - new_application: {str(e)}")
        return jsonify({"error": str(e)}), 500
   

#Steps:
# 1. Get the application id from the webhook
# 2. Get the candidate id from the webhook
# 3. Get the job id from the webhook
# 4. Get the resume url from the merge SDK 


# Find a way to ger email, full name


# 5. Download the resume
# 6. Using the merge job ID, get the get job ID from jobs table, then use job ID to get config from job_interview_config table TODO: still need to get job description, not sure how
# 7. Analyze the resume using the ResumeAnalyzer class (takes in resume path, job description, and job config)
# 8. Use ResumeParser to parse the resume into experience, education, and skills
# 9. Store the ResumeParser results in the applicants table 
# 10. Store the ResumeAnalyzer results in the applications table 