from flask import Blueprint, request, jsonify
from app.controllers.merge_handler import MergeHandler
import hmac, hashlib, base64
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any
from functools import wraps

from app.controllers.supabase_db import SupabaseDB

load_dotenv()

MERGE_WEBHOOK_SECRET = os.getenv("MERGE_WEBHOOK_SECRET")
if not MERGE_WEBHOOK_SECRET:
    raise ValueError("MERGE_WEBHOOK_SECRET environment variable is not set")

merge_bp = Blueprint("merge", __name__)

def verify_signature(raw_body: bytes, webhook_signature: str) -> bool:
    """
    Verify the signature of the request
    """
    if not webhook_signature:
        logging.warning("No webhook signature provided")
        return False
        
    hmac_digest = hmac.new(MERGE_WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256).digest()
    b64_encoded = base64.urlsafe_b64encode(hmac_digest).decode()
    return hmac.compare_digest(b64_encoded, webhook_signature)


def validate_webhook(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        sig = request.headers.get("X-Merge-Webhook-Signature", "")
        raw = request.get_data()
        
        if not verify_signature(raw, sig):
            logging.warning(f"Invalid webhook signature received from {request.remote_addr}")
            return jsonify({"error": "Invalid signature"}), 401
            
        try:
            payload = request.get_json(force=True)
            if not payload:
                logging.error("Empty payload received")
                return jsonify({"error": "No data found in payload"}), 400
        except Exception as e:
            logging.error(f"Failed to parse JSON payload: {str(e)}")
            return jsonify({"error": "Invalid JSON payload"}), 400
            
        return f(*args, **kwargs)
    return decorated_function


@merge_bp.route("/new-job", methods=["POST"])
@validate_webhook
def new_job():
    logging.info("New job webhook received")
    try:
        payload = request.get_json()
        handler = MergeHandler()

        logging.info(f"Payload: {payload}")

        handler.handle_new_job(payload)
        
        logging.info("New job processed successfully")
        return jsonify({"message": "New job created successfully"}), 200
    except Exception as e:
        logging.error(f"Error processing new job: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@merge_bp.route("/new-application", methods=["POST"])
@validate_webhook
def new_application():
    logging.info("New application webhook received")
    try:
        payload = request.get_json(force=True)
        handler = MergeHandler()

        logging.info(f"Payload: {payload}")

        if not handler.handle_new_application(payload):
            logging.error("Failed to process application")
            return jsonify({"error": "Failed to process application"}), 500
        
        logging.info("Application processed successfully")
        return jsonify({"message": "Application processed"}), 200
    except Exception as e:
        logging.error(f"Error processing new application: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# Steps:
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