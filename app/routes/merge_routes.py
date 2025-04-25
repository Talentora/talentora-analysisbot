from flask import Blueprint, request, jsonify
from app.controllers.merge_handler import MergeHandler


merge_bp = Blueprint("merge", __name__)

@merge_bp.route("/merge/webhook", methods=["POST"])
def merge_webhook():
    try:
        payload = request.get_json()

        # Extract core fields from payload
        data = payload.get("data", {})
        application_id = data.get("id")
        candidate_id = data.get("candidate_id")  
        job_id = data.get("job_id")
       
        handler = MergeHandler()
        handler.run_resume_analysis(candidate_id, job_id, application_id)
        
    except Exception as e:
        print(f"Error in merge_webhook: {str(e)}")
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