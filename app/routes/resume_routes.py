from flask import Blueprint, request, jsonify
import os
from app.controllers.supabase_db import SupabaseDB
from app.services.resume.analyzer import ResumeAnalyzer

resume_bp = Blueprint('resume', __name__)

@resume_bp.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    """Analyze a resume given a URL to an S3 object, job description, and job config."""
    data = request.json
    s3_link = data.get('s3_link')
    job_description = data.get('job_description')
    job_config = data.get('job_config')

    if not s3_link or not job_description or not job_config:
        return jsonify({"error": "Missing required fields"}), 400
    # Download the file directly from the signed URL
    try:
        import requests
        response = requests.get(s3_link)
        if response.status_code == 200:
            # Save the file to temporary storage
            pdf_path = 'temp_resume.pdf'
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download file from {s3_link}")

        # Analyze the resume
        analyzer = ResumeAnalyzer()
        result = analyzer.analyze(pdf_path, job_description, job_config)

        # Clean up the temporary file
        os.remove(pdf_path)

        if result:
            # Convert any non-serializable objects to strings or simple types
            serializable_result = {
                'overall_match': float(result.get('overall_match', 0)),
                'skills_match': float(result.get('skills_match', 0)),
                'education_match': float(result.get('education_match', 0)),
                'experience_match': float(result.get('experience_match', 0)),
                'skill_scores': {k: float(v) for k, v in result.get('skill_scores', {}).items()},
                'education_details': {
                    'overall_match': float(result.get('education_details', {}).get('overall_match', 0)),
                    'degree_level_match': float(result.get('education_details', {}).get('degree_level_match', 0)),
                    'field_match': float(result.get('education_details', {}).get('field_match', 0)),
                    'details': result.get('education_details', {}).get('details', {})
                },
                'experience_details': {
                    'overall_match': float(result.get('experience_details', {}).get('overall_match', 0)),
                    'role_similarity': float(result.get('experience_details', {}).get('role_similarity', 0)),
                    'years_experience': float(result.get('experience_details', {}).get('years_experience', 0)),
                    'years_match': float(result.get('experience_details', {}).get('years_match', 0)),
                    'company_relevance': float(result.get('experience_details', {}).get('company_relevance', 0))
                }
            }
            return jsonify(serializable_result), 200
        else:
            return jsonify({"error": "Analysis failed"}), 500

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Print the full traceback for debugging
        return jsonify({"error": str(e)}), 500 