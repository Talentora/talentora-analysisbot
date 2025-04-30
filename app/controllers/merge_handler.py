import os
import tempfile
import openai
import requests
import requests
from app.controllers.supabase_db import SupabaseDB
from app.services.resume.analyzer import ResumeAnalyzer
from app.services.resume.parsers import ResumeParser

class MergeHandler:
    def __init__(self):
        # Configure the Merge ATS API client
        self.merge_url = "https://talentora-database-production.up.railway.app/"
        self.supabase = SupabaseDB()
    
    
    
    def get_latest_resume(self, merge_linked_account_id: str, candidate_id: str) -> str:
        """
        Retrieves the latest resume for a given candidate ID and saves it to a temporary file.

        :param candidate_id: The Merge candidate ID.
        :return: The file path to the downloaded resume.
        :raises Exception: If no resume is found or download fails.
        """
        
        company_token = self.supabase.get_company_token(merge_linked_account_id)
        
        params = {
            "account_token": company_token
        }
        
        req_url = self.merge_url + f"/merge/resume/{candidate_id}"
        
        response = requests.get(req_url, params=params)
        
        return response['data']['file_url']

    def get_merge_candidate_data(self, merge_linked_account_id: str, merge_candidate_id: str) -> dict:
        """
        Retrieves candidate data from the Merge API.

        :param merge_candidate_id: The Merge candidate ID.
        :return: A dictionary containing candidate data.
        :raises: ValueError if candidate_id is invalid
        :raises: Exception for other API errors
        """
        company_token = self.supabase.get_company_token(merge_linked_account_id)
        
        params = {
            "account_token": company_token
        } 
        
        req_url = self.merge_url + f"/merge/candidate/{merge_candidate_id}"
        
        response = requests.get(req_url, params=params)
        
        
        if not response.json()['candidate']:
            raise ValueError(f"Missing candidate data for candidate ID: {merge_candidate_id}")
        
        
        
        if response.status_code != 200:
            raise Exception(f"Error retrieving candidate data: {response.status_code}")
        
        return response.json()['data'] #return the data from the response

    def _get_job_description(self, merge_linked_account_id: str, merge_job_id: str) -> str:
        """
        Retrieves the job description for a given Merge job ID.

        :param merge_job_id: The Merge job ID.
        :return: The job description for the given job.
        """
        
        company_token = self.supabase.get_company_token(merge_linked_account_id)
        
        params = {
            "account_token": company_token
        }   
        
        req_url = self.merge_url + f"/merge/job/{merge_job_id}"
        
        response = requests.get(req_url, params=params)
        
        return response.json()['data']['description']
    
    
    def _download_resume(self, url: str, temp_file) -> None:
        """Download resume to temporary file"""
        response = requests.get(url)
        response.raise_for_status()
        temp_file.write(response.content)
        temp_file.flush()
    
    
    def run_resume_analysis(self, resume_url: str, job_desc : str, job_config: dict) -> dict:
        """
        Runs the resume analysis using the ResumeAnalyzer class. Uploads experience, education, and skills to the applicants table. 
        Uploads analysis scores to the applications table.
        """
        try:


            # Initialize services once
            analyzer = ResumeAnalyzer()
            parser = ResumeParser()
  

            # Process resume #TODO: Fetch link and scan document remotely rather than downloading
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as temp_pdf:
                self._download_resume(resume_url, temp_pdf)
                
                # Analyze resume
                resume_analysis = analyzer.analyze(temp_pdf.name, job_desc, job_config)
                parser_result = parser.parse_pdf(temp_pdf.name)
                
                # Combine results
                combined_result = {
                    "resume_analysis": resume_analysis,
                    "parser_result": parser_result
                }
                
                # Save to database                
                return combined_result

        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to download resume: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Resume analysis failed: {str(e)}")
        
    def generate_ai_job_config(self, job_description: str) -> dict:
        """
        Generates a job config for a given job description using the AI API.
        """
        prompt = (
            "You are an expert technical recruiter. Given the following job description, "
            "extract the required technical and soft skills (with importance weights from 0.1 to 1.0), "
            "minimum years of experience, required education (degree, field, min_gpa), "
            "and assign category weights for skills, experience, and education (summing to 1.0). "
            "Return your answer as a JSON object with the following format:\n\n"
            "{\n"
            "  \"skill_weights\": {\"Skill1\": 1.0, \"Skill2\": 0.8, ...},\n"
            "  \"min_years_experience\": 0,\n"
            "  \"required_education\": {\"degree\": \"\", \"field\": \"\", \"min_gpa\": 0.0},\n"
            "  \"category_weights\": {\"skills\": 0.5, \"experience\": 0.3, \"education\": 0.2}\n"
            "}\n\n"
            "Job Description:\n"
            f"{job_description}\n"
            "Respond ONLY with the JSON object."
        )

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an experienced recruiter assessing a candidate."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "job_config_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "skill_weights": {
                                "type": "object",
                                "additionalProperties": { "type": "number" }
                            },
                            "min_years_experience": { "type": "number" },
                            "required_education": {
                                "type": "object",
                                "properties": {
                                    "degree": { "type": "string" },
                                    "field": { "type": "string" },
                                    "min_gpa": { "type": "number" }
                                },
                                "required": ["degree", "field", "min_gpa"],
                                "additionalProperties": False
                            },
                            "category_weights": {
                                "type": "object",
                                "properties": {
                                    "skills": { "type": "number" },
                                    "experience": { "type": "number" },
                                    "education": { "type": "number" }
                                },
                                "required": ["skills", "experience", "education"],
                                "additionalProperties": False
                            }
                        },
                        "required": [
                            "skill_weights",
                            "min_years_experience",
                            "required_education",
                            "category_weights"
                        ],
                        "additionalProperties": False
                    }
                }
            }
        )
        return response.choices[0].message.content
        
    def validate_job_config(job_config: dict, *, eps: float = 1e-6) -> None:
        """
        Validates the structure and numeric constraints of a job_config dict.
        Raises ValueError if any check fails.
        
        :param job_config: Dict to validate.
        :param eps: Tolerance for floating-point sums.
        """
        # 1. Top-level keys
        expected_keys = {
            "skill_weights",
            "min_years_experience",
            "required_education",
            "category_weights"
        }
        missing = expected_keys - job_config.keys()
        extra   = set(job_config.keys()) - expected_keys
        if missing:
            raise ValueError(f"Missing keys: {missing}")
        if extra:
            raise ValueError(f"Unexpected keys: {extra}")

        # 2. skill_weights: dict[str, number in [0.0,1.0]]
        sw = job_config["skill_weights"]
        if not isinstance(sw, dict):
            raise ValueError("skill_weights must be a dict")
        for skill, weight in sw.items():
            if not isinstance(skill, str):
                raise ValueError(f"Skill name must be a string, got {type(skill)}")
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Weight for {skill} is not numeric")
            if not (0.0 <= weight <= 1.0):
                raise ValueError(f"Weight for {skill} ({weight}) must be between 0.0 and 1.0")

        # 3. min_years_experience: non-negative number
        mxe = job_config["min_years_experience"]
        if not isinstance(mxe, (int, float)):
            raise ValueError("min_years_experience must be a number")
        if mxe < 0:
            raise ValueError("min_years_experience cannot be negative")

        # 4. required_education: dict with degree(str), field(str), min_gpa(number)
        re = job_config["required_education"]
        if not isinstance(re, dict):
            raise ValueError("required_education must be a dict")
        for key, expected_type in [("degree", str), ("field", str), ("min_gpa", (int, float))]:
            if key not in re:
                raise ValueError(f"required_education missing '{key}'")
            if not isinstance(re[key], expected_type):
                raise ValueError(f"required_education['{key}'] must be {expected_type}")
        if not (0.0 <= re["min_gpa"] <= 4.0):
            raise ValueError("min_gpa should be between 0.0 and 4.0")

        # 5. category_weights: dict[str, number], sum == 1.0
        cw = job_config["category_weights"]
        if not isinstance(cw, dict):
            raise ValueError("category_weights must be a dict")
        total = 0.0
        for cat, w in cw.items():
            if cat not in ("skills", "experience", "education"):
                raise ValueError(f"Invalid category '{cat}' in category_weights")
            if not isinstance(w, (int, float)):
                raise ValueError(f"Weight for '{cat}' is not numeric")
            if w < 0.0:
                raise ValueError(f"Weight for '{cat}' cannot be negative")
            total += w
        if abs(total - 1.0) > eps:
            raise ValueError(f"category_weights must sum to 1.0 (got {total:.6f})")


        return True
    
    def handle_new_job(self, data: dict) -> None:
        """
        Handles a new job event from the Merge API.

        :param payload: The payload containing job data.
        """
        job_desc = data.get("description")   
        
        job_config = self.generate_ai_job_config(job_desc)
        
        if not self.validate_job_config(job_config):
            raise ValueError("Invalid job config")
        
        if not self.supabase.insert_new_job(job_config, job_desc):
            raise ValueError("Failed to insert new job")
        
        return True
    
    def handle_new_application(self, payload: dict) -> None:
        """
        Handles a new application event from the Merge API.
        """
        try:
            # Extract required fields from payload
            if not all(payload.get(field) for field in ["job", "candidate", "id", "linked_account"]):
                raise ValueError("Missing required fields in payload")
                
            merge_job_id = payload["job"]
            merge_candidate_id = payload["candidate"]
            merge_application_id = payload["id"]
            merge_linked_account_id = payload["linked_account"]["id"]
            
            # Get required data
            job_desc = self._get_job_description(merge_linked_account_id, merge_job_id)

            resume_config = self.supabase.get_job_resume_config(merge_job_id)
            if not resume_config:
                #TODO: Handle this
                pass
            
            resume_url = self.get_latest_resume(merge_linked_account_id, merge_candidate_id)
            
            # Run analysis and store results
            combined_result = self.run_resume_analysis(resume_url, job_desc, resume_config)
            self.supabase.insert_new_ai_summary(merge_application_id, combined_result)
            self.supabase.insert_new_application(merge_application_id, merge_job_id)
            
            return True
            
        except ValueError as e:
            print(f"Invalid payload data: {str(e)}")
            raise
        except Exception as e:
            print(f"Error processing application: {str(e)}")
            raise
        
        
        
