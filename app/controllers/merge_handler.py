import os
import tempfile
import requests
import requests
from app.controllers.supabase_db import SupabaseDB
from app.services.resume.analyzer import ResumeAnalyzer
from app.services.resume.parsers import ResumeParser

class MergeHandler:
    def __init__(self):
        # Configure the Merge ATS API client
        self.merge_url = "https://talentora-database-production.up.railway.app/"
    
    
        
    def get_latest_resume(self, job_id: str, candidate_id: str) -> str:
        """
        Retrieves the latest resume for a given candidate ID and saves it to a temporary file.

        :param candidate_id: The Merge candidate ID.
        :return: The file path to the downloaded resume.
        :raises Exception: If no resume is found or download fails.
        """
        
        company_token = self.get_company_token(job_id)
        
        params = {
            "account_token": company_token
        }
        
        req_url = self.merge_url + f"/merge/resume/{candidate_id}"
        
        response = requests.get(req_url, params=params)
        
        return response['data']['file_url']

    def get_merge_candidate_data(self, job_id: str, merge_candidate_id: str) -> dict:
        """
        Retrieves candidate data from the Merge API.

        :param merge_candidate_id: The Merge candidate ID.
        :return: A dictionary containing candidate data.
        :raises: ValueError if candidate_id is invalid
        :raises: Exception for other API errors
        """
        company_token = self.get_company_token(job_id)
        
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

    def _get_job_description(self, merge_job_id: str) -> str:
        """
        Retrieves the job description for a given Merge job ID.

        :param merge_job_id: The Merge job ID.
        :return: The job description for the given job.
        """
        
        company_token = self.get_company_token(merge_job_id)
        
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
    
    
    def run_resume_analysis(self, merge_candidate_id: str, merge_job_id: str, merge_application_id: str) -> dict:
        """
        Runs the resume analysis using the ResumeAnalyzer class. Uploads experience, education, and skills to the applicants table. 
        Uploads analysis scores to the applications table.
        """
        try:
            # Validate inputs
            if not all([merge_candidate_id, merge_job_id, merge_application_id]):
                raise ValueError("Missing required parameters")

            # Initialize services once
            analyzer = ResumeAnalyzer()
            parser = ResumeParser()
            supabase = SupabaseDB()

            # Get required data
            resume_url = self.get_latest_resume(merge_candidate_id)
            job_description = self._get_job_description(merge_job_id)
            # job_config = self._get_job_config(merge_job_id)  # Fetch from database #TODO: implement this
            
            #temporary job config
            job_config = {
                'skill_weights': {
                    # Required technical skills with importance weights
                    'Python': 1.0,          # Core requirement
                    'JavaScript': 1.0,      # Core requirement
                    'React': 0.8,           # Important but not core
                    'AWS': 0.7,             # Cloud platform experience
                    'CI/CD': 0.6,           # DevOps practices
                    'DevOps': 0.6,          # DevOps practices
                    'Docker': 0.5,          # Nice to have
                    'Kubernetes': 0.4,      # Nice to have
                    'SQL': 0.7,             # Database experience
                    'Git': 0.8              # Version control
                },
                
                'min_years_experience': 5,  # Minimum years of experience required
                
                'required_education': {
                    'degree': 'Bachelor',   # Required degree level
                    'field': 'Computer Science',  # Required field of study
                    'min_gpa': 3.0          # Minimum GPA requirement
                },
                
                'category_weights': {
                    'skills': 0.5,          # Skills contribute 50% to overall score
                    'experience': 0.3,      # Experience contributes 30% to overall score
                    'education': 0.2        # Education contributes 20% to overall score
                }
            }

            # Process resume
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as temp_pdf:
                self._download_resume(resume_url, temp_pdf)
                
                # Analyze resume
                resume_analysis = analyzer.analyze(temp_pdf.name, job_description, job_config)
                parser_result = parser.parse_pdf(temp_pdf.name)
                
                # Combine results
                combined_result = {
                    "resume_analysis": resume_analysis,
                    "parser_result": parser_result
                }
                
                # Save to database
                application_id = supabase.save_application(merge_application_id, merge_job_id)
                supabase.save_analysis(application_id, combined_result)
                
                return combined_result

        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to download resume: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Resume analysis failed: {str(e)}")