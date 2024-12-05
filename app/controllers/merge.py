from typing import Dict, Optional
from dataclasses import dataclass
import requests
import json


@dataclass
class JobData:
    id: str
    status: str
    error: Optional[str] = None
    data: Optional[Dict] = None

class MergeAPIClient:
    def __init__(self, account_token: str, bearer_token: str, base_url: str = "https://api.merge.dev/api/ats/v1"):
        self.account_token = account_token
        self.bearer_token = bearer_token
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json',
            'X-Account-Token': self.account_token,
            'Authorization': f'Bearer {self.bearer_token}'
        }

    def get_job(self, job_id: str, include_remote_data: bool = False) -> Dict:
        """
        Get a job by ID from Merge API
        
        Args:
            job_id: The ID of the job to retrieve
            include_remote_data: Whether to include the original third-party data
            
        Returns:
            Dict containing the job data
        """
        endpoint = f"{self.base_url}/jobs/{job_id}"
        
        params = {}
        if include_remote_data:
            params['include_remote_data'] = str(include_remote_data).lower()

        # Print request details for debugging
        print("\nSubmitting request with:")
        print(f"Endpoint: {endpoint}")
        print("Headers:", json.dumps(self.headers, indent=2))
        print("Params:", json.dumps(params, indent=2))
        
        response = requests.get(endpoint, headers=self.headers, params=params)
        if response.status_code != 200:
            print(f"\nError Response Status: {response.status_code}")
            print("Response Headers:", json.dumps(dict(response.headers), indent=2))
            print("Response Body:", response.text)
            
        response.raise_for_status()
        return response.json()

    def get_jobs(self, 
                 created_after: Optional[str] = None,
                 created_before: Optional[str] = None,
                 cursor: Optional[str] = None,
                 include_remote_data: bool = False,
                 page_size: Optional[int] = None) -> Dict:
        """
        Get a list of jobs with optional filters
        
        Args:
            created_after: Only return jobs created after this datetime
            created_before: Only return jobs created before this datetime
            cursor: The pagination cursor value
            include_remote_data: Whether to include original third-party data
            page_size: Number of results to return per page
            
        Returns:
            Dict containing list of jobs and pagination information
        """
        endpoint = f"{self.base_url}/jobs"
        
        params = {k: v for k, v in {
            'created_after': created_after,
            'created_before': created_before,
            'cursor': cursor,
            'include_remote_data': str(include_remote_data).lower() if include_remote_data else None,
            'page_size': page_size
        }.items() if v is not None}

        response = requests.get(endpoint, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def process_job_data(self, job_id: str) -> JobData:
        """
        Process a complete job data retrieval workflow
        
        Args:
            job_id: The ID of the job to process
            
        Returns:
            JobData object containing processed job information
        """
        try:
            # Get job data
            job_data = self.get_job(job_id, include_remote_data=True)
            
            # Create JobData object
            return JobData(
                id=job_data.get('id', ''),
                status='success',
                data=job_data
            )
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error occurred: {e}")
            return JobData(
                id=job_id,
                status='error',
                error=f"HTTP Error - {str(e)}"
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            return JobData(
                id=job_id,
                status='error',
                error=str(e)
            )