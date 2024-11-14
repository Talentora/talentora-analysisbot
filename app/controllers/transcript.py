import requests
import time
import os
from dotenv import load_dotenv
from typing import Dict, Optional
from dataclasses import dataclass

load_dotenv()

@dataclass
class JobStatus:
    id: str
    status: str
    error: Optional[str] = None
    output: Optional[Dict] = None

class DailyBatchProcessor:
    def __init__(self, api_token: str, base_url: str = "https://api.daily.co/v1"):
        self.api_token = api_token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

    def submit_transcript_job(self, recording_id: str) -> Dict:
        """
        Submit a job to transcribe an MP4 file using its recording ID.
        
        Args:
            recording_id: The ID of the recording in Daily's database
            
        Returns:
            Dict containing the job ID and other response data
        """
        endpoint = f"{self.base_url}/batch-processor"
        
        payload = {
            "preset": "transcript",
            "inParams": {
                "sourceType": "recordingId",
                "recordingId": recording_id
            },
            "outParams": {
                "s3Config": {
                    "s3KeyTemplate": "transcript"
                }
            }
        }
        
        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get the current status of a job.
        
        Args:
            job_id: The ID of the submitted job
            
        Returns:
            JobStatus object containing status information
        """
        endpoint = f"{self.base_url}/batch-processor/{job_id}"
        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        
        return JobStatus(
            id=data["id"],
            status=data["status"],
            error=data.get("error"),
            output=data.get("output")
        )

    def get_access_links(self, job_id: str) -> Dict:
        """
        Get the download links for a completed job.
        
        Args:
            job_id: The ID of the completed job
            
        Returns:
            Dict containing the access links for the outputs
        """
        endpoint = f"{self.base_url}/batch-processor/{job_id}/access-link"
        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, job_id: str, max_attempts: int = 60, delay: int = 10) -> Optional[Dict]:
        """
        Poll the job status until it completes or reaches maximum attempts.
        
        Args:
            job_id: The ID of the submitted job
            max_attempts: Maximum number of polling attempts
            delay: Delay in seconds between polling attempts
            
        Returns:
            Dict containing the output access links if successful, None if timed out
        """
        for attempt in range(max_attempts):
            job_status = self.get_job_status(job_id)
            
            if job_status.status == "finished":
                return self.get_access_links(job_id)
            elif job_status.status == "error":
                raise Exception(f"Job failed with error: {job_status.error}")
            
            # Print progress update
            print(f"Job status: {job_status.status} (attempt {attempt + 1}/{max_attempts})")
            
            time.sleep(delay)
        return None

def process_transcription_job(processor: DailyBatchProcessor, recording_id: str):
    """
    Process a complete transcription job workflow.
    """
    try:
        # Submit the transcription job
        print("Submitting transcription job...")
        job_response = processor.submit_transcript_job(recording_id)
        job_id = job_response["id"]
        print(f"Job submitted successfully. Job ID: {job_id}")
        
        # Wait for job completion and get access links
        print("Waiting for job completion...")
        output = processor.wait_for_completion(job_id)
        
        if output:
            print("\nTranscription completed successfully!")
            print("\nAvailable transcripts:")
            for transcript in output["transcription"]:
                print(f"Format: {transcript['format']}")
                print(f"Download link: {transcript['link']}\n")
            
            # Print S3 information if available
            job_status = processor.get_job_status(job_id)
            if job_status.output and "transcription" in job_status.output:
                print("\nS3 Storage Information:")
                for transcript in job_status.output["transcription"]:
                    print(f"\nFormat: {transcript['format']}")
                    print(f"S3 Bucket: {transcript['s3Config']['bucket']}")
                    print(f"S3 Key: {transcript['s3Config']['Key']}")
                    print(f"Region: {transcript['s3Config']['region']}")
        else:
            print("Job timed out. Please check the job status manually.")
            
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    
    api_token = os.environ.get("DAILY_API_KEY")
    
    recording_id = "YOUR_RECORDING_ID"
    
    processor = DailyBatchProcessor(api_token)
    process_transcription_job(processor, recording_id)

if __name__ == "__main__":
    main()