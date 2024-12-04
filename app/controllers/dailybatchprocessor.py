import requests
import time
import os
import json
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

    def submit_batch_processor_job(self, recording_id: str) -> Dict:
        """
        Submit a job to transcribe an MP4 file using its recording ID.
        
        Args:
            recording_id: The ID of the recording in Daily's database
            
        Returns:
            Dict containing the job ID and other response data
        """
        endpoint = f"{self.base_url}/batch-processor"
        
        payload = {
            "preset": "summarize",
            "inParams": {
                "sourceType": "recordingId",
                "recordingId": recording_id
            },
            "outParams": {
                "s3Config": {
                    "s3KeyTemplate": "summary"
                }
            }
        }
        # Print the request details for debugging
        print("\nSubmitting request with:")
        print(f"Endpoint: {endpoint}")
        print("Headers:", json.dumps(self.headers, indent=2))
        print("Payload:", json.dumps(payload, indent=2))
        response = requests.post(endpoint, headers=self.headers, json=payload)
        if response.status_code != 200:
            print(f"\nError Response Status: {response.status_code}")
            print("Response Headers:", json.dumps(dict(response.headers), indent=2))
            print("Response Body:", response.text)
            
        response.raise_for_status()
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


    def process_transcription_job(self, job_id: str):
        """
        Process a complete transcription job workflow.
        """
        try:
            # Wait for job completion and get access links
            output = self.get_access_links(job_id)
            print("sucessfully got access link")
            
            # if output:
            #     print("\nTranscription completed successfully!")
            #     print("\nAvailable transcripts:")
            #     for transcript in output["transcription"]:
            #         print(f"Format: {transcript['format']}")
            #         print(f"Download link: {transcript['link']}\n")
                
            #     # Print S3 information if available
            #     job_status = processor.get_job_status(job_id)
            #     if job_status.output and "transcription" in job_status.output:
            #         print("\nS3 Storage Information:")
            #         for transcript in job_status.output["transcription"]:
            #             print(f"\nFormat: {transcript['format']}")
            #             print(f"S3 Bucket: {transcript['s3Config']['bucket']}")
            #             print(f"S3 Key: {transcript['s3Config']['key']}")
            #             print(f"Region: {transcript['s3Config']['region']}")
            # else:
            #     print("Job timed out. Please check the job status manually.")
            
            if output and "transcription" in output:
                # Find the TXT format transcript
                txt_transcript = next(
                    (t for t in output["transcription"] if t["format"] == "txt"),
                    None
                )
                
                if txt_transcript:
                    # Get the download link for the TXT file
                    download_link = txt_transcript["link"]
                    print("Downloading transcript text...")
                    
                    # Download and split into lines
                    response = requests.get(download_link)
                    response.raise_for_status()
                    # Split text into lines and remove empty lines
                    transcript_lines = [
                        line.strip() 
                        for line in response.text.split('\n') 
                        if line.strip()
                    ]
                    return transcript_lines
                else:
                    return ["Error: No TXT format transcript found"]
            else:
                return ["Error: Job timed out or failed to complete"]
                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error occurred: {e}")
            return [f"Error: HTTP Error - {str(e)}"]
        except Exception as e:
            print(f"An error occurred: {e}")
            return [f"Error: {str(e)}"]
        
    def process_summary_job(self, job_id: str):
        """
        Process a complete transcription job workflow.
        """
        try:
            # Wait for job completion and get access links
            output = self.get_access_links(job_id)
            print("sucessfully got access link")
            
            if output and "summary" in output:
                # Find the TXT format transcript
                txt_summary = next(
                    (t for t in output["summary"] if t["format"] == "txt"),
                    None
                )
                
                if txt_summary:
                    # Get the download link for the TXT file
                    download_link = txt_summary["link"]
                    print("Downloading summary text...")
                    
                    # Download and split into lines
                    response = requests.get(download_link)
                    response.raise_for_status()
                    # Split text into lines and remove empty lines
                    summary_lines = [
                        line.strip() 
                        for line in response.text.split('\n') 
                        if line.strip()
                    ]
                    print("".join(summary_lines))
                    return "".join(summary_lines)
                else:
                    return "Error: No TXT format summary found"
            else:
                return "Error: Job timed out or failed to complete"
                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error occurred: {e}")
            return f"Error: HTTP Error - {str(e)}"
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"Error: {str(e)}"

def main():
    api_key = os.environ.get("DAILY_API_KEY")
    if not api_key:
        print("Error: DAILY_API_KEY not found in environment variables")
        print("Please ensure you have created a .env file with your API key")
        return
            
    recording_id = "cf6bcc01-14ac-48d5-9473-bbc516522e1c"
    
    processor = DailyBatchProcessor(api_key)
    job_response = processor.submit_batch_processor_job(recording_id)
    job_id = job_response["id"]
    print(processor.process_transcription_job(job_id))

if __name__ == "__main__":
    main()