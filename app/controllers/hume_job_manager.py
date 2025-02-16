from typing import Dict, List, Optional, Any
import asyncio
import json
import os
import tempfile
from datetime import datetime
from colorama import Fore, Style, init
from hume import HumeClient, AsyncHumeClient
from app.configs.hume_config import HUME_API_KEY, HUME_API_SECRET
from app.controllers.supabase_db import SupabaseDB
from app.services.sentiment_analysis import EmotionAnalyzer
from tqdm import tqdm
# Initialize colorama for colored terminal output
init()

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

class JobManager:
    """
    A class to manage Hume AI emotion analysis jobs.
    
    This class handles starting jobs, monitoring their progress, and retrieving results
    using both synchronous and asynchronous Hume clients.
    """

    def __init__(self, client: HumeClient):
        """
        Initialize the JobManager with a HumeClient instance.
        
        Args:
            client (HumeClient): An initialized Hume client for making API calls
        """
        self.client = client
        self.async_client = None

    async def initialize_async_client(self):
        """
        Initialize the async Hume client if it hasn't been created yet.
        This is used for async operations like getting job details and predictions.
        """
        if not self.async_client:
            self.async_client = AsyncHumeClient(api_key=HUME_API_KEY)

    def start_job(self, urls: List[str], models: Dict, transcriptions: Optional[List] = None,
                 text: Optional[List[str]] = None, callback_url: Optional[str] = None, 
                 notify: bool = False) -> Optional[str]:
        """
        Start a new emotion analysis inference job.

        Args:
            urls (List[str]): List of media URLs to analyze
            models (Dict): Configuration for which emotion models to run
            transcriptions (Optional[List]): Optional transcriptions for the media
            text (Optional[List[str]]): Optional text to analyze
            callback_url (Optional[str]): URL to receive job completion webhook
            notify (bool): Whether to send email notifications

        Returns:
            Optional[str]: Job ID if successful, None if failed
        """
        # Build job configuration, filtering out None values
        job_payload = {
            "urls": urls,
            "models": models,
            "transcription": transcriptions,
            "text": text,
            "callback_url": callback_url,
            "notify": notify
        }
        job_payload = {k: v for k, v in job_payload.items() if v is not None}
        print(f"{Fore.CYAN}Starting job with payload:{Style.RESET_ALL}")
        print(job_payload)

        try:
            response = self.client.expression_measurement.batch.start_inference_job(**job_payload)
            print(f"{Fore.GREEN}Job started successfully{Style.RESET_ALL}")
            # print(response)
            return response
        except Exception as e:
            print(f"{Fore.RED}Error starting job: {e}{Style.RESET_ALL}")
            return None

    async def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details about a specific job's status and configuration.

        Args:
            job_id (str): The ID of the job to query

        Returns:
            Optional[Dict[str, Any]]: Job details if successful, None if failed
        """
        print(f"\n{'-'*40}{Fore.YELLOW} Getting Job Details {Style.RESET_ALL}{'-'*40}")
        print(f"{Fore.CYAN}Fetching details for job ID: {job_id}{Style.RESET_ALL}")

        if not self.async_client:
            await self.initialize_async_client()

        try:
            job = await self.async_client.expression_measurement.batch.get_job_details(id=job_id)
            # Example of structured response we could return instead of raw job data:
            # return {
            #     'job_id': job.job_id,
            #     'models': job.request.models,
            #     'type': job.type,
            #     'created_at': job.state.created_timestamp_ms,
            #     'started_at': job.state.started_timestamp_ms,
            #     'ended_at': job.state.ended_timestamp_ms,
            #     'status': job.state.status
            # }
            return job
        except Exception as e:
            print(f"{Fore.RED}Error fetching job details: {e}{Style.RESET_ALL}")
            return None

    async def wait_for_job_completion(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Monitor a job's progress until it completes or fails.
        Polls the job status every 10 seconds and prints progress updates.

        Args:
            job_id (str): The ID of the job to monitor

        Returns:
            Optional[Dict[str, Any]]: Job predictions if completed successfully, None if failed
        """
        print(f"{Fore.CYAN}Waiting for job completion...{Style.RESET_ALL}")
        start_time = datetime.now()
        
        while True:
            job_details = await self.get_job_details(job_id)
            if not job_details:
                return None
            
            status = job_details.state.status
            
            if status == "COMPLETED":
                print(f"{Fore.GREEN}Analysis complete{Style.RESET_ALL}")
                result = await self.get_job_predictions(job_id)
                
                return result
            elif status == "FAILED":
                print(f"{Fore.RED}Job failed{Style.RESET_ALL}")
                return None
            else:
                elapsed_time = (datetime.now() - start_time)
                print(f"{Fore.YELLOW}Status: {status}, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Elapsed Time: {elapsed_time}{Style.RESET_ALL}")
                await asyncio.sleep(10)

    async def get_job_predictions(self, job_id: str):
        """
        Retrieve the emotion analysis predictions for a completed job.

        Args:
            job_id (str): The ID of the completed job

        Returns:
            Optional[Dict]: Prediction results if successful, None if failed
        """
        if not self.async_client:
            await self.initialize_async_client()

        try:
            result = await self.async_client.expression_measurement.batch.get_job_predictions(id=job_id)
            

            return result
        except Exception as e:
            print(f"{Fore.RED}Error fetching job predictions: {e}{Style.RESET_ALL}")
            return None

    def list_jobs(self, limit: Optional[int] = None, status: str = "IN_PROGRESS",
                 when: Optional[str] = None, timestamp_ms: Optional[int] = None,
                 sort_by: Optional[str] = None, direction: Optional[str] = None) -> List[Dict]:
        """
        List jobs with optional filtering and sorting parameters.

        Args:
            limit (Optional[int]): Maximum number of jobs to return
            status (str): Filter by job status (default: "IN_PROGRESS")
            when (Optional[str]): Time filter specification
            timestamp_ms (Optional[int]): Timestamp for time-based filtering
            sort_by (Optional[str]): Field to sort results by
            direction (Optional[str]): Sort direction ("asc" or "desc")

        Returns:
            List[Dict]: List of job information dictionaries
        """
        job_payload = {
            "limit": limit,
            "status": status,
            "when": when,
            "timestamp_ms": timestamp_ms,
            "sort_by": sort_by,
            "direction": direction
        }
        job_payload = {k: v for k, v in job_payload.items() if v is not None}

        try:
            response = self.client.expression_measurement.batch.list_jobs(**job_payload)
            return [{
                'job_id': job.job_id,
                'request': job.request,
                'status': job.state,
                'type': job.type
            } for job in response]
        except Exception as e:
            print(f"{Fore.RED}Error listing jobs: {e}{Style.RESET_ALL}")
            return []

async def main():
    """
    Example usage of the JobManager class that processes a test interview:
    1. Gets a file URL from Supabase storage
    2. Runs emotion analysis
    3. Processes results using EmotionAnalyzer
    4. Stores processed results in Supabase storage
    """
    # Initialize Supabase client and get test file
    supabase_db = SupabaseDB()
    files = supabase_db.list_storage_files(
        bucket_id="Interviews",
        folder_path="Test",
        # limit=2
    )

    if not files:
        print(f"{Fore.RED}No files found in Test folder{Style.RESET_ALL}")
        return
    
    
    urls = supabase_db.create_signed_url(
        bucket_id="Interviews",
        files=files
    )

    if not urls:
        print(f"{Fore.RED}Failed to create signed URL{Style.RESET_ALL}")
        return

    # Initialize clients and services
    hume_client = HumeClient(api_key=HUME_API_KEY)
    job_manager = JobManager(hume_client)
    emotion_analyzer = EmotionAnalyzer(HUME_API_KEY)

    # job_id = "5cc410a0-be08-492e-b72b-b1ac3e58ac51"

    # Start emotion analysis job
    print(f"{Fore.CYAN}Starting Hume analysis job...{Style.RESET_ALL}")
    job_id = job_manager.start_job(
        urls=urls,
        models={
            "face": {},
            "language": {},
            "prosody": {}
        }
    )

    if not job_id:
        print(f"{Fore.RED}Failed to start Hume job{Style.RESET_ALL}")
        return

    # Wait for and process results
    raw_results = await job_manager.wait_for_job_completion(job_id)
    if not raw_results:
        print(f"{Fore.RED}No results received from Hume job{Style.RESET_ALL}")
        return

    print(f"{Fore.GREEN}Processing emotion analysis results...{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Number of results: {len(raw_results)}{Style.RESET_ALL}")
    
    try:
        # Process emotions using EmotionAnalyzer
        with tqdm(total=1, desc="Processing and uploading", unit="file") as pbar:
            for i in range(len(raw_results)):
                result = raw_results[i]
                file_path = result.results.predictions[0].file

            
                processed_results = emotion_analyzer.process_predictions(result)
                
                # Add metadata
                final_results = {
                    'job_id': job_id,
                    'timestamp': datetime.now().isoformat(),
                    'file_analyzed': file_path,
                    'results': processed_results
                }

                # Convert to JSON with proper formatting
                json_results = json.dumps(final_results, indent=2)

                # Get original filename without extension and add timestamp
                original_filename = os.path.splitext(os.path.basename(file_path))[0]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{original_filename}_{timestamp}.json"

                print(f"{Fore.CYAN}Saving results to temporary file...{Style.RESET_ALL}")
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                    temp_file.write(json_results)
                    temp_filepath = temp_file.name

                print(f"{Fore.CYAN}Uploading processed results to storage...{Style.RESET_ALL}")
                
                # Upload temporary file to Supabase storage
                with open(temp_filepath, 'rb') as file:
                    upload_result = supabase_db.upload_file(
                        bucket_id="Hume Output",
                        destination_path=filename,
                        file_data=file
                    )

                # Clean up temporary file
                os.unlink(temp_filepath)

               
    except Exception as e:
        print(f"{Fore.RED}Error processing or uploading results: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main())