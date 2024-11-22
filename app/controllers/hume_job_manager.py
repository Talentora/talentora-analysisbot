from typing import Dict, List, Optional
import time
from hume import HumeClient
from app.configs.hume_config import HUME_API_KEY

class JobManager:
    def __init__(self, client: HumeClient):
        """Initialize the JobManager with a HumeClient instance."""
        self.client = client

    def start_job(self, urls: List[str], models: Dict, transcriptions: Optional[List] = None,
                 text: Optional[List[str]] = None, callback_url: Optional[str] = None, 
                 notify: bool = False) -> Optional[str]:
        """Start a new measurement inference job."""
        job_payload = {
            "urls": urls,
            "models": models,
            "transcription": transcriptions,
            "text": text,
            "callback_url": callback_url,
            "notify": notify
        }
        job_payload = {k: v for k, v in job_payload.items() if v is not None}

        try:
            return self.client.expression_measurement.batch.start_inference_job(**job_payload)
        except Exception as e:
            print(f"Error starting job: {e}")
            return None

    def monitor_job(self, job_id: str, interval: int = 10) -> str:
        """Monitor the job status until completion."""
        while True:
            try:
                job = self.client.expression_measurement.batch.get_job_details(id=job_id)
                status = job.state.status
                print(f"Job Status: {status}")
                if status in ["COMPLETED", "FAILED"]:
                    return status
                time.sleep(interval)
            except Exception as e:
                print(f"Error monitoring job: {e}")
                return "FAILED"

    def get_job_details(self, job_id: str) -> Optional[Dict]:
        """Get details of a specific job."""
        try:
            job = self.client.expression_measurement.batch.get_job_details(id=job_id)
            return {
                'job_id': job.job_id,
                'models': job.request.models,
                'type': job.type,
                'created_at': job.state.created_timestamp_ms,
                'started_at': job.state.started_timestamp_ms,
                'ended_at': job.state.ended_timestamp_ms,
                'status': job.state.status
            }
        except Exception as e:
            print(f"Error fetching job details: {e}")
            return None

    def get_job_predictions(self, job_id: str):
        """Get predictions for a completed job."""
        try:
            return self.client.expression_measurement.batch.get_job_predictions(id=job_id)
        except Exception as e:
            print(f"Error fetching job predictions: {e}")
            return None

    def list_jobs(self, limit: Optional[int] = None, status: str = "IN_PROGRESS",
                 when: Optional[str] = None, timestamp_ms: Optional[int] = None,
                 sort_by: Optional[str] = None, direction: Optional[str] = None) -> List[Dict]:
        """List jobs with optional filters and sorting."""
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
            print(f"Error listing jobs: {e}")
            return []