import time
import os
from dotenv import load_dotenv
from hume import HumeClient
from hume.expression_measurement.batch.types import Models, Face, Prosody, Language

load_dotenv()

# Replace with your actual Hume API key
API_KEY = os.environ.get("HUME_API_KEY")

# Initialize the Hume Client
client = HumeClient(api_key=API_KEY)

def list_jobs(limit=None, status="IN_PROGRESS", when=None, timestamp_ms=None, sort_by=None, direction=None):
    """
    List jobs with optional filters and sorting.
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
        response = client.expression_measurement.batch.list_jobs(**job_payload)
        print(f"Listing {len(response)} jobs:")
        for job in response:
            print(f"Job ID: {job.job_id}, Request: {job.request}, Status: {job.state}, Type: {job.type}")
    except Exception as e:
        print(f"Error listing jobs: {e}")

def start_inference_job(urls, models, transactions=None, text=None, callback_url=None, notify=False):
    """
    Start a new measurement inference job.
    """
    job_payload = {
        "urls": urls,
        "models": Models(**models),
        "transactions": transactions,
        "text": text,
        "callback_url": callback_url,
        "notify": notify
    }
    print(job_payload.get('models'))
    
    # Remove keys with None values
    job_payload = {k: v for k, v in job_payload.items() if v is not None}
    
    try:
        job_id = client.expression_measurement.batch.start_inference_job(**job_payload)
        print(f"Started new job with ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Error starting job: {e}")
        return None

def get_job_details(job_id):
    """
    Get details of a specific job.
    """
    try:
        job = client.expression_measurement.batch.get_job_details(id=job_id)
        print(f"Job ID: {job.job_id}")
        print(f"Models: {job.request.models}")
        print(f"Status: {job.state.status}")
        print(f"Type: {job.type}")
        print(f"Created At: {job.state.created_timestamp_ms}")
        print(f"Started At: {job.state.started_timestamp_ms}")
        print(f"Ended At: {job.state.ended_timestamp_ms}")
    except Exception as e:
        print(f"Error fetching job details: {e}")

def get_job_predictions(job_id):
    """
    Get predictions of a completed job.
    """
    try:
        response = client.expression_measurement.batch.get_job_predictions(id=job_id)
        print(f"Predictions for Job ID: {job_id}")
        for prediction in response:
            print(f"Source: {prediction.source}")
            for result in prediction.results.predictions:
                print(f"  File: {result.file}")
                print(f"  Models: {result.models}")
                for face in result.models.face.grouped_predictions:
                    print(f"    Face ID: {face.id}")
                    for pred in face.predictions:
                        print(f"      Frame: {pred.frame}, Time: {pred.time}s, Probability: {pred.prob}")
                        print(f"      Emotions:")
                        for emotion in pred.emotions:
                            print(f"        {emotion.name}: {emotion.score}")
                # for file_pred in result.get('predictions', []):
                #     print(f"    File: {file_pred.file}")
                #     for face in file_pred.models.get('face', {}).get('grouped_predictions', []):
                #         print(f"      Face ID: {face.id}")
                #         for pred in face.predictions:
                #             print(f"        Frame: {pred.frame}, Time: {pred.time}s, Probability: {pred.prob}")
                #             print(f"        Emotions:")
                #             for emotion in pred.emotions:
                #                 print(f"          {emotion.name}: {emotion.score}")
    except Exception as e:
        print(f"Error fetching job predictions: {e}")

def monitor_job(job_id, interval=10):
    """
    Monitor the job status until it is completed or failed.
    """
    
    while True:
        try:
            job = client.expression_measurement.batch.get_job_details(id=job_id)
            status = job.state.status
            print(f"Job Status: {status}")
            if status in ["COMPLETED", "FAILED"]:
                break
            time.sleep(interval)
        except Exception as e:
            print(f"Error monitoring job: {e}")
            break

def main():
    # Example URLs to media files (replace with your own URLs)
    media_urls = [
        "https://hume-tutorials.s3.amazonaws.com/faces.zip"
    ]
    
    # Example text input
    text_inputs = [
        "I am so happy today!",
        "This is terrible."
    ]
    
    # Specify the models you want to use
    models = {
        "face": Face(),
        "prosody": Prosody(),
        "language": Language()
    }
    
    # Start a new inference job
    job_id = start_inference_job(
        urls=media_urls,
        models=models,
        text=text_inputs,
        callback_url=None,  # Optional: specify if you have a callback endpoint
    )
        
    if not job_id:
        return
    
    # Monitor the job until completion
    monitor_job(job_id)
    
    # Get job details
    get_job_details(job_id)
    
    # If job completed successfully, get predictions
    job = client.expression_measurement.batch.get_job_details(id=job_id)
    if job.state.status == "COMPLETED":
        get_job_predictions(job_id)
    else:
        print("Job did not complete successfully.")

    # List recent jobs
    list_jobs(limit=5)

if __name__ == "__main__":
    main()
