import time
import os
from dotenv import load_dotenv
from hume import HumeClient

load_dotenv()

# Replace with your actual Hume API key
API_KEY = os.environ.get("HUME_API_KEY")

# Initialize the Hume Client
client = HumeClient(api_key=API_KEY)

# Define weights for emotions
EMOTION_WEIGHTS = {
    # Positive Emotions
    "Admiration": 0.5,
    "Adoration": 0.5,
    "Aesthetic Appreciation": 0.4,
    "Amusement": 0.6,
    "Awe": 0.5,
    "Calmness": 0.7,
    "Concentration": 0.6,
    "Contemplation": 0.4,
    "Contentment": 0.7,
    "Determination": 0.6,
    "Enthusiasm": 0.6,
    "Interest": 0.7,
    "Joy": 0.8,
    "Love": 0.5,
    "Nostalgia": 0.4,
    "Realization": 0.5,
    "Triumph": 0.6,

    # Negative Emotions
    "Anger": -0.6,
    "Annoyance": -0.5,
    "Anxiety": -0.5,
    "Awkwardness": -0.4,
    "Boredom": -0.5,
    "Confusion": -0.6,
    "Contempt": -0.6,
    "Disappointment": -0.5,
    "Disapproval": -0.6,
    "Disgust": -0.7,
    "Distress": -0.5,
    "Doubt": -0.5,
    "Embarrassment": -0.4,
    "Empathic Pain": -0.3,
    "Fear": -0.6,
    "Guilt": -0.5,
    "Horror": -0.7,
    "Sadness": -0.6,
    "Shame": -0.5,
    "Surprise (negative)": -0.4,
    "Surprise (positive)": 0.3,  # Mixed, slight positive
    "Sympathy": 0.2,  # Slightly positive
    "Tiredness": -0.3,

    # Neutral or Context-Dependent Emotions
    "Craving": 0.0,
    "Desire": 0.0,
    "Excitement": 0.4,
    "Pain": -0.4,
    "Satisfaction": 0.5,
    "Romance": 0.0,  # Context-dependent
    # Add more emotions as needed
}

def aggregate_emotion_score(emotion_scores, weights=EMOTION_WEIGHTS):
    """
    Aggregates emotion scores into a single score out of 10.

    :param emotion_scores: Dictionary of emotion names to their averaged scores.
    :param weights: Dictionary of emotion weights.
    :return: Aggregated score out of 10.
    """
    total = 0.0
    max_possible = 0.0
    min_possible = 0.0

    for emotion, score in emotion_scores.items():
        weight = weights.get(emotion, 0.0)
        total += weight * score
        if weight > 0:
            max_possible += weight  # Max positive contribution
        elif weight < 0:
            min_possible += weight  # Max negative contribution

    # Normalize the total to a 0-10 scale
    # Assuming total ranges between min_possible and max_possible
    if max_possible - min_possible == 0:
        normalized_score = 5  # Neutral score if no variance
    else:
        normalized_score = 10 * (total - min_possible) / (max_possible - min_possible)

    # Clamp the score between 0 and 10
    normalized_score = max(0, min(10, normalized_score))

    return round(normalized_score, 2)

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

def start_inference_job(urls, models, transcriptions=None, text=None, callback_url=None, notify=False):
    """
    Start a new measurement inference job.
    """
    job_payload = {
        "urls": urls,
        "models": models,
        "transcription": transcriptions,
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
        print(f"Type: {job.type}")
        print(f"Created At: {job.state.created_timestamp_ms}")
        print(f"Started At: {job.state.started_timestamp_ms}")
        print(f"Ended At: {job.state.ended_timestamp_ms}")
    except Exception as e:
        print(f"Error fetching job details: {e}")

def get_job_predictions(job_id):
    """
    Get predictions of a completed job and calculate aggregate emotion score across the entire video.
    """
    try:
        response = client.expression_measurement.batch.get_job_predictions(id=job_id)
        print(f"Predictions for Job ID: {job_id}")

        # Initialize a dictionary to accumulate emotion scores
        face_accumulated_emotions = {}
        prosody_accumulated_emotions = {}

        # Initialize a counter for the number of frames processed
        frame_count = 0

        for prediction in response:
            print(f"Source: {prediction.source}")
            for result in prediction.results.predictions:
                # Process Face Model Predictions
                if result.models.face:
                    for face in result.models.face.grouped_predictions:
                        for pred in face.predictions:
                            emotions = {emotion.name: emotion.score for emotion in pred.emotions}
                            for emotion, score in emotions.items():
                                face_accumulated_emotions[emotion] = face_accumulated_emotions.get(emotion, 0.0) + score
                            frame_count += 1

                # Process Prosody Model Predictions
                if result.models.prosody:
                    for prosody in result.models.prosody.grouped_predictions:
                        for pred in prosody.predictions:
                            emotions = {emotion.name: emotion.score for emotion in pred.emotions}
                            for emotion, score in emotions.items():
                                prosody_accumulated_emotions[emotion] = prosody_accumulated_emotions.get(emotion, 0.0) + score

                # # Process Language Model Predictions
                # if result.models.language:
                #     for lang_pred in result.models.language.grouped_predictions:
                #         for language in lang_pred.predictions:
                #             emotions = {emotion.name: emotion.score for emotion in language.emotions}
                #             for emotion, score in emotions.items():
                #                 accumulated_emotions[emotion] = accumulated_emotions.get(emotion, 0.0) + score
                #             frame_count += 1

        if frame_count == 0:
            print("No emotion data found in the predictions.")
            return

        # Calculate average emotion scores
        face_average_emotions = {emotion: total_score / frame_count for emotion, total_score in face_accumulated_emotions.items()}
        prosody_average_emotions = {emotion: total_score / frame_count for emotion, total_score in prosody_accumulated_emotions.items()}

        print(f"Processed {frame_count} frames.")
        print("Average Face Emotion Scores:")
        for emotion, avg_score in face_average_emotions.items():
            print(f"  {emotion}: {avg_score:.4f}")
        print("Average Prosody Emotion Scores:")
        for emotion, avg_score in prosody_average_emotions.items():
            print(f"  {emotion}: {avg_score:.4f}")

        # Calculate the aggregate emotion score
        face_aggregate_score = aggregate_emotion_score(face_average_emotions)
        prosody_aggregate_score = aggregate_emotion_score(prosody_average_emotions)
        print(f"\nAggregate Face Emotion Score for the Video: {face_aggregate_score}/10")
        print(f"\nAggregate Prosody Emotion Score for the Video: {prosody_aggregate_score}/10")

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

<<<<<<< HEAD:app/test/hume_test.py
def main():
    # Example URLs to media files (replace with your own URLs)
    media_urls = [
        "https://www.dropbox.com/scl/fi/eesz1rkgpxjdl81m9qu51/P7.avi?rlkey=w7ewszrtzf9m64x1yx1ajzobp&st=tsk6loki&dl=1"
    ]

    # Example text input
    text_inputs = [
        "I am so happy today!",
        "This is terrible."
    ]

    # Specify the models you want to use
    models = {
        "face": {},
        "language": {},
        "prosody": {}
    }

=======
def analyze(text_inputs=list(str), models={}, media_urls =""):
    
>>>>>>> feature/deploy:app/services/sentiment_analysis .py
    # Start a new inference job
    job_id = start_inference_job(
        urls=media_urls,
        models=models,
<<<<<<< HEAD:app/test/hume_test.py
        # text=text_inputs,
=======
        text=text_inputs,
>>>>>>> feature/deploy:app/services/sentiment_analysis .py
        callback_url=None,  # Optional: specify if you have a callback endpoint
    )

    if not job_id:
        return

    # Monitor the job until completion
    monitor_job(job_id)

    # Get job details
    print("Getting job details...")
    get_job_details(job_id)

    # If job completed successfully, get predictions and aggregate emotions
    job = client.expression_measurement.batch.get_job_details(id=job_id)
    if job.state.status == "COMPLETED":
        get_job_predictions(job_id)
    else:
        print("Job did not complete successfully.")

    # List recent jobs
    list_jobs(limit=5)


def main():
    media_urls = [
        "https://www.dropbox.com/scl/fi/eesz1rkgpxjdl81m9qu51/P7.avi?rlkey=w7ewszrtzf9m64x1yx1ajzobp&st=tsk6loki&dl=0"
    ]

    models = {
        "face": {},
        "language": {}
    }

    analyze(media_urls=media_urls, models=models)





