import time
import os
from dotenv import load_dotenv
from hume import HumeClient
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import cv2
import requests

load_dotenv()

API_KEY = os.environ.get("HUME_API_KEY")

client = HumeClient(api_key=API_KEY)

def get_job_predictions(job_id):
    """
    Get predictions of a completed job and return structured data.
    """
    try:
        response = client.expression_measurement.batch.get_job_predictions(id=job_id)
        
        face_predictions = []
        prosody_predictions = []
        language_predictions = []
        
        for prediction in response:
            # Face Model Predictions
            if prediction.results.predictions[0].models.face:
                for face in prediction.results.predictions[0].models.face.grouped_predictions:
                    for pred in face.predictions:
                        face_pred = {
                            'face_id': face.id,
                            'frame': pred.frame,
                            'time': pred.time,
                            'probability': pred.prob,
                            'emotions': {emotion.name: emotion.score for emotion in pred.emotions}
                        }
                        face_predictions.append(face_pred)
            
            # Prosody Model Predictions
            if prediction.results.predictions[0].models.prosody:
                for prosody in prediction.results.predictions[0].models.prosody.grouped_predictions:
                    for pred in prosody.predictions:
                        prosody_pred = {
                            'time': pred.time.begin,
                            'text': pred.text,
                            'confidence': pred.confidence,
                            'speaker_confidence': pred.speaker_confidence,
                            'emotions': {emotion.name: emotion.score for emotion in pred.emotions}
                        }
                        prosody_predictions.append(prosody_pred)
            
            # Language Model Predictions
            if prediction.results.predictions[0].models.language:
                for language in prediction.results.predictions[0].models.language.grouped_predictions:
                    for pred in language.predictions:
                        language_pred = {
                            'time': pred.time.begin,
                            'text': pred.text,
                            'confidence': pred.confidence,
                            'speaker_confidence': pred.speaker_confidence,
                            'emotions': {emotion.name: emotion.score for emotion in pred.emotions}
                        }
                        language_predictions.append(language_pred)
        
        return {
            'face': face_predictions,
            'prosody': prosody_predictions,
            'language': language_predictions
        }
    
    except Exception as e:
        print(f"Error fetching job predictions: {e}")
        return None
    
def visualize_predictions(predictions):
    """
    Create visualizations from the Hume API predictions.

    :param predictions: Dictionary containing face, prosody, and language predictions.
    """
    # Convert predictions to DataFrames
    face_df = pd.DataFrame(predictions['face'])
    prosody_df = pd.DataFrame(predictions['prosody'])
    language_df = pd.DataFrame(predictions['language'])

    # 1. Emotion Over Time (Face Model)
    if not face_df.empty:
        # Melt the emotions dictionary to separate columns
        face_emotions = face_df[['time', 'emotions']].copy()
        face_emotions = face_emotions.explode('emotions')
        face_emotions = pd.concat([
            face_emotions.drop(['emotions'], axis=1),
            face_emotions['emotions'].apply(pd.Series)
        ], axis=1)
        face_emotions = face_emotions.melt(id_vars=['time'], var_name='emotion', value_name='score')
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=face_emotions, x='time', y='score', hue='emotion')
        plt.title('Face Emotions Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Emotion Score')
        plt.legend(title='Emotion')
        plt.tight_layout()
        plt.savefig('face_emotions_over_time.png')
        plt.show()

    # 2. Emotion Distribution (Prosody Model)
    if not prosody_df.empty:
        # Aggregate emotions
        prosody_emotions = prosody_df['emotions'].apply(pd.Series).fillna(0)
        prosody_emotions = prosody_emotions.sum().reset_index()
        prosody_emotions.columns = ['emotion', 'total_score']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=prosody_emotions, x='emotion', y='total_score', palette='viridis')
        plt.title('Prosody Emotion Distribution')
        plt.xlabel('Emotion')
        plt.ylabel('Total Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('prosody_emotion_distribution.png')
        plt.show()

    # 3. Speech Confidence Over Time (Prosody Model)
    if not prosody_df.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=prosody_df, x='time', y='confidence', label='Confidence')
        sns.lineplot(data=prosody_df, x='time', y='speaker_confidence', label='Speaker Confidence')
        plt.title('Speech Confidence Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Confidence Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig('speech_confidence_over_time.png')
        plt.show()

    # 4. Language Emotion Over Time
    if not language_df.empty:
        language_emotions = language_df[['time', 'emotions']].copy()
        language_emotions = language_emotions.explode('emotions')
        language_emotions = pd.concat([
            language_emotions.drop(['emotions'], axis=1),
            language_emotions['emotions'].apply(pd.Series)
        ], axis=1)
        language_emotions = language_emotions.melt(id_vars=['time'], var_name='emotion', value_name='score')
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=language_emotions, x='time', y='score', hue='emotion')
        plt.title('Language Emotions Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Emotion Score')
        plt.legend(title='Emotion')
        plt.tight_layout()
        plt.savefig('language_emotions_over_time.png')
        plt.show()

    # 5. Interactive Emotion Over Time (Using Plotly)
    if not face_df.empty:
        fig = px.line(face_emotions, x='time', y='score', color='emotion', title='Face Emotions Over Time')
        fig.write_html('face_emotions_over_time.html')
        fig.show()
        
def visualize_top_emotions(predictions, top_n=5, window_size=3):
    """
    Create visualizations for the top N emotions from Hume API predictions with smoothing.

    :param predictions: Dictionary containing face, prosody, and language predictions.
    :param top_n: Number of top emotions to visualize.
    :param window_size: Window size for moving average smoothing.
    """
    # Convert predictions to DataFrames
    face_df = pd.DataFrame(predictions['face'])
    prosody_df = pd.DataFrame(predictions['prosody'])
    language_df = pd.DataFrame(predictions['language'])
    
    print(face_df.head())
    print(face_df['emotions'].head())

    # Top Emotion Over Time (Face Model)
    if not face_df.empty:

        face_emotions = face_df[['time', 'emotions']].copy()
        
        face_emotions = face_emotions[face_emotions['emotions'].apply(lambda x: isinstance(x, dict))]
        
        emotions_normalized = pd.json_normalize(face_emotions['emotions'])

        # concatenate with reset index
        face_emotions = pd.concat([
            face_emotions.drop(['emotions'], axis=1).reset_index(drop=True),
            emotions_normalized.reset_index(drop=True)
        ], axis=1)

        # melt the DataFrame to have one row per emotion per time point
        face_emotions = face_emotions.melt(
            id_vars=['time'], 
            var_name='emotion', 
            value_name='score'
        )

        # identify the top emotion based on average score
        top_emotions = face_emotions.groupby('emotion')['score'].mean().nlargest(top_n).index.tolist()
        print(f"Top {top_n} emotions: {top_emotions}")

        top_face_emotions = face_emotions[face_emotions['emotion'].isin(top_emotions)]
        
        # sort the data by time
        top_face_emotions = top_face_emotions.sort_values(by='time')
        
        top_face_emotions['smoothed_score'] = top_face_emotions.groupby('emotion')['score'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )

        plt.figure(figsize=(14, 8))
        sns.lineplot(
            data=top_face_emotions, 
            x='time', 
            y='smoothed_score', 
            hue='emotion',
            marker='o'
        )
        plt.title(f'Top {top_n} Emotions Over Time (Smoothed)')
        plt.xlabel('Time (s)')
        plt.ylabel('Emotion Score (Smoothed)')
        plt.legend(title='Emotion')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('top_emotions_over_time_smoothed.png')
        plt.show()

def visualize_top_emotions_interactive(predictions, top_n=5, window_size=3):
    """
    Create interactive visualizations for the top N emotions with smoothing using Plotly.

    :param predictions: Dictionary containing face, prosody, and language predictions.
    :param top_n: Number of top emotions to visualize.
    :param window_size: Window size for moving average smoothing.
    """
    # Convert predictions to DataFrames
    face_df = pd.DataFrame(predictions['face'])
    prosody_df = pd.DataFrame(predictions['prosody'])
    language_df = pd.DataFrame(predictions['language'])

    if not face_df.empty:
        face_emotions = face_df[['time', 'emotions']].copy()
        face_emotions = face_emotions[face_emotions['emotions'].apply(lambda x: isinstance(x, dict))]
        emotions_normalized = pd.json_normalize(face_emotions['emotions'])
        face_emotions = pd.concat([
            face_emotions.drop(['emotions'], axis=1).reset_index(drop=True),
            emotions_normalized.reset_index(drop=True)
        ], axis=1)
        face_emotions = face_emotions.melt(
            id_vars=['time'], 
            var_name='emotion', 
            value_name='score'
        )
        top_emotions = face_emotions.groupby('emotion')['score'].mean().nlargest(top_n).index.tolist()
        top_face_emotions = face_emotions[face_emotions['emotion'].isin(top_emotions)]
        top_face_emotions = top_face_emotions.sort_values(by='time')
        top_face_emotions['smoothed_score'] = top_face_emotions.groupby('emotion')['score'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )

        # Create interactive plot
        fig = px.line(
            top_face_emotions, 
            x='time', 
            y='smoothed_score', 
            color='emotion',
            markers=True,
            title=f'Top {top_n} Emotions Over Time (Smoothed)'
        )
        fig.update_layout(
            xaxis_title='Time (s)',
            yaxis_title='Emotion Score (Smoothed)',
            legend_title='Emotion',
            template='plotly_white'
        )
        fig.write_html('top_emotions_over_time_smoothed.html')
        fig.show()


def annotate_video(video_path, face_predictions, output_path='annotated_video.avi'):
    """
    Annotate video frames with the top 3 emotion scores.
    
    :param video_path: Path to the input video file.
    :param face_predictions: List of face prediction dictionaries.
    :param output_path: Path to save the annotated video.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'mp4v' for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # create mapping from frame number to emotions
    emotions_map = {}
    for pred in face_predictions:
        frame = pred['frame']
        emotions_map.setdefault(frame, []).append(pred['emotions'])
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # check if current frame has emotion data
        if frame_number in emotions_map:
            for emotions in emotions_map[frame_number]:
                # Select Top 3 Emotions
                sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
                top_emotions = sorted_emotions[:3]
                
                # Prepare Emotion Text
                emotion_text = ', '.join([f"{emotion}: {score:.2f}" for emotion, score in top_emotions])
                
                # Calculate Text Size for Positioning
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(emotion_text, font, font_scale, thickness)
                
                # Determine Text Position (Top Right)
                margin = 10  # Pixels from the edge
                x = width - text_width - margin
                y = text_height + margin  # Start a bit below the top edge
                
                # OpenCV uses BGR format; blue is (255, 0, 0)
                cv2.putText(
                    frame, 
                    emotion_text, 
                    (x, y), 
                    font, 
                    font_scale, 
                    (255, 0, 0),  # Blue
                    thickness, 
                    cv2.LINE_AA
                )
        
        out.write(frame)
        
        frame_number += 1
        if frame_number % 100 == 0 or frame_number == total_frames:
            percent_complete = (frame_number / total_frames) * 100
            print(f"Processed {frame_number}/{total_frames} frames ({percent_complete:.2f}%)")
    
    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")


def download_video(url, save_path='input_video.avi'):
    """
    Download video from a URL to a local path.
    
    :param url: Direct download URL of the video.
    :param save_path: Local path to save the video.
    :return: Path to the saved video file.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Video downloaded to {save_path}")
        return save_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None
    
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
    # Example URLs to media files (replace with your own hosted URLs)
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
        "face": {
            "fps_pred": 3.0,
            "identify_faces": True
            },
        "language": {}
    }
    
    # Start a new inference job
    job_id = start_inference_job(
        urls=media_urls,
        models=models,
        # text=text_inputs,
        callback_url=None,  # Optional: specify if you have a callback endpoint
    )
        
    if not job_id:
        return
    
    # Monitor the job until completion
    monitor_job(job_id)
    
    # Get job details
    print("Getting job details...")
    get_job_details(job_id)
    
    # If job completed successfully, get predictions
    job = client.expression_measurement.batch.get_job_details(id=job_id)
    if job.state.status == "COMPLETED":
        predictions = get_job_predictions(job_id)
        if predictions:
            # Save predictions to a JSON file for later use (optional)
            with open('predictions.json', 'w') as f:
                json.dump(predictions, f, indent=4)
            # Proceed to visualize
            visualize_top_emotions_interactive(predictions, window_size=50)
            
            # Annotate the video
            # video_url = media_urls[0]
            # # Download the video locally
            # local_video_path = download_video(video_url)
            # if local_video_path:
            #     annotate_video(local_video_path, predictions['face'])
        else:
            print("No predictions available.")
    else:
        print("Job did not complete successfully.")
    
    list_jobs(limit=5)

if __name__ == "__main__":
    main()
