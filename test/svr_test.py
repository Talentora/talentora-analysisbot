import time
import random
import pandas as pd
import numpy as np
import os

# Hume Client and Job Manager Setup 
from hume import HumeClient  
from typing import Dict, List, Optional


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
        print("Job payload:", job_payload)

        try:
            return self.client.expression_measurement.batch.start_inference_job(**job_payload)
        except Exception as e:
            print(f"Error starting job: {e}")
            return None

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
                'ended_at': getattr(job.state, 'ended_timestamp_ms', None),
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

# Sentiment Analysis (Emotion Analyzer)
EMOTION_WEIGHTS = {
    "happy": 1.0,
    "sad": -1.0,
    "neutral": 0.0,
    
}

from typing import Union

class EmotionAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the EmotionAnalyzer with a Hume API key."""
        self.client = HumeClient(api_key=api_key)

    def aggregate_emotion_score(self, emotion_scores: Dict[str, float], weights: Dict[str, float] = EMOTION_WEIGHTS) -> float:
        """
        Aggregate emotion scores into a single score out of 10.
        """
        total = 0.0
        max_possible = 0.0
        min_possible = 0.0

        for emotion, score in emotion_scores.items():
            weight = weights.get(emotion, 0.0)
            total += weight * score
            if weight > 0:
                max_possible += weight
            elif weight < 0:
                min_possible += weight

        if max_possible - min_possible == 0:
            return 0.0

        normalized_score = 10 * (total - min_possible) / (max_possible - min_possible)
        return round(max(0, min(10, normalized_score)), 2)
    
    def get_top_emotions(self, emotions: Dict[str, float], n: int = 3) -> List[Dict[str, Union[str, float]]]:
        """
        Get the top n emotions with highest scores.
        Returns list of dicts with emotion name and score.
        """
        sorted_emotions = sorted(
            emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return [
            {"emotion": emotion, "score": round(score, 3)}
            for emotion, score in sorted_emotions
        ]

    def process_predictions(self, predictions) -> Dict[str, Dict[str, Union[float, Dict[str, float], List[Dict]]]]:
        """
        Process predictions and return accumulated emotion scores.
        """
        face_timeline = []
        prosody_timeline = []
        language_timeline = []
        
        face_accumulated = {}
        prosody_accumulated = {}
        language_accumulated = {}
        counts = {'face': 0, 'prosody': 0, 'language': 0}

        for prediction in predictions:
            for result in prediction.results.predictions:
                # Process Face Model Predictions
                if result.models.face:
                    for group in result.models.face.grouped_predictions:
                        for pred in group.predictions:
                            emotions = {emotion.name: emotion.score for emotion in pred.emotions}
                            face_timeline.append({
                                'time': pred.time,  # Time in seconds
                                'frame': pred.frame,
                                'emotions': emotions,
                                'aggregate_score': self.aggregate_emotion_score(emotions),
                                'id': group.id  # Track individual faces
                            })
                            
                            for emotion, score in emotions.items():
                                face_accumulated[emotion] = face_accumulated.get(emotion, 0.0) + score
                            counts['face'] += 1

                # Process Prosody Model Predictions
                if result.models.prosody:
                    for group in result.models.prosody.grouped_predictions:
                        for pred in group.predictions:
                            emotions = {emotion.name: emotion.score for emotion in pred.emotions}
                            prosody_timeline.append({
                                'time_start': pred.time.begin,
                                'time_end': pred.time.end,
                                'emotions': emotions,
                                'aggregate_score': self.aggregate_emotion_score(emotions),
                                'id': group.id,
                                'text': getattr(pred, 'text', None)
                            })
                            
                            for emotion, score in emotions.items():
                                prosody_accumulated[emotion] = prosody_accumulated.get(emotion, 0.0) + score
                            counts['prosody'] += 1

                # Process Language Model Predictions
                if result.models.language:
                    for group in result.models.language.grouped_predictions:
                        for pred in group.predictions:
                            emotions = {emotion.name: emotion.score for emotion in pred.emotions}
                            entry = {
                                'emotions': emotions,
                                'aggregate_score': self.aggregate_emotion_score(emotions),
                                'text': pred.text,
                                'position': {
                                    'begin': pred.position.begin,
                                    'end': pred.position.end
                                }
                            }
                            
                            # Add time information if available
                            if hasattr(pred, 'time') and pred.time:
                                entry['time_start'] = pred.time.begin
                                entry['time_end'] = pred.time.end
                                
                            language_timeline.append(entry)
                            
                            for emotion, score in emotions.items():
                                language_accumulated[emotion] = language_accumulated.get(emotion, 0.0) + score
                            counts['language'] += 1

        # Calculate averages
        averages = {
            'face': {
                emotion: score / counts['face'] 
                for emotion, score in face_accumulated.items()
            } if counts['face'] > 0 else {},
            'prosody': {
                emotion: score / counts['prosody']
                for emotion, score in prosody_accumulated.items()
            } if counts['prosody'] > 0 else {},
            'language': {
                emotion: score / counts['language']
                for emotion, score in language_accumulated.items()
            } if counts['language'] > 0 else {}
        }
        
        overall_emotions = {}
        valid_modalities = 0

        for modality in ['face', 'prosody', 'language']:
            if counts[modality] > 0:
                valid_modalities += 1
                for emotion, score in averages[modality].items():
                    overall_emotions[emotion] = overall_emotions.get(emotion, 0) + score

        # Average the emotions across modalities
        if valid_modalities > 0:
            overall_emotions = {
                emotion: score / valid_modalities 
                for emotion, score in overall_emotions.items()
            }

        # Get top emotions
        top_emotions = self.get_top_emotions(overall_emotions) if overall_emotions else []

        return {
            'timeline': {
                'face': sorted(face_timeline, key=lambda x: x['time']),
                'prosody': sorted(prosody_timeline, key=lambda x: x['time_start']),
                'language': sorted(language_timeline, key=lambda x: x.get('time_start', x['position']['begin']))
            },
            'averages': {
                'face': {
                    'emotions': averages['face'],
                    'aggregate_score': self.aggregate_emotion_score(averages['face']) if averages['face'] else 0
                },
                'prosody': {
                    'emotions': averages['prosody'],
                    'aggregate_score': self.aggregate_emotion_score(averages['prosody']) if averages['prosody'] else 0
                },
                'language': {
                    'emotions': averages['language'],
                    'aggregate_score': self.aggregate_emotion_score(averages['language']) if averages['language'] else 0
                }
            },
            'overall': {
                'emotions': overall_emotions,
                'top_emotions': top_emotions,
                'aggregate_score': self.aggregate_emotion_score(overall_emotions) if overall_emotions else 0
            },
            'metadata': {
                'counts': counts
            }
        }

# Regression Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Main Pipeline
def main():
    client = HumeClient(api_key=os.environ.get("HUME_API_KEY"))
    
    # Initialize JobManager
    job_manager = JobManager(client=client)
    
    interview_urls = [
        "https://laieiinzukjqqbaglafj.supabase.co/storage/v1/object/sign/Interviews/MP4/P10.mp4?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJJbnRlcnZpZXdzL01QNC9QMTAubXA0IiwiaWF0IjoxNzM5NDgzMzQ5LCJleHAiOjE3NDAwODgxNDl9.nL8lOw1s1ivA33sVEbFv05cyrnoxoVdWCXy68Kvc_QU",
        "https://laieiinzukjqqbaglafj.supabase.co/storage/v1/object/sign/Interviews/MP4/P11.mp4?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJJbnRlcnZpZXdzL01QNC9QMTEubXA0IiwiaWF0IjoxNzM5NDgzMzc1LCJleHAiOjE3NDAwODgxNzV9.qG7EqgWdIkS4_FG_a0sVZM7dLyMTj76X5pcuO0nNXS0",
        "https://laieiinzukjqqbaglafj.supabase.co/storage/v1/object/sign/Interviews/MP4/P12.mp4?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJJbnRlcnZpZXdzL01QNC9QMTIubXA0IiwiaWF0IjoxNzM5NDgzMzg4LCJleHAiOjE3NDAwODgxODh9.qP6fc_IW2K8-FuaT90g8D_rej_637rrCBiaIyPqAXaA"
    ]
    
    models = {
            "face": {},
            "language": {},
            "prosody": {}
        }
    
    # Start the job
    job_id = job_manager.start_job(urls=interview_urls, models=models)
    if not job_id:
        print("Failed to start job.")
        return

    print(f"Started job with ID: {job_id}")
    
    # Poll for job completion
    status = None
    while True:
        details = job_manager.get_job_details(job_id)
        status = details.get('status') if details else None
        print(f"Job status: {status}")
        # Adjust the status check based on your Hume job responses (e.g., "COMPLETED")
        if status and status.upper() == "COMPLETED":
            break
        time.sleep(10)  # Wait before polling again
    
    # Get job predictions
    predictions = job_manager.get_job_predictions(job_id)
    if not predictions:
        print("No predictions found.")
        return
    
    # Process predictions using the EmotionAnalyzer
    emotion_analyzer = EmotionAnalyzer(api_key=os.environ.get("HUME_API_KEY"))
    processed_data = emotion_analyzer.process_predictions(predictions)
    
    # Extract example features from the processed predictions.
    overall_agg = processed_data['overall']['aggregate_score']
    face_agg = processed_data['averages']['face']['aggregate_score']
    prosody_agg = processed_data['averages']['prosody']['aggregate_score']
    language_agg = processed_data['averages']['language']['aggregate_score']
    
    print("Extracted Aggregated Scores:")
    print(f"Overall: {overall_agg}, Face: {face_agg}, Prosody: {prosody_agg}, Language: {language_agg}")
    
    # For demonstration, simulate a dataset by replicating these features for multiple candidates.
    simulated_data = []
    for i in range(10):
        candidate_features = {
            'face_aggregate': face_agg + random.uniform(-0.5, 0.5),
            'prosody_aggregate': prosody_agg + random.uniform(-0.5, 0.5),
            'language_aggregate': language_agg + random.uniform(-0.5, 0.5),
            'overall_aggregate': overall_agg + random.uniform(-0.5, 0.5),
            # Simulated hiring score (target) between 5 and 10.
            'hiring_score': random.uniform(5, 10)
        }
        simulated_data.append(candidate_features)
    
    df = pd.DataFrame(simulated_data)
    print("\nSimulated Dataset:")
    print(df.head())
    
    # Prepare features and target for regression
    feature_columns = ['face_aggregate', 'prosody_aggregate', 'language_aggregate', 'overall_aggregate']
    X = df[feature_columns]
    y = df['hiring_score']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Linear SVR to Determine Emotion Weights
    linear_svr = LinearSVR(max_iter=10000)
    param_grid_linear = {'C': [0.1, 1, 10]}
    svr_cv = GridSearchCV(linear_svr, param_grid_linear, cv=5, scoring='neg_mean_squared_error')
    svr_cv.fit(X_train, y_train)
    best_linear_svr = svr_cv.best_estimator_
    
    # Predictions and performance
    svr_predictions = best_linear_svr.predict(X_test)
    print("\nLinear SVR Performance:")
    print("R^2 Score:", r2_score(y_test, svr_predictions))
    print("MSE:", mean_squared_error(y_test, svr_predictions))
    
    # Extract emotion weights from the linear SVR coefficients
    emotion_weights_svr = {feature: coef for feature, coef in zip(feature_columns, best_linear_svr.coef_)}
    print("\nLinear SVR Emotion Weights:")
    for feature, weight in emotion_weights_svr.items():
        print(f"{feature}: {weight}")
    
    # Lasso Regression for Additional Interpretability
    lasso = Lasso(max_iter=10000)
    param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    lasso_cv = GridSearchCV(lasso, param_grid_lasso, cv=5, scoring='neg_mean_squared_error')
    lasso_cv.fit(X_train, y_train)
    best_lasso = lasso_cv.best_estimator_
    
    lasso_predictions = best_lasso.predict(X_test)
    print("\nLasso Regression Performance:")
    print("R^2 Score:", r2_score(y_test, lasso_predictions))
    print("MSE:", mean_squared_error(y_test, lasso_predictions))
    
    lasso_feature_importance = {feature: coef for feature, coef in zip(feature_columns, best_lasso.coef_)}
    print("\nLasso Feature Importance:")
    for feature, coef in lasso_feature_importance.items():
        print(f"{feature}: {coef}")

if __name__ == "__main__":
    main()
