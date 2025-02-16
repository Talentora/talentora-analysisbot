import os
import time
import random
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from supabase import create_client
from colorama import Fore, Style, init
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Initialize colorama
init()

# Initialize env variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") 

# ---------------------------
# Supabase Data Access Class
# ---------------------------
class SupabaseDB:
    def __init__(self):
        print(f"{Fore.CYAN}Initializing Supabase client...{Style.RESET_ALL}")
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"{Fore.GREEN}✓ Supabase client initialized successfully{Style.RESET_ALL}")

    def list_storage_files(self, bucket_id: str, folder_path: str = "", limit: int = None):
        try:
            folder_path = folder_path.strip('/')
            options = {"limit": limit} if limit else None
            response = self.client.storage.from_(bucket_id).list(
                path=folder_path,
                options=options
            )
            file_list = [f"{folder_path}/{file['name']}" if folder_path else file['name']
                        for file in response if file['id']]
            return file_list
        except Exception as e:
            print(f"{Fore.RED}✗ Error listing files: {str(e)}{Style.RESET_ALL}")
            return None

    def create_signed_url(self, bucket_id: str, files: list, expires_in: int = 3600):
        try:
            response = self.client.storage.from_(bucket_id).create_signed_urls(
                paths=files,
                expires_in=expires_in
            )
            urls = [url["signedURL"] for url in response]
            return urls
        except Exception as e:
            print(f"{Fore.RED}✗ Error creating signed URL: {str(e)}{Style.RESET_ALL}")
            return None

# ---------------------------
# Emotion Analyzer (from Hume)
# ---------------------------
EMOTION_WEIGHTS = {
    "happy": 1.0,
    "sad": -1.0,
    "neutral": 0.0,
    # Adjust or extend as needed.
}

from typing import Dict, List, Union
from hume import HumeClient  # Make sure HumeClient is installed/configured

class EmotionAnalyzer:
    def __init__(self, api_key: str):
        self.client = HumeClient(api_key=api_key)

    def aggregate_emotion_score(self, emotion_scores: Dict[str, float], weights: Dict[str, float] = EMOTION_WEIGHTS) -> float:
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
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:n]
        return [{"emotion": emotion, "score": round(score, 3)} for emotion, score in sorted_emotions]

    def process_predictions(self, predictions) -> Dict[str, Dict[str, Union[float, Dict[str, float], List[Dict]]]]:
        # (The implementation remains as in your original code.)
        face_timeline, prosody_timeline, language_timeline = [], [], []
        face_accumulated, prosody_accumulated, language_accumulated = {}, {}, {}
        counts = {'face': 0, 'prosody': 0, 'language': 0}

        for prediction in predictions:
            for result in prediction.results.predictions:
                if result.models.face:
                    for group in result.models.face.grouped_predictions:
                        for pred in group.predictions:
                            emotions = {e.name: e.score for e in pred.emotions}
                            face_timeline.append({
                                'time': pred.time,
                                'frame': pred.frame,
                                'emotions': emotions,
                                'aggregate_score': self.aggregate_emotion_score(emotions),
                                'id': group.id
                            })
                            for emotion, score in emotions.items():
                                face_accumulated[emotion] = face_accumulated.get(emotion, 0.0) + score
                            counts['face'] += 1
                if result.models.prosody:
                    for group in result.models.prosody.grouped_predictions:
                        for pred in group.predictions:
                            emotions = {e.name: e.score for e in pred.emotions}
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
                if result.models.language:
                    for group in result.models.language.grouped_predictions:
                        for pred in group.predictions:
                            emotions = {e.name: e.score for e in pred.emotions}
                            entry = {
                                'emotions': emotions,
                                'aggregate_score': self.aggregate_emotion_score(emotions),
                                'text': pred.text,
                                'position': {'begin': pred.position.begin, 'end': pred.position.end}
                            }
                            if hasattr(pred, 'time') and pred.time:
                                entry['time_start'] = pred.time.begin
                                entry['time_end'] = pred.time.end
                            language_timeline.append(entry)
                            for emotion, score in emotions.items():
                                language_accumulated[emotion] = language_accumulated.get(emotion, 0.0) + score
                            counts['language'] += 1

        averages = {
            'face': {emotion: score / counts['face'] for emotion, score in face_accumulated.items()} if counts['face'] > 0 else {},
            'prosody': {emotion: score / counts['prosody'] for emotion, score in prosody_accumulated.items()} if counts['prosody'] > 0 else {},
            'language': {emotion: score / counts['language'] for emotion, score in language_accumulated.items()} if counts['language'] > 0 else {}
        }
        
        overall_emotions = {}
        valid_modalities = 0
        for modality in ['face', 'prosody', 'language']:
            if counts[modality] > 0:
                valid_modalities += 1
                for emotion, score in averages[modality].items():
                    overall_emotions[emotion] = overall_emotions.get(emotion, 0) + score
        if valid_modalities > 0:
            overall_emotions = {emotion: score / valid_modalities for emotion, score in overall_emotions.items()}
        top_emotions = self.get_top_emotions(overall_emotions) if overall_emotions else []
        return {
            'timeline': {
                'face': sorted(face_timeline, key=lambda x: x['time']),
                'prosody': sorted(prosody_timeline, key=lambda x: x['time_start']),
                'language': sorted(language_timeline, key=lambda x: x.get('time_start', x['position']['begin']))
            },
            'averages': {
                'face': {'emotions': averages['face'], 'aggregate_score': self.aggregate_emotion_score(averages['face']) if averages['face'] else 0},
                'prosody': {'emotions': averages['prosody'], 'aggregate_score': self.aggregate_emotion_score(averages['prosody']) if averages['prosody'] else 0},
                'language': {'emotions': averages['language'], 'aggregate_score': self.aggregate_emotion_score(averages['language']) if averages['language'] else 0}
            },
            'overall': {
                'emotions': overall_emotions,
                'top_emotions': top_emotions,
                'aggregate_score': self.aggregate_emotion_score(overall_emotions) if overall_emotions else 0
            },
            'metadata': {'counts': counts}
        }

# ---------------------------
# Data Loading Helpers
# ---------------------------
def load_hume_features():
    """
    Fetch Hume JSON outputs from Supabase, process them, and return a DataFrame of features.
    """
    supabase = SupabaseDB()
    # List files in the "Hume Output" bucket
    files = supabase.list_storage_files(bucket_id="Hume Output")
    if not files:
        print("No files found in Hume Output bucket.")
        return pd.DataFrame()
    
    # Create signed URLs for the files
    urls = supabase.create_signed_url(bucket_id="Hume Output", files=files, expires_in=3600)
    print(f"Generated {len(urls)} URLs for Hume output files.")

    json_objects = []
    for url in tqdm(urls, desc="Downloading Hume outputs"):
        try:
            response = requests.get(url)
            response.raise_for_status()
            # Assume the JSON structure has a top-level "results" key
            json_data = response.json().get("results")
            if json_data:
                json_objects.append(json_data)
        except Exception as e:
            print(f"Error downloading/parsing JSON from {url}: {e}")
    
    # Process each JSON object using the EmotionAnalyzer
    # (Assuming each JSON object corresponds to one candidate/interview)
    emotion_analyzer = EmotionAnalyzer(api_key=os.environ.get("HUME_API_KEY"))
    features_list = []
    for idx, predictions in enumerate(json_objects):
        try:
            processed = emotion_analyzer.process_predictions(predictions)
            features = {
                "candidate_id": idx,  # or use an identifier from your data if available
                "face_aggregate": processed["averages"]["face"]["aggregate_score"],
                "prosody_aggregate": processed["averages"]["prosody"]["aggregate_score"],
                "language_aggregate": processed["averages"]["language"]["aggregate_score"],
                "overall_aggregate": processed["overall"]["aggregate_score"]
            }
            features_list.append(features)
        except Exception as e:
            print(f"Error processing predictions for candidate {idx}: {e}")
    
    df_features = pd.DataFrame(features_list)
    print("Hume feature DataFrame:")
    print(df_features.head())
    return df_features

def load_labels():
    """
    Fetch the labeled CSV file from Supabase and return it as a DataFrame.
    """
    supabase = SupabaseDB()
    files = supabase.list_storage_files(bucket_id="Interviews", folder_path="Labels")
    if not files:
        print("No label files found.")
        return pd.DataFrame()
    labeled_file = files[0]  # Choose the appropriate file
    labeled_urls = supabase.create_signed_url(bucket_id="Interviews", files=[labeled_file], expires_in=3600)
    try:
        response = requests.get(labeled_urls[0], stream=True)
        response.raise_for_status()
        temp_file_path = os.path.join("/tmp", "labeled_scores.csv")
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        df_labels = pd.read_csv(temp_file_path)
        print("Labeled CSV DataFrame:")
        print(df_labels.head())
        return df_labels
    except Exception as e:
        print(f"Error loading labeled scores: {e}")
        return pd.DataFrame()

# ---------------------------
# SVR Training Pipeline
# ---------------------------
def train_regression_model(df):
    """
    Given a merged DataFrame with features and labels, train and evaluate regression models.
    """
    # Assume df has columns: candidate_id, face_aggregate, prosody_aggregate, language_aggregate, overall_aggregate, hiring_score
    feature_columns = ['face_aggregate', 'prosody_aggregate', 'language_aggregate', 'overall_aggregate']
    X = df[feature_columns]
    y = df['hiring_score']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # --- Linear SVR ---
    linear_svr = LinearSVR(max_iter=10000)
    param_grid_linear = {'C': [0.1, 1, 10]}
    svr_cv = GridSearchCV(linear_svr, param_grid_linear, cv=5, scoring='neg_mean_squared_error')
    svr_cv.fit(X_train, y_train)
    best_linear_svr = svr_cv.best_estimator_
    
    svr_predictions = best_linear_svr.predict(X_test)
    print("\nLinear SVR Performance:")
    print("R^2 Score:", r2_score(y_test, svr_predictions))
    print("MSE:", mean_squared_error(y_test, svr_predictions))
    
    emotion_weights_svr = {feature: coef for feature, coef in zip(feature_columns, best_linear_svr.coef_)}
    print("\nLinear SVR Emotion Weights:")
    for feature, weight in emotion_weights_svr.items():
        print(f"{feature}: {weight}")
    
    # --- Lasso Regression ---
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

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    # Load features from Hume outputs
    df_features = load_hume_features()
    if df_features.empty:
        print("No features loaded. Exiting.")
        return

    # Load labeled scores
    df_labels = load_labels()
    if df_labels.empty:
        print("No labels loaded. Exiting.")
        return

    # Merge on candidate_id or another common key (adjust as needed)
    # Here, we assume that the order matches or you have a common key.
    # For demonstration, we add a column 'candidate_id' to df_labels if missing.
    if 'candidate_id' not in df_labels.columns:
        df_labels['candidate_id'] = range(len(df_labels))
    
    df_merged = pd.merge(df_features, df_labels, on="candidate_id", how="inner")
    print("\nMerged Dataset:")
    print(df_merged.head())
    
    # Now train the regression models using the merged dataset.
    train_regression_model(df_merged)

if __name__ == "__main__":
    main()
