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
from typing import Dict, List, Union
from hume import HumeClient  # Make sure HumeClient is installed/configured

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize colorama
init()

# Initialize env variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY_TEMP")

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
            file_list = [
                f"{folder_path}/{file['name']}" if folder_path else file['name']
                for file in response if file['id']
            ]
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
# (Optional) Emotion Analyzer Class
# ---------------------------
EMOTION_WEIGHTS = {
    "happy": 1.0,
    "sad": -1.0,
    "neutral": 0.0,
    # Adjust or extend as needed.
}

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
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:n]
        return [{"emotion": emotion, "score": round(score, 3)} for emotion, score in sorted_emotions]

    def aggregate_emotion_score_2(self, emotion_scores: Dict[str, float], weights: Dict[str, float] = EMOTION_WEIGHTS) -> float:
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

# ---------------------------
# 1. Helper to Extract Aggregates from "Final" JSON
# ---------------------------
def extract_aggregates(json_data: dict) -> dict:
    """
    Given the final output JSON from the emotion analyzer,
    extract the aggregate scores and a participant_id from 'file_analyzed'.
    """
    results = json_data.get("results", {})
    overall = results.get("overall", {})
    timeline = results.get("timeline", {})

    # Parse out participant ID from file_analyzed, e.g. "PP1.mp4" -> "pp1"
    file_analyzed = json_data.get("file_analyzed", "")
    participant_id = file_analyzed.lower().replace(".mp4", "")

    # Compute the face aggregate
    face_entries = timeline.get("face", [])
    if isinstance(face_entries, list) and face_entries:
        face_aggregate = sum(item.get("aggregate_score", 0) for item in face_entries) / len(face_entries)
    else:
        face_aggregate = 0.0

    # Compute the prosody aggregate
    prosody_entries = timeline.get("prosody", [])
    if isinstance(prosody_entries, list) and prosody_entries:
        prosody_aggregate = sum(item.get("aggregate_score", 0) for item in prosody_entries) / len(prosody_entries)
    else:
        prosody_aggregate = 0.0

    # Compute the language aggregate
    language_data = timeline.get("language", {})
    if isinstance(language_data, list) and language_data:
        language_aggregate = sum(item.get("aggregate_score", 0) for item in language_data) / len(language_data)
    elif isinstance(language_data, dict):
        language_aggregate = language_data.get("aggregate_score", 0.0)
    else:
        language_aggregate = 0.0

    # Overall aggregate
    overall_aggregate = overall.get("aggregate_score", 0.0)

    return {
        "participant_id": participant_id,
        "face_aggregate": face_aggregate,
        "prosody_aggregate": prosody_aggregate,
        "language_aggregate": language_aggregate,
        "overall_aggregate": overall_aggregate
    }

# ---------------------------
# 2. Load the JSON Files from Supabase
# ---------------------------
def load_hume_features():
    supabase = SupabaseDB()
    files = supabase.list_storage_files(bucket_id="Hume Output")
    if not files:
        print("No files found in Hume Output bucket.")
        return pd.DataFrame()

    # Filter out any placeholder files
    files = [f for f in files if ".emptyFolderPlaceholder" not in f]

    urls = supabase.create_signed_url(bucket_id="Hume Output", files=files, expires_in=3600)
    print(f"Generated {len(urls)} URLs for Hume output files.")

    features_list = []
    for url in tqdm(urls, desc="Downloading Hume outputs"):
        try:
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()
            if not isinstance(json_data, dict):
                print(f"Skipping URL {url} because JSON is not a dict.")
                continue
            aggregates = extract_aggregates(json_data)
            features_list.append(aggregates)
        except Exception as e:
            print(f"Error processing URL {url}: {e}")

    df_features = pd.DataFrame(features_list)
    print("\nHume feature DataFrame:")
    print(df_features.head())
    return df_features

# ---------------------------
# 3. Load and Filter CSV from Supabase
# ---------------------------
def load_labels():
    """
    Fetch the labeled CSV file from Supabase and return it as a DataFrame.
    """
    supabase = SupabaseDB()
    files = supabase.list_storage_files(bucket_id="Interviews", folder_path="Labels")
    if not files:
        print("No label files found.")
        return pd.DataFrame()

    labeled_file = files[1]  # Choose the appropriate file
    labeled_urls = supabase.create_signed_url(bucket_id="Interviews", files=[labeled_file], expires_in=3600)

    try:
        response = requests.get(labeled_urls[0], stream=True)
        response.raise_for_status()
        temp_file_path = os.path.join("/tmp", "turker_scores_full_interview.csv")
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        df_labels = pd.read_csv(temp_file_path)
        print("\nFull Labeled CSV DataFrame (Before Filtering):")
        print(df_labels.head())

        # Filter rows to keep only those with "pp" in the "Participant" column (case-insensitive)
        df_labels = df_labels[df_labels['Participant'].str.contains('pp', case=False, na=False)].copy()
        
        # Only keep rows where Worker equals "AGGR"
        df_labels = df_labels[df_labels['Worker'].str.upper() == "AGGR"].copy()

        # Create a new column "participant_id" to match the JSON (e.g., "PP1" -> "pp1")
        df_labels['participant_id'] = df_labels['Participant'].str.lower()

        print("\nFiltered Labeled CSV DataFrame (Only 'pp'):")
        print(df_labels.head(n=30))

        return df_labels
    except Exception as e:
        print(f"Error loading labeled scores: {e}")
        return pd.DataFrame()

# ---------------------------
# Data Visualization Function
# ---------------------------
def visualize_data(df):
    feature_columns = ['face_aggregate', 'prosody_aggregate', 'language_aggregate', 'overall_aggregate']
    
    # Pairplot of features and target
    sns.pairplot(df[feature_columns + ['RecommendHiring']])
    plt.suptitle("Pairplot of Features and Target", y=1.02)
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[feature_columns + ['RecommendHiring']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()
    
    # Distribution of the target variable
    plt.figure(figsize=(8, 6))
    sns.histplot(df['RecommendHiring'], kde=True, color='purple')
    plt.title("Distribution of RecommendHiring")
    plt.xlabel("RecommendHiring")
    plt.ylabel("Frequency")
    plt.show()

# ---------------------------
# 4. SVR/Lasso Training Pipeline with Visualizations
# ---------------------------
def train_regression_model(df):
    """
    Given a merged DataFrame with features and labels, train and evaluate regression models.
    We'll use 'RecommendHiring' from the CSV as the ground truth (target).
    """
    if "RecommendHiring" not in df.columns:
        print("No 'RecommendHiring' column in merged data. Exiting.")
        return

    # Define features and target
    feature_columns = ['face_aggregate', 'prosody_aggregate', 'language_aggregate', 'overall_aggregate']
    X = df[feature_columns]
    y = df['RecommendHiring']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- Linear SVR ---
    linear_svr = LinearSVR(max_iter=100000)
    param_grid_linear = {'C': [0.1, 1, 10]}
    svr_cv = GridSearchCV(linear_svr, param_grid_linear, cv=5, scoring='neg_mean_squared_error')
    svr_cv.fit(X_train, y_train)
    best_linear_svr = svr_cv.best_estimator_

    svr_predictions = best_linear_svr.predict(X_test)
    print("\nLinear SVR Performance (predicting 'RecommendHiring'):")
    print("R^2 Score:", r2_score(y_test, svr_predictions))
    print("MSE:", mean_squared_error(y_test, svr_predictions))

    emotion_weights_svr = {feature: coef for feature, coef in zip(feature_columns, best_linear_svr.coef_)}
    print("\nLinear SVR Feature Weights:")
    for feature, weight in emotion_weights_svr.items():
        print(f"{feature}: {weight}")

    # Plot Actual vs. Predicted for Linear SVR
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, svr_predictions, alpha=0.7, color='blue')
    plt.xlabel("Actual RecommendHiring")
    plt.ylabel("Predicted RecommendHiring")
    plt.title("Linear SVR: Actual vs Predicted")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.show()

    # Plot residual distribution for Linear SVR
    residuals_svr = y_test - svr_predictions
    plt.figure(figsize=(8, 6))
    plt.hist(residuals_svr, bins=20, color='blue', alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Linear SVR: Residual Distribution")
    plt.show()

    # --- Lasso Regression ---
    lasso = Lasso(max_iter=10000)
    param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    lasso_cv = GridSearchCV(lasso, param_grid_lasso, cv=5, scoring='neg_mean_squared_error')
    lasso_cv.fit(X_train, y_train)
    best_lasso = lasso_cv.best_estimator_

    lasso_predictions = best_lasso.predict(X_test)
    print("\nLasso Regression Performance (predicting 'RecommendHiring'):")
    print("R^2 Score:", r2_score(y_test, lasso_predictions))
    print("MSE:", mean_squared_error(y_test, lasso_predictions))

    lasso_feature_importance = {feature: coef for feature, coef in zip(feature_columns, best_lasso.coef_)}
    print("\nLasso Feature Importance:")
    for feature, coef in lasso_feature_importance.items():
        print(f"{feature}: {coef}")

    # Plot Actual vs. Predicted for Lasso Regression
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, lasso_predictions, alpha=0.7, color='green')
    plt.xlabel("Actual RecommendHiring")
    plt.ylabel("Predicted RecommendHiring")
    plt.title("Lasso Regression: Actual vs Predicted")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.show()

    # Plot residual distribution for Lasso Regression
    residuals_lasso = y_test - lasso_predictions
    plt.figure(figsize=(8, 6))
    plt.hist(residuals_lasso, bins=20, color='green', alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Lasso Regression: Residual Distribution")
    plt.show()

# ---------------------------
# 5. Main Pipeline
# ---------------------------
def main():
    # Load features from Hume outputs
    df_features = load_hume_features()
    if df_features.empty:
        print("No features loaded. Exiting.")
        return

    # Load labeled scores from CSV (filtering only rows that contain "pp")
    df_labels = load_labels()
    if df_labels.empty:
        print("No labels loaded. Exiting.")
        return

    # Merge the two DataFrames on "participant_id"
    df_merged = pd.merge(df_features, df_labels, on="participant_id", how="inner")
    print("\nMerged Dataset:")
    print(df_merged.head(n=30))
    
    # Visualize the merged data (features and target)
    visualize_data(df_merged)
    
    # Train the regression models using the merged dataset
    train_regression_model(df_merged)

if __name__ == "__main__":
    main()
