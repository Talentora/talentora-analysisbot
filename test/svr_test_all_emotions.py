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
# (If you're re-processing raw predictions; might not be needed if you already have final JSON)
# ---------------------------
EMOTION_WEIGHTS = {
    "happy": 1.0,
    "sad": -1.0,
    "neutral": 0.0,
    # Adjust or extend as needed.
}

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

    # If you have raw predictions, you would define process_predictions here
    # But if your JSON is already "final", you can skip re-processing.


# ---------------------------
# 1. Helper to Extract Emotion Features from All Modalities
# ---------------------------
def extract_emotion_features(json_data: dict) -> dict:
    """
    Given the final output JSON from the emotion analyzer,
    extract individual emotion values from the "averages" section for each modality:
      - face, prosody, language, and overall.
    For each emotion, a feature is created with a prefix (e.g., "face_Admiration").
    Also extracts a participant_id from 'file_analyzed'.
    """
    averages = json_data.get("results", {}).get("averages", {})
    features = {}
    modalities = ["face", "prosody", "language", "overall"]
    for modality in modalities:
        modality_data = averages.get(modality, {})
        emotions = modality_data.get("emotions", {})
        for emotion, score in emotions.items():
            key = f"{modality}_{emotion.replace(' ', '_').replace('(', '').replace(')', '')}"
            features[key] = score
    # Extract participant_id from file_analyzed (e.g., "PP1.mp4" -> "pp1")
    file_analyzed = json_data.get("file_analyzed", "")
    participant_id = file_analyzed.lower().replace(".mp4", "")
    features["participant_id"] = participant_id
    return features

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
            feature_dict = extract_emotion_features(json_data)
            features_list.append(feature_dict)
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

        # Create a new column "participant_id" to match the JSON
        # E.g. "PP1" -> "pp1"
        df_labels['participant_id'] = df_labels['Participant'].str.lower()

        print("\nFiltered Labeled CSV DataFrame (Only 'pp'):")
        print(df_labels.head(n=30))

        return df_labels
    except Exception as e:
        print(f"Error loading labeled scores: {e}")
        return pd.DataFrame()

# ---------------------------
# 4. SVR/Lasso Training Pipeline
# ---------------------------
def train_regression_model(df):
    """
    Given a merged DataFrame with features and labels, train and evaluate regression models.
    We'll use 'RecommendHiring' from the CSV as the ground truth (target).
    """
    # Ensure "RecommendHiring" exists in the merged DataFrame
    if "RecommendHiring" not in df.columns:
        print("No 'RecommendHiring' column in merged data. Exiting.")
        return

    modality_prefixes = ("face_", "prosody_", "language_", "overall_")
    feature_columns = [col for col in df.columns if col.startswith(modality_prefixes)]
    if not feature_columns:
        print("No emotion feature columns found. Exiting.")
        return

    X = df[feature_columns]
    y = df['RecommendHiring']  # <--- Using "RecommendHiring" as the target

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
    print("\nLinear SVR Performance (predicting 'RecommendHiring'):")
    print("R^2 Score:", r2_score(y_test, svr_predictions))
    print("MSE:", mean_squared_error(y_test, svr_predictions))

    emotion_weights_svr = {feature: coef for feature, coef in zip(feature_columns, best_linear_svr.coef_)}
    print("\nLinear SVR Feature Weights:")
    for feature, weight in emotion_weights_svr.items():
        print(f"{feature}: {weight}")

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

# ---------------------------
# 5. Main Pipeline
# ---------------------------
def main():
    # Load features from final Hume outputs
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

    # Train the regression models using the merged dataset
    train_regression_model(df_merged)

if __name__ == "__main__":
    main()
