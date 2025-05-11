from collections import defaultdict
import os
import statistics
import joblib
import numpy as np
import pandas as pd
import ast

import requests

from app.controllers.supabase_db import SupabaseDB

"""This script contains functions that are used to preprocess the data from HUME's emotion models.
The functions are used to calculate the aggregates for each emotion, and to expand the dictionary columns into separate columns.
"""

class DataPreprocessor:
    """A class for preprocessing data from HUME's emotion models.
    
    This class handles the calculation of aggregates for each emotion,
    and the generation of model input dataframes, as well as merging the data with the labels for a model-ready training set.
    It also has a function to prepare a single input for the SVR model. Which will be used to predict a single video (post training).
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.model_input_dataframes = {} 
        self.model_labels = {}

    def get_model_input_dataframes(self):
        """Get the model input dataframes for single predictions."""
        if not self.model_input_dataframes:
            raise ValueError("Model input dataframes not initialized, please run process_prediction first")
        return self.model_input_dataframes
        
    def calculate_aggregates(self, emotion_scores):
        """Calculate statistical aggregates for each emotion's scores.
        
        Args:
            emotion_scores (dict): A dictionary where each key is an emotion name (str),
                and the corresponding value is a list of numerical scores (List[float])
                associated with that emotion.

        Returns:
            dict: A dictionary with the flattened keys for each emotion, and the values are the calculated aggregates.
            Example: {"admiration_mean": 0.5, ..., "fear_mean": 0.3, "fear_max": 0.8, ...}
        """
        aggregates = {}
        for emotion, scores in emotion_scores.items():
            if scores:  # Ensure the list is not empty
                mean_val = statistics.mean(scores)
                max_val = max(scores)
                min_val = min(scores)
                std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
            else:
                mean_val = max_val = min_val = std_dev = 0.0
            
            
            aggregates[f"{emotion}_mean"] = mean_val
            aggregates[f"{emotion}_max"] = max_val
            aggregates[f"{emotion}_min"] = min_val
            aggregates[f"{emotion}_std_dev"] = std_dev

        return aggregates
    
    
    def process_training_predictions(self, hume_json, write_to_csv = False):
        """
        Process predictions from HUME's Face, Prosody, and Language emotion models for MULTIPLE videos.
        Returns three pandas DataFrames: face_df, prosody_df, language_df, this is used for training the SVR model.
        """
        try:
            print("Starting process_training_predictions...")
            print(f"Input JSON type: {type(hume_json)}")
            print(f"Input JSON length: {len(hume_json) if isinstance(hume_json, (list, dict)) else 'N/A'}")
            
            face_aggregates = {}
            language_aggregates = {}
            prosody_aggregates = {}
            
            for video_dict in hume_json:
                try:
                    prediction = video_dict.get("results", {}).get("predictions", [])[0]
                    
                    file_name = prediction.get("file", "unknown_file")
                    print(f"Processing file: {file_name}")
                    
                    face_data_emotions = defaultdict(list)
                    language_data_emotions = defaultdict(list)
                    prosody_data_emotions = defaultdict(list)
                    
                    # Process face predictions
                    face_grouped_predictions = prediction.get("models", {}).get("face", {}).get("grouped_predictions", [])
                    for group in face_grouped_predictions:
                        for face_prediction in group.get("predictions", []):
                            for emotion in face_prediction.get("emotions", []):
                                face_data_emotions[emotion.get("name")].append(emotion.get("score", 0.0))

                    # Process language predictions
                    language_grouped_predictions = prediction.get("models", {}).get("language", {}).get("grouped_predictions", [])
                    for group in language_grouped_predictions:
                        for language_prediction in group.get("predictions", []):
                            for emotion in language_prediction.get("emotions", []):
                                language_data_emotions[emotion.get("name")].append(emotion.get("score", 0.0))
                    
                    # Process prosody predictions
                    prosody_grouped_predictions = prediction.get("models", {}).get("prosody", {}).get("grouped_predictions", [])
                    for group in prosody_grouped_predictions:
                        for prosody_prediction in group.get("predictions", []):
                            for emotion in prosody_prediction.get("emotions", []):
                                prosody_data_emotions[emotion.get("name")].append(emotion.get("score", 0.0))

                    # Calculate aggregates
                    print("Calculating aggregates...")
                    face_aggregates[file_name] = self.calculate_aggregates(face_data_emotions)
                    language_aggregates[file_name] = self.calculate_aggregates(language_data_emotions)
                    prosody_aggregates[file_name] = self.calculate_aggregates(prosody_data_emotions)
                    
                except Exception as e:
                    print(f"Error processing video: {str(e)}")
                    print(f"Video data: {video_dict}")
                    raise
            
            print("Creating DataFrames...")
            # Create DataFrames
            face_df = pd.DataFrame.from_dict(face_aggregates, orient='index')
            prosody_df = pd.DataFrame.from_dict(prosody_aggregates, orient='index')
            language_df = pd.DataFrame.from_dict(language_aggregates, orient='index')
            
            print(f"Face DataFrame shape: {face_df.shape}")
            print(f"Prosody DataFrame shape: {prosody_df.shape}")
            print(f"Language DataFrame shape: {language_df.shape}")
            
            # Convert index to column
            face_df = face_df.reset_index().rename(columns={'index': 'video_id'})
            prosody_df = prosody_df.reset_index().rename(columns={'index': 'video_id'})
            language_df = language_df.reset_index().rename(columns={'index': 'video_id'})
            
            if write_to_csv:
                print("Writing to CSV files...")
                face_df.to_csv('face_predictions.csv', index=False)
                prosody_df.to_csv('prosody_predictions.csv', index=False)
                language_df.to_csv('language_predictions.csv', index=False)
                print(f"Generated {len(face_df)} samples for face_df")
                print(f"Generated {len(prosody_df)} samples for prosody_df")
                print(f"Generated {len(language_df)} samples for language_df")
            
            return face_df, prosody_df, language_df
            
        except Exception as e:
            print(f"Error in process_training_predictions: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            raise

    def process_single_prediction_from_obj(self, hume_json):
        """
        Process predictions from HUME's Face, Prosody, and Language emotion models for a SINGLE video.
        Returns an object with the video name, timeline, averages, and top 3 emotions.
        The predictions will be added when the SVR model predicts the dataframes that are generated here.
        """        
        face_timeline = []
        prosody_timeline = []
        language_timeline = []
        
        face_data_emotions = defaultdict(list) #these dicts save some code by automatically creating an empty [] if a new 
        language_data_emotions = defaultdict(list) #emotion [] is inserted, immediately appending to the empty or existing one
        prosody_data_emotions = defaultdict(list) #each of these will store a list of emotion scores for each emotion, used to calculate aggregates
        
        # Get the first prediction from the results
        video = hume_json[0]['results']['predictions'][0]
        file_name = video.get("file", "unknown_file")
        
        # Process face predictions
        face_grouped_predictions = video.get("models", {}).get("face", {}).get("grouped_predictions", [])
        for group in face_grouped_predictions:
            for face_prediction in group.get("predictions", []):
                emotions = {}
                for emotion in face_prediction.get("emotions", []):
                    face_data_emotions[emotion.get("name")].append(emotion.get("score", 0.0))
                    emotions[emotion.get("name")] = emotion.get("score", 0.0)
                face_timeline.append({
                        'time': face_prediction.get('time', {}),
                        'frame': face_prediction.get('frame', {}),
                        'emotions': emotions,
                        'id': group.get('id')
                    })

        # Process language predictions
        lang_grouped_predictions = video.get("models", {}).get("language", {}).get("grouped_predictions", [])
        for group in lang_grouped_predictions:
            for language_prediction in group.get("predictions", []):
                emotions = {}
                for emotion in language_prediction.get("emotions", []):
                    language_data_emotions[emotion.get("name")].append(emotion.get("score", 0.0))
                    emotions[emotion.get("name")] = emotion.get("score", 0.0)
                entry = {
                    'position': {
                        'begin': language_prediction.get('position', {}).get('begin'),
                        'end': language_prediction.get('position', {}).get('end')
                    },
                    'emotions': emotions,
                    'text': language_prediction.get('text')
                }
                if hasattr(language_prediction, 'time'):
                    entry['time_start'] = language_prediction.time.begin
                    entry['time_end'] = language_prediction.time.end
                language_timeline.append(entry)

        # Process prosody predictions
        prosody_data = video.get("models", {}).get("prosody", {}).get("grouped_predictions", [])
        for group in prosody_data:
            for prosody_prediction in group.get("predictions", []):
                emotions = {}
                for emotion in prosody_prediction.get("emotions", []):
                    prosody_data_emotions[emotion.get("name")].append(emotion.get("score", 0.0))
                    emotions[emotion.get("name")] = emotion.get("score", 0.0)
                prosody_timeline.append({
                                'time_start': prosody_prediction.get('time', {}).get('begin'),
                                'time_end': prosody_prediction.get('time', {}).get('end'),
                                'emotions': emotions,
                                'id': group.get('id'),
                                'text': prosody_prediction.get('text')
                            })
            

        #now that we have our dict of emotion lists for this video (across all 3 modalities)
        #generate a dict of aggregates for each emotion
        face_aggregates = self.calculate_aggregates(face_data_emotions)
        language_aggregates = self.calculate_aggregates(language_data_emotions)
        prosody_aggregates = self.calculate_aggregates(prosody_data_emotions)
        
        # Create DataFrames with the aggregates as columns
        self.model_input_dataframes = {
            'face': pd.DataFrame([face_aggregates]),
            'language': pd.DataFrame([language_aggregates]),
            'prosody': pd.DataFrame([prosody_aggregates])
        }
        
        # Add video_id column
        for df in self.model_input_dataframes.values():
            df['video_id'] = file_name
        
        #sum up the emotions for all modalities, used for the top 3 emotions, and average scores
        summed_emotions = {
            'face': {emotion_name: sum(scores_arr) for emotion_name, scores_arr in face_data_emotions.items()},
            'language': {emotion_name: sum(scores_arr) for emotion_name, scores_arr in language_data_emotions.items()},
            'prosody': {emotion_name: sum(scores_arr) for emotion_name, scores_arr in prosody_data_emotions.items()}
        }
            
        # Initialize combined scores and modality counts
        combined_scores = {}
        modality_counts = {}

        # Combine scores and count modalities
        for modality, emotions in summed_emotions.items():
            for emotion, summed_score in emotions.items():
                if emotion in combined_scores:
                    combined_scores[emotion] += summed_score
                    modality_counts[emotion] += 1
                else:
                    combined_scores[emotion] = summed_score
                    modality_counts[emotion] = 1

        # Calculate average scores by dividing by number of modalities
        averaged_scores = {
            emotion: score / modality_counts[emotion]
            for emotion, score in combined_scores.items()
        }
        
        #sort the emotions by score, used for the top 3 emotions
        all_avg_emotions = dict(sorted(averaged_scores.items(), key=lambda x: x[1], reverse=True))

        # Get top 3 emotions
        top_3_combined_emotions = dict(list(all_avg_emotions.items())[:3])
            
            
        return { # return None for the SVR predictions, as they will be filled in by the SVR model
                'file_name': file_name,
                'timeline': {
                    'face': sorted(face_timeline, key=lambda x: x['time']),
                    'prosody': sorted(prosody_timeline, key=lambda x: x['time_start']),
                    'language': sorted(language_timeline, key=lambda x: x.get('time_start', x['position']['begin']))
                },
                'averages': {
                    'face': {
                        'emotion_aggregates': face_data_emotions,
                        'aggregate_score': None
                    },
                    'prosody': {
                        'emotion_aggregates': prosody_data_emotions, 
                        'aggregate_score': None
                        },
                    'language': {
                        'emotion_aggregates': language_data_emotions,
                        'aggregate_score': None
                    }
                },
                'overall_score': None,
                'top_emotions': top_3_combined_emotions
        }
    
    def load_labels_supabase(self):
        """
        Fetch all labeled CSV files from Supabase and return it as a DataFrame.
        """
        print("Loading labels...")
        print("Accessing Supabase...")
        supabase = SupabaseDB()
        files = supabase.list_storage_files(bucket_id="Interviews", folder_path="Labels")
        print(files)
        if not files:
            print("No label files found.")
            return pd.DataFrame()

        labeled_urls = supabase.create_signed_url(bucket_id="Interviews", files=files, expires_in=3600)[1]

        try:
            response = requests.get(labeled_urls, stream=True)
            response.raise_for_status()
            temp_file_path = os.path.join("/tmp", "turker_scores_full_interview.csv")
            with open(temp_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            df_labels = pd.read_csv(temp_file_path)
            # Only keep rows where Worker equals "AGGR"
            df_labels = df_labels[df_labels['Worker'].str.upper() == "AGGR"].copy()

            # Create a new column "participant_id" to match the JSON (e.g., "PP1" -> "pp1")
            df_labels['participant_id'] = df_labels['Participant'].str.lower()

            
            #keep only the participant ID and the now Aggregate ReccomendHiring Score 
            df_labels = df_labels[['participant_id', 'RecommendHiring']]
            
            # Set participant_id as the index
            df_labels.set_index('participant_id', inplace=True)
            
            print("Labels Retrieved...")
            return df_labels
        except Exception as e:
            print(f"Error loading labeled scores: {e}")
            return pd.DataFrame()
    
    def merge_data_with_labels(self, face_df, prosody_df, language_df, labels_df):
        """
        Merge the modality DataFrames with the labels DataFrame.
        Returns three merged DataFrames and the target array.
        """
        # First rename video_id to participant_id
        face_df = face_df.rename(columns={'video_id': 'participant_id'})
        prosody_df = prosody_df.rename(columns={'video_id': 'participant_id'})
        language_df = language_df.rename(columns={'video_id': 'participant_id'})
        
        # Clean up participant IDs in all DataFrames
        face_df['participant_id'] = face_df['participant_id'].str.replace('.mp4', '').str.lower()
        prosody_df['participant_id'] = prosody_df['participant_id'].str.replace('.mp4', '').str.lower()
        language_df['participant_id'] = language_df['participant_id'].str.replace('.mp4', '').str.lower()

        
        # Merge each modality DataFrame with labels
        face_merged = face_df.merge(labels_df, left_on='participant_id', right_index=True, how='inner')
        prosody_merged = prosody_df.merge(labels_df, left_on='participant_id', right_index=True, how='inner')
        language_merged = language_df.merge(labels_df, left_on='participant_id', right_index=True, how='inner')
        
        print(f"\nNumber of matches found:")
        print(f"Face: {len(face_merged)}")
        print(f"Prosody: {len(prosody_merged)}")
        print(f"Language: {len(language_merged)}")
        
        # Extract target values
        target = face_merged['RecommendHiring'].values
        
        # Drop the target column and participant_id from the DataFrames
        face_merged = face_merged.drop(columns=['RecommendHiring', 'participant_id'])
        prosody_merged = prosody_merged.drop(columns=['RecommendHiring', 'participant_id'])
        language_merged = language_merged.drop(columns=['RecommendHiring', 'participant_id'])
        
        # Verify all columns are numeric
        print("\nVerifying numeric columns:")
        print("Face columns:", face_merged.dtypes)
        print("Prosody columns:", prosody_merged.dtypes)
        print("Language columns:", language_merged.dtypes)
        
        return face_merged, prosody_merged, language_merged, target
    
    def prepare_model_input(self, face_df, prosody_df, language_df):
        """
        Prepare the final input format for the MultiModalRegressor.
        Returns a dictionary of DataFrames and the target array.
        """
        # Load labels
        labels_df = self.load_labels_local("/Users/abdelazimlokma/Desktop/Desktop/CS Projects/Talentora/talentora-analysisbot/app/services/mmr/turker_scores_full_interview.csv")
    
        # Merge with labels
        face_merged, prosody_merged, language_merged, target = self.merge_data_with_labels(
            face_df, prosody_df, language_df, labels_df
        )
        
        # Create the model input dictionary
        model_input = {
            'face': face_merged,
            'prosody': prosody_merged,
            'language': language_merged
        }
        
        self.model_labels = target
        
        return model_input, target
    
    def prepare_single_prediction(self, model_input_dataframes):
        """
        Prepare a single prediction's dataframes to match the model's expected format.
        
        Args:
            model_input_dataframes (dict): Dictionary containing DataFrames for each modality

            
        Returns:
            dict: Processed DataFrames ready for model prediction
        """
        processed_data = {}
        
        for modality, df in model_input_dataframes.items():
            # Create a copy to avoid modifying the original
            processed_df = df.copy()
            
            # Rename video_id to participant_id and set its value
            processed_df = processed_df.rename(columns={'video_id': 'participant_id'})
            processed_df['participant_id'] = processed_df['participant_id'].str.replace('.mp4', '').str.lower()
            
            # Remove any columns that aren't features or participant_id
            feature_columns = [col for col in processed_df.columns 
                             if col != 'participant_id' and not col.startswith('Unnamed')]
            processed_df = processed_df[['participant_id'] + feature_columns]
            
            # Ensure all columns are numeric except participant_id
            for col in feature_columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            processed_data[modality] = processed_df
            
        return processed_data
    

    def update_prediction_results(self, json_results, overall_score, svr_predictions):
        """
        Update the JSON results with the overall score and individual SVR predictions.
        
        Args:
            json_results (dict): The original JSON results from process_single_prediction_from_json
            overall_score (float): The overall prediction score
            svr_predictions (dict): Dictionary of predictions for each modality
            
        Returns:
            dict: Updated JSON results
        """
        # Update overall score
        json_results['overall_score'] = overall_score
        
        # Update individual modality scores
        for mod, score in svr_predictions.items():
            if mod in json_results['averages']:
                json_results['averages'][mod]['aggregate_score'] = score
        
        return json_results
    
    def load_labels_local(self, file_path):
        """
        Load labeled data from a local CSV file.
        
        Args:
            file_path (str): Path to the local CSV file containing labeled data
            
        Returns:
            pd.DataFrame: DataFrame with labels, indexed by participant_id
        """
        print("Loading labels from local file...")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return pd.DataFrame()
            
            # Read the CSV file
            df_labels = pd.read_csv(file_path)
            print(f"Loaded CSV with {len(df_labels)} rows")
            
            # Only keep rows where Worker equals "AGGR"
            if 'Worker' in df_labels.columns:
                df_labels = df_labels[df_labels['Worker'].str.upper() == "AGGR"].copy()
                print(f"Filtered to {len(df_labels)} aggregated rows")

            # Create a new column "participant_id" to match the JSON (e.g., "PP1" -> "pp1")
            if 'Participant' in df_labels.columns:
                df_labels['participant_id'] = df_labels['Participant'].str.lower()
            else:
                print("Warning: 'Participant' column not found, assuming participant_id exists")
            
            # Keep only the participant ID and the RecommendHiring Score
            if 'RecommendHiring' in df_labels.columns and 'participant_id' in df_labels.columns:
                df_labels = df_labels[['participant_id', 'RecommendHiring']]
            else:
                print("Warning: Required columns not found. Available columns:", df_labels.columns)
                return pd.DataFrame()
            
            # Set participant_id as the index
            df_labels.set_index('participant_id', inplace=True)
            
            print("Labels loaded successfully")
            return df_labels
        
        except Exception as e:
            print(f"Error loading labeled scores from local file: {str(e)}")
            return pd.DataFrame()
    
