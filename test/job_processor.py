import os
import statistics
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import ast

import requests

from app.controllers.supabase_db import SupabaseDB

class JobProcessor():
    """
        Processes HUME emotion analysis jobs to compute statistical aggregates.

        The JobProcessor class takes in emotion analysis data and computes
        statistical aggregates (mean, max, min, standard deviation) for each
        emotion across different modalities (face, prosody, language).

        Attributes:
            job (list): A list containing video analysis data.
            face_aggregates (dict): Aggregated emotion data for the face modality.
            prosody_aggregates (dict): Aggregated emotion data for the prosody modality.
            language_aggregates (dict): Aggregated emotion data for the language modality.

        Methods:
            __init__(self, data):
                Initializes the JobProcessor with the provided data.
            calculate_aggregates(self, emotion_scores):
                Calculates statistical aggregates for each emotion's scores.
            process_predictions(self, write_to_csv=False):
                Processes predictions to compute aggregates and optionally writes them to CSV files.
    """
    def __init__(self):
        self.face_aggregates = {}
        self.prosody_aggregates = {}
        self.language_aggregates = {}
        self.dataframes = {}
        
    def load_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    

    def calculate_aggregates(self, emotion_scores):
        """
        Calculate statistical aggregates for each emotion's scores.

        This method computes the mean, maximum, minimum, and standard deviation
        for each emotion based on the provided scores.

        Args:
            emotion_scores (dict): A dictionary where each key is an emotion name (str),
                and the corresponding value is a list of numerical scores (List[float])
                associated with that emotion.

        Returns:
            dict: A dictionary where each key is an emotion name (str), and the
                corresponding value is another dictionary containing the calculated
                aggregates:
                    - 'mean' (float): The average score for the emotion.
                    - 'max' (float): The highest score for the emotion.
                    - 'min' (float): The lowest score for the emotion.
                    - 'std_dev' (float): The standard deviation of the scores for the emotion.
        """
        aggregates = {}
        for emotion, scores in emotion_scores.items():
            if scores:  # Ensure the list is not empty
                aggregates[emotion] = {
                    'mean': statistics.mean(scores),
                    'max': max(scores),
                    'min': min(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0
                }
            else:
                aggregates[emotion] = {
                    'mean': 0.0,
                    'max': 0.0,
                    'min': 0.0,
                    'std_dev': 0.0
                } 
        return aggregates
    

    def process_predictions(self, data, write_to_csv = False):
        """
        Process predictions from HUME's Face, Prosody, and Language emotion models.
        Returns three pandas DataFrames: face_df, prosody_df, language_df.
        Each DataFrame has one row per video and columns for each emotion's mean, max, min, and std.
        """
        
        for video_analysis in data:
            for video_dict in video_analysis: #this loop is one video
                predictions = video_analysis.get("results", {}).get("predictions", [])
                for prediction in predictions:
                    file_name = prediction.get("file", "unknown_file")
                    face_data_emotions = defaultdict(list) #these dicts save some code by automatically creating an empty [] if a new 
                    language_data_emotions = defaultdict(list) #emotion [] is inserted, immediately appending to the empty or existing one
                    prosody_data_emotions = defaultdict(list)

                    #fetch the predictions for the face model, return [] if not found, loop below wont execute in that case
                    face_data = prediction.get("models", {}).get("face", {}).get("grouped_predictions", [])
                    for group in face_data:
                        for face_prediction in group.get("predictions", []):
                            for emotion in face_prediction.get("emotions", []):
                                face_data_emotions[emotion.get("name")].append(emotion.get("score", 0.0))
                                #for each emotion dict, get its name and score, fetch the corresponding emotion
                                #list from face_data_emotions and add the emotion score to the list (default to 0.0 if emotion score isnt found)
                                #this logic also repeats for language and prosody 

                    language_data = prediction.get("models", {}).get("language", {}).get("grouped_predictions", [])
                    for group in language_data:
                        for language_prediction in group.get("predictions", []):
                            for emotion in language_prediction.get("emotions", []):
                                language_data_emotions[emotion.get("name")].append(emotion.get("score", 0.0))
                    
                    prosody_data = prediction.get("models", {}).get("prosody", {}).get("grouped_predictions", [])
                    for group in prosody_data:
                        for prosody_prediction in group.get("predictions", []):
                            for emotion in prosody_prediction.get("emotions", []):
                                prosody_data_emotions[emotion.get("name")].append(emotion.get("score", 0.0))
            
                #now that we have our dict of emotion lists for this video (across all 3 modalities)
                #generate a dict of aggregates for each emotion and set that as a value for the aggregates dictionairy, where the
                #key is the video name - example: {"video_1.mp4" : {"admiration" : {"mean" : 0.0, "min" : 0.0, "max" : 0.0,...}, ...}, ...}                            
            self.face_aggregates[file_name] = self.calculate_aggregates(face_data_emotions) 
            self.language_aggregates[file_name] = self.calculate_aggregates(language_data_emotions)
            self.prosody_aggregates[file_name] = self.calculate_aggregates(prosody_data_emotions)
        
        #create DataFrames for each modality. Each DataFrame will have video_id as index.
        face_df = pd.DataFrame.from_dict(self.face_aggregates, orient='index')
        prosody_df = pd.DataFrame.from_dict(self.prosody_aggregates, orient='index')
        language_df = pd.DataFrame.from_dict(self.language_aggregates, orient='index')
        
        #convert index to a column named 'video_id' for clarity
        face_df = face_df.reset_index().rename(columns={'index': 'video_id'})
        prosody_df = prosody_df.reset_index().rename(columns={'index': 'video_id'})
        language_df = language_df.reset_index().rename(columns={'index': 'video_id'})
        
        if write_to_csv:
            face_df.to_csv('face_predictions.csv', index=False)
            prosody_df.to_csv('prosody_predictions.csv', index=False)
            language_df.to_csv('language_predictions.csv', index=False)
            print(f"Generated {len(face_df)} number of samples for face_df")
            print(f"Generated {len(prosody_df)} number of samples for prosody_df")
            print(f"Generated {len(language_df)} number of samples for language_df")
        
        return face_df, prosody_df, language_df
    
    def load_labels(self):
        """
        Fetch the labeled CSV file from Supabase and return it as a DataFrame.
        """
        print("Loading labels...")
        print("Accessing Supabase...")
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
            # Only keep rows where Worker equals "AGGR"
            df_labels = df_labels[df_labels['Worker'].str.upper() == "AGGR"].copy()

            # Create a new column "participant_id" to match the JSON (e.g., "PP1" -> "pp1")
            df_labels['participant_id'] = df_labels['Participant'].str.lower()

            
            #keep only the participant ID and the now Aggregate ReccomendHiring Score 
            df_labels = df_labels[['participant_id', 'RecommendHiring']]
            print("Labels Retrieved...")
            return df_labels
        except Exception as e:
            print(f"Error loading labeled scores: {e}")
            return pd.DataFrame()
        
    def merge_data(self, labels, face, prosody, language):
        print("Merging data and labels...")
      
        # Create participant_id columns from video_id
        face['participant_id'] = face['video_id'].str.removesuffix('.mp4').str.lower()
        prosody['participant_id'] = prosody['video_id'].str.removesuffix('.mp4').str.lower()
        language['participant_id'] = language['video_id'].str.removesuffix('.mp4').str.lower()

        # Merge each modality with labels
        merged_lang = pd.merge(language, labels, on="participant_id", how="inner")
        merged_face = pd.merge(face, labels, on="participant_id", how="inner")
        merged_pros = pd.merge(prosody, labels, on="participant_id", how="inner")
        
        # Optionally, set the video_id as the index to keep track of each video.
        # (Make sure 'video_id' is still in the merged DataFrame. If not, you might use 'participant_id' or keep video_id before merging.)
        merged_lang = merged_lang.set_index("participant_id")
        merged_face = merged_face.set_index("participant_id")
        merged_pros = merged_pros.set_index("participant_id")
        
        merged_lang = merged_lang.drop(columns=["video_id"], errors="ignore")
        merged_face = merged_face.drop(columns=["video_id"], errors="ignore")
        merged_pros = merged_pros.drop(columns=["video_id"], errors="ignore")
        
        return merged_face, merged_pros, merged_lang
    
    def generate_mmr_compatible_input(self, merged_face, merged_pros, merged_lang):
        """
        Takes merged data for facial, prosody, and language modalities,
        verifies that they have the same number of samples, stores them
        in a dictionary with modality names as keys, and extracts the labels.
        
        Assumes that the label column is named 'RecommendHiring' in the merged facial DataFrame.
        
        Returns:
            tuple: (data_dict, labels)
                data_dict: A dictionary with keys 'facial', 'prosody', and 'language'
                        containing the corresponding merged DataFrames with the label column removed.
                labels: A pandas Series with the label values.
        """
        # Check that each modality has the same number of samples
        n_face = merged_face.shape[0] if hasattr(merged_face, 'shape') else len(merged_face)
        n_pros = merged_pros.shape[0] if hasattr(merged_pros, 'shape') else len(merged_pros)
        n_lang = merged_lang.shape[0] if hasattr(merged_lang, 'shape') else len(merged_lang)
        
        if not (n_face == n_pros == n_lang):
            raise ValueError("All modalities must have the same number of samples.")
        
        # Extract labels from the facial dataframe
        if 'RecommendHiring' in merged_face.columns:
            labels = merged_face['RecommendHiring']
            # Remove the label column from the facial dataframe
            merged_face = merged_face.drop(columns=['RecommendHiring'])
        else:
            raise ValueError("Label column 'RecommendHiring' not found in merged data. Please verify the CSV or merge function.")
        
        # Remove the label column from other modalities if it exists
        merged_pros = merged_pros.drop(columns=['RecommendHiring'], errors='ignore')
        merged_lang = merged_lang.drop(columns=['RecommendHiring'], errors='ignore')
        
        # Store the merged data in a dictionary
        self.dataframes = {
            'facial': merged_face,
            'prosody': merged_pros,
            'language': merged_lang
        }
        
        return self.dataframes, labels
    
    def expand_dict_columns(self, df):
        """
        For any column in df that contains string representations of dictionaries,
        parse the strings into dictionaries and expand them into separate columns.
        The new columns will be prefixed by the original column name.
        """
        for col in df.columns:
            # Only consider object-type columns.
            if df[col].dtype == 'object':
                non_null = df[col].dropna()
                if not non_null.empty:
                    sample = non_null.iloc[0]
                    # Check if the string looks like a dictionary
                    if isinstance(sample, str) and sample.strip().startswith("{") and sample.strip().endswith("}"):
                        try:
                            # Convert the column values to dictionaries
                            parsed = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})
                            # Expand the dictionaries into a DataFrame
                            expanded = parsed.apply(pd.Series)
                            # Optionally, add a prefix to the new columns
                            expanded = expanded.add_prefix(f"{col}_")
                            # Drop the original column and join the new columns
                            df = df.drop(columns=[col]).join(expanded)
                        except Exception as e:
                            print(f"Error expanding column {col}: {e}")
        return df
    
    def generate_dataframes_from_csv(self, face_csv_path, pros_csv_path, lang_csv_path, 
                                    face_csv_path2=None, pros_csv_path2=None, lang_csv_path2=None):
        """
        Reads CSV files for facial, prosody, and language modalities, converts them into pandas DataFrames,
        verifies that each set (if only one set is provided, or each set separately if two sets are provided)
        has the same number of samples, optionally concatenates a second set of CSVs into the first set,
        and expands any columns that contain dictionary strings (e.g., emotion aggregates) into separate numeric columns.

        Parameters:
            face_csv_path (str): File path to the CSV file with facial features (set 1).
            pros_csv_path (str): File path to the CSV file with prosody features (set 1).
            lang_csv_path (str): File path to the CSV file with language features (set 1).
            face_csv_path2 (str, optional): File path to a second CSV file with facial features.
            pros_csv_path2 (str, optional): File path to a second CSV file with prosody features.
            lang_csv_path2 (str, optional): File path to a second CSV file with language features.

        Returns:
            tuple: A tuple (df_face, df_pros, df_lang) containing the processed (and merged, if applicable) DataFrames.
        """
        # Read the first set of CSV files into DataFrames
        print("Generating DataFrames from saved CSV files (set 1)...")
        df_face = pd.read_csv(face_csv_path)
        df_pros = pd.read_csv(pros_csv_path)
        df_lang = pd.read_csv(lang_csv_path)

        # Check that set 1 DataFrames have the same number of rows
        if not (len(df_face) == len(df_pros) == len(df_lang)):
            raise ValueError("Set 1: All CSV files must have the same number of rows/samples.")
        
        # If a second set of CSV paths is provided, read and concatenate them
        if face_csv_path2 and pros_csv_path2 and lang_csv_path2:
            print("Generating DataFrames from saved CSV files (set 2)...")
            df_face2 = pd.read_csv(face_csv_path2)
            df_pros2 = pd.read_csv(pros_csv_path2)
            df_lang2 = pd.read_csv(lang_csv_path2)
            
            # Check that set 2 DataFrames have the same number of rows
            if not (len(df_face2) == len(df_pros2) == len(df_lang2)):
                raise ValueError("Set 2: All CSV files must have the same number of rows/samples.")
            
            print("Concatenating set 1 and set 2 DataFrames...")
            df_face = pd.concat([df_face, df_face2], ignore_index=True)
            df_pros = pd.concat([df_pros, df_pros2], ignore_index=True)
            df_lang = pd.concat([df_lang, df_lang2], ignore_index=True)
        
        print("All CSV files have been processed. Expanding dictionary columns...")
        # Expand any dictionary columns into separate columns
        df_face = self.expand_dict_columns(df_face)
        df_pros = self.expand_dict_columns(df_pros)
        df_lang = self.expand_dict_columns(df_lang)
        
        print("Dictionary expansion successful. Final face DataFrame preview:")
        print(df_face.head())
        
        return df_face, df_pros, df_lang


def main():
    # Specify the path to your JSON file
    json_file_path = '/Users/abdelazimlokma/Desktop/new_hume2.json'

    # Create an instance of JobProcessor
    processor = JobProcessor()

    # Load the JSON data
    json_data = processor.load_json_file(json_file_path)

    # Initialize the processor with the loaded data
    processor.job = json_data

    # Proceed with processing
    face_df, prosody_df, language_df = processor.process_predictions(processor.job, write_to_csv=True)
    # Further processing or analysis can be done here

if __name__ == "__main__":
    main()