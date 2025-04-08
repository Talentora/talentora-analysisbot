import pandas as pd
import joblib
import numpy as np
from ensemble_model.job_processor import JobProcessor


def get_soft_skills_score(applicant_video_id: str, hume_json_path: str) -> float:
    """
    Process HUME emotion analysis data to generate a soft skills score for a specific video.
    
    Args:
        applicant_video_id (str): The ID of the video to score (e.g., 'video1.mp4')
        hume_json_path (str): Path to the HUME JSON analysis file
        
    Returns:
        float: Soft skills score between 0 and 10
    """
    job_processor = JobProcessor()
    
    # Process the HUME JSON data
    face_df, prosody_df, language_df = job_processor.process_predictions(hume_json_path)
    
    # Filter dataframes for the specific video
    face_df = face_df[face_df['video_id'] == applicant_video_id]
    prosody_df = prosody_df[prosody_df['video_id'] == applicant_video_id]
    language_df = language_df[language_df['video_id'] == applicant_video_id]
    
    if face_df.empty or prosody_df.empty or language_df.empty:
        raise ValueError(f"No data found for video {applicant_video_id}")
    
    # Load the pre-trained MultiModalRegressor model
    try:
        model_file = 'multimodalregressor_model.pkl'
        mmr = joblib.load(model_file)
    except FileNotFoundError:
        raise FileNotFoundError("MultiModalRegressor model file not found. Please ensure the model is trained and saved.")
    
    # Prepare input data for prediction
    modalities = ['face', 'prosody', 'language']
    input_data = {
        'face': face_df,
        'prosody': prosody_df,
        'language': language_df
    }
    
    # Generate features for prediction
    features = np.column_stack([
        mmr.svr_models[mod].predict(input_data[mod][mmr.selected_features[mod]])
        for mod in modalities
    ])

    # Predict the soft skills score
    prediction = mmr.predict(features)

    # Return the score rounded to 2 decimal places
    return float(round(prediction[0], 2))


if __name__ == "__main__":
    # For testing
    test_video_id = "video1.mp4"
    test_json_path = "hume_analysis.json"
    
    try:
        score = get_soft_skills_score(test_video_id, test_json_path)
        print(f"Soft Skills Score for video {test_video_id}: {score}")
    except Exception as e:
        print(f"Error: {str(e)}")