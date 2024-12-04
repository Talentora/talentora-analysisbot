from typing import Dict, List, Optional, Union
import time
from hume import HumeClient
from app.configs.hume_config import EMOTION_WEIGHTS

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

        normalized_score = 10 * (total - min_possible) / (max_possible - min_possible)
        return round(max(0, min(10, normalized_score)), 2)

    def process_predictions(self, predictions) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
        """
        Process predictions and return accumulated emotion scores.
        """
        face_accumulated_emotions = {}
        prosody_accumulated_emotions = {}
        language_accumulated_emotions = {}
        face_frame_count = 0
        prosody_frame_count = 0
        language_count = 0

        for prediction in predictions:
            for result in prediction.results.predictions:
                # Process Face Model Predictions
                if result.models.face:
                    for face in result.models.face.grouped_predictions:
                        for pred in face.predictions:
                            emotions = {emotion.name: emotion.score for emotion in pred.emotions}
                            for emotion, score in emotions.items():
                                face_accumulated_emotions[emotion] = face_accumulated_emotions.get(emotion, 0.0) + score
                            face_frame_count += 1

                # Process Prosody Model Predictions
                if result.models.prosody:
                    for prosody in result.models.prosody.grouped_predictions:
                        for pred in prosody.predictions:
                            emotions = {emotion.name: emotion.score for emotion in pred.emotions}
                            for emotion, score in emotions.items():
                                prosody_accumulated_emotions[emotion] = prosody_accumulated_emotions.get(emotion, 0.0) + score
                            prosody_frame_count += 1
                                
                if result.models.language:
                    for language in result.models.language.grouped_predictions:
                        for pred in language.predictions:
                            emotions = {emotion.name: emotion.score for emotion in pred.emotions}
                            for emotion, score in emotions.items():
                                language_accumulated_emotions[emotion] = language_accumulated_emotions.get(emotion, 0.0) + score
                            language_count += 1

        if face_frame_count == 0 and prosody_frame_count == 0 and language_count == 0:
            return {}

        # Calculate average emotions
        face_average_emotions = {emotion: total_score / face_frame_count 
                               for emotion, total_score in face_accumulated_emotions.items()}
        prosody_average_emotions = {emotion: total_score / prosody_frame_count
                                  for emotion, total_score in prosody_accumulated_emotions.items()}
        language_average_emotions = {emotion: total_score / language_count
                                     for emotion, total_score in language_accumulated_emotions.items()}

        return {
            'face': {
                'average_emotions': face_average_emotions,
                'aggregate_score': self.aggregate_emotion_score(face_average_emotions)
            },
            'prosody': {
                'average_emotions': prosody_average_emotions,
                'aggregate_score': self.aggregate_emotion_score(prosody_average_emotions)
            },
            'language': {
                'average_emotions': language_average_emotions,
                'aggregate_score': self.aggregate_emotion_score(language_average_emotions)
            },
            'metadata': {
                'face_frame_count': face_frame_count,
                'prosody_frame_count': prosody_frame_count,
                'language_count': language_count
            }
        }