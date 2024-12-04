from typing import Dict, List, Union
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

        if max_possible - min_possible == 0:
            return 0.0

        normalized_score = 10 * (total - min_possible) / (max_possible - min_possible)
        return round(max(0, min(10, normalized_score)), 2)

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
            'metadata': {
                'counts': counts
            }
        }