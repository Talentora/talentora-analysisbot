# config.py
import os
from dotenv import load_dotenv

load_dotenv()

HUME_API_KEY = os.environ.get("HUME_API_KEY")
DEFAULT_JOB_MONITOR_INTERVAL = 10

# Emotion weights configuration
EMOTION_WEIGHTS = {
    # Positive Emotions
    "Admiration": 0.5,
    "Adoration": 0.5,
    "Aesthetic Appreciation": 0.4,
    "Amusement": 0.6,
    "Awe": 0.5,
    "Calmness": 0.7,
    "Concentration": 0.6,
    "Contemplation": 0.4,
    "Contentment": 0.7,
    "Determination": 0.6,
    "Enthusiasm": 0.6,
    "Interest": 0.7,
    "Joy": 0.8,
    "Love": 0.5,
    "Nostalgia": 0.4,
    "Realization": 0.5,
    "Triumph": 0.6,

    # Negative Emotions
    "Anger": -0.6,
    "Annoyance": -0.5,
    "Anxiety": -0.5,
    "Awkwardness": -0.4,
    "Boredom": -0.5,
    "Confusion": -0.6,
    "Contempt": -0.6,
    "Disappointment": -0.5,
    "Disapproval": -0.6,
    "Disgust": -0.7,
    "Distress": -0.5,
    "Doubt": -0.5,
    "Embarrassment": -0.4,
    "Empathic Pain": -0.3,
    "Fear": -0.6,
    "Guilt": -0.5,
    "Horror": -0.7,
    "Sadness": -0.6,
    "Shame": -0.5,
    "Surprise (negative)": -0.4,
    "Surprise (positive)": 0.3,
    "Sympathy": 0.2,
    "Tiredness": -0.3,

    # Neutral or Context-Dependent Emotions
    "Craving": 0.0,
    "Desire": 0.0,
    "Excitement": 0.4,
    "Pain": -0.4,
    "Satisfaction": 0.5,
    "Romance": 0.0,
}