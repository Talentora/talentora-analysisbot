from typing import List, Dict
from app.configs.hume_config import HUME_API_KEY
from app.services.sentiment_analysis import EmotionAnalyzer
from app.controllers.hume_job_manager import JobManager
from hume import HumeClient

def run_emotion_analysis(media_urls: List[str], models: Dict):
    """Main function to run emotion analysis on media files."""
    # Initialize components
    client = HumeClient(api_key=HUME_API_KEY)
    job_manager = JobManager(client)
    emotion_analyzer = EmotionAnalyzer(HUME_API_KEY)

    # Start job
    job_id = job_manager.start_job(urls=media_urls, models=models)
    if not job_id:
        return None

    # Monitor job
    status = job_manager.monitor_job(job_id)
    if status != "COMPLETED":
        print("Job failed or was interrupted")
        return None

    # Get predictions and analyze
    predictions = job_manager.get_job_predictions(job_id)
    if not predictions:
        return None

    # Process and return results
    return emotion_analyzer.process_predictions(predictions)

def main():
    # Example usage
    media_urls = [
        "https://www.dropbox.com/scl/fi/eesz1rkgpxjdl81m9qu51/P7.avi?rlkey=w7ewszrtzf9m64x1yx1ajzobp&st=tsk6loki&dl=1"
    ]
    
    models = {
        "face": {},
        "language": {},
        "prosody": {}
    }

    results = run_emotion_analysis(media_urls, models)
    
    if results:
        print("\nAnalysis Results:")
        print(f"Processed {results['metadata']['frame_count']} frames")
        print(f"\nFace Emotion Aggregate Score: {results['face']['aggregate_score']}/10")
        print(f"Prosody Emotion Aggregate Score: {results['prosody']['aggregate_score']}/10")
        
        print("\nDetailed Face Emotions:")
        for emotion, score in results['face']['average_emotions'].items():
            print(f"  {emotion}: {score:.4f}")
            
        print("\nDetailed Prosody Emotions:")
        for emotion, score in results['prosody']['average_emotions'].items():
            print(f"  {emotion}: {score:.4f}")

if __name__ == "__main__":
    main()