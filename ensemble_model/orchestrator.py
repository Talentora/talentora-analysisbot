# orchestrator.py
from data_ingestion import preprocess_text
from embedding_generation import generate_embedding
from soft_skills import get_soft_skills_score
from scoring_aggregator import compute_fit_score
import numpy as np

def run_candidate_pipeline(company_culture_text: str, job_traits_text: str,
                           interview_transcript: str, resume_text: str,
                           audio_features: dict) -> dict:
    """
    Orchestrate the candidate evaluation by processing inputs through
    each module and aggregating the scores.
    """
    # Preprocess text inputs
    culture_text = preprocess_text(company_culture_text)
    job_text = preprocess_text(job_traits_text)
    transcript = preprocess_text(interview_transcript)
    resume = preprocess_text(resume_text)
    
    # Generate embeddings for culture, job traits, and candidate transcript
    culture_embedding = generate_embedding(culture_text)
    job_embedding = generate_embedding(job_text)
    candidate_embedding = generate_embedding(transcript)
    
    # Compute similarity-based fit scores
    culture_fit_score = compute_fit_score(candidate_embedding, culture_embedding)
    job_trait_fit_score = compute_fit_score(candidate_embedding, job_embedding)
    
    # Soft skills score (from audio analysis)
    soft_skills_score = get_soft_skills_score(audio_features)
    
    # For technical fit, we simulate using resume embedding vs. a target "technical requirements" embedding.
    # In practice, use a dedicated resume analysis module.
    required_tech_text = "python, machine learning, data analysis"  # Example target text
    resume_embedding = generate_embedding(resume)
    tech_req_embedding = generate_embedding(required_tech_text)
    technical_score = compute_fit_score(resume_embedding, tech_req_embedding)
    
    # Aggregate results in a dictionary
    results = {
        "culture_fit_score": round(culture_fit_score, 2),
        "job_trait_fit_score": round(job_trait_fit_score, 2),
        "soft_skills_score": soft_skills_score,
        "technical_score": round(technical_score, 2)
    }
    
    return results

if __name__ == "__main__":
    # Sample input data
    company_culture = "We value innovation, teamwork, and transparency."
    job_traits = "The ideal candidate is proactive, detail-oriented, and adaptable."
    interview_transcript = ("I have always taken initiative in my work and believe in continuous learning. "
                            "I enjoy collaborating with teams and solving complex problems.")
    resume_text = "Experienced in Python, machine learning, and data analysis."
    audio_features = {}  # Replace with actual audio feature data in production
    
    scores = run_candidate_pipeline(company_culture, job_traits, interview_transcript, resume_text, audio_features)
    print("Candidate Evaluation Scores:")
    for key, value in scores.items():
        print(f"  {key}: {value}")