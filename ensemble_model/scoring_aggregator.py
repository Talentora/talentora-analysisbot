# scoring_aggregator.py
import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.
    """
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def scale_similarity_to_score(similarity: float) -> float:
    """
    Scale cosine similarity (assumed to be between 0 and 1) to a 0-10 score.
    """
    return similarity * 10

def compute_fit_score(candidate_embedding: np.ndarray, target_embedding: np.ndarray) -> float:
    """
    Compute the fit score based on cosine similarity and scale it.
    """
    sim = cosine_similarity(candidate_embedding, target_embedding)
    return scale_similarity_to_score(sim)

if __name__ == "__main__":
    # Create dummy embeddings for testing
    candidate = np.random.rand(384)
    target = np.random.rand(384)
    print("Fit Score:", compute_fit_score(candidate, target))