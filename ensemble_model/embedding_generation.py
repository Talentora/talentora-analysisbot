from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained embedding model (adjust model name as needed)
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding for the input text using a pre-trained model.
    """
    return model.encode(text)

if __name__ == "__main__":
    sample = "example text for embedding"
    embedding = generate_embedding(sample)
    print("Embedding shape:", embedding.shape)