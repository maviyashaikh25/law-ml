from models.embedder import model
import numpy as np

def generate_embedding(text: str):
    # normalize_embeddings=True provides better cosine similarity performance
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
