from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the model once at server start
model = SentenceTransformer(MODEL_NAME)
