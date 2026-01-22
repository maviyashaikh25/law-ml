from fastapi import FastAPI
from schemas.request import ClassificationRequest
from schemas.embedding_request import EmbeddingRequest
from services.classify_service import classify_text
from services.embedding_service import generate_embedding
from services.summary_service import generate_summary
from schemas.summary_request import SummaryRequest

app = FastAPI(title="LawLens Document Classifier")

@app.post("/classify")
def classify_document(request: ClassificationRequest):
    return classify_text(request.text)

@app.post("/embed")
def embed_text(request: EmbeddingRequest):
    embedding = generate_embedding(request.text)
    return {
        "embedding": embedding,
        "dimensions": len(embedding)
    }

@app.post("/summarize")
def summarize_text(request: SummaryRequest):
    summary = generate_summary(request.text)
    return {"summary": summary}

@app.post("/extract_clauses")
def get_clauses(request: ClassificationRequest):
    from services.clause_service import extract_clauses
    return extract_clauses(request.text)
