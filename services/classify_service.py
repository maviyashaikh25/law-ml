# ml-service/services/classify_service.py

import torch
from models.classifier import model, tokenizer

LABELS = ["Judgment", "Contract", "Legal Notice", "Other"]

def classify_text(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)

    return {
        "document_type": LABELS[predicted_class.item()],
        "confidence": round(confidence.item(), 3)
    }
