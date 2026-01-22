# ml-service/models/classifier.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4
)

model.eval()
