# ml-service/schemas/request.py

from pydantic import BaseModel

class ClassificationRequest(BaseModel):
    text: str
