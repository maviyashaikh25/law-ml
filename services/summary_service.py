from transformers import pipeline

print("Loading Summarization Model (sshleifer/distilbart-cnn-12-6)...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(text: str) -> str:
    try:
        # split text if it's too long (rudimentary) - but for now let's rely on the pipeline's truncation
        # max_length=150 is usually good for a concise summary
        # min_length=40 ensures we get some substance
        
        # We need to handle max input length for BERT-based models (usually 1024 tokens)
        # The pipeline handles truncation but explicitly setting it is safer
        
        result = summarizer(text, max_length=150, min_length=40, do_sample=False, truncation=True)
        summary_text = result[0]['summary_text']
        
        # Format as bullet points
        sentences = summary_text.replace('. ', '.\n').split('\n')
        formatted_summary = "\n".join([f"• {s.strip()}" for s in sentences if s.strip()])
        
        return formatted_summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""
