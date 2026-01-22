import torch
from sentence_transformers import SentenceTransformer, util

# Initialize the model (using the same one as embedding service if possible, or a lightweight one)
# 'all-MiniLM-L6-v2' is efficient and good for this.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define Anchor Sentences for each Clause Type
CLAUSE_ANCHORS = {
    "Confidentiality": [
        "The parties agree to keep all information confidential and not disclose it to third parties.",
        "Recipient shall not disclose the Disclosing Party’s Confidential Information.",
        "This Non-Disclosure Agreement shall govern the exchange of proprietary information."
    ],
    "Termination": [
        "This agreement may be terminated by either party with 30 days written notice.",
        "This Agreement shall terminate automatically upon the expiration of the term.",
        "Either party may terminate this agreement for cause immediately upon written notice."
    ],
    "Indemnification": [
        "The Service Provider agrees to indemnify and hold harmless the Client from any claims.",
        "Each party shall indemnify the other against any losses, damages, or liabilities.",
        "Client agrees to indemnify Provider against all claims arising from Client's use of services."
    ],
    "Governing Law": [
        "This Agreement shall be governed by and construed in accordance with the laws of India.",
        "Any disputes arising under this agreement shall be resolved in the courts of New York.",
        "The laws of the State of California shall govern the validity and interpretation of this Agreement."
    ],
    "Payment Terms": [
        "Client shall pay Provider the fees set forth in the attached Statement of Work.",
        "Invoices are due and payable within 30 days of receipt.",
        "All payments shall be made in US Dollars via wire transfer."
    ],
    "Liability": [
        "In no event shall either party be liable for any indirect, special, or consequential damages.",
        "The total liability of the Service Provider shall not exceed the fees paid by the Client.",
        "Limitation of Liability: Neither party shall be liable for lost profits."
    ],
    "Non-Compete": [
        "Employee agrees not to compete with the Company for a period of 12 months following termination.",
        "During the term of this Agreement and for a period of one year thereafter, Employee shall not engage in any business competing with the Company.",
        "Employee shall not solicit any customers or employees of the Company."
    ],
    "Severance": [
        "In the event of termination without cause, Employee shall be entitled to severance pay.",
        "Company shall pay Employee a lump sum equal to three months of base salary.",
        "Severance package includes continuation of benefits for a period of 3 months."
    ],
    "IP Assignment": [
        "All work product created during employment shall be the exclusive property of the Company.",
        "Employee hereby assigns to Company all rights, title, and interest in any Intellectual Property.",
        "Company shall own all inventions, discoveries, and improvements made by Employee."
    ],
    "Dispute Resolution": [
        "Any dispute arising out of or in connection with this agreement shall be resolved by arbitration.",
        "The parties agree to submit any disputes to binding arbitration.",
        "All disputes shall be settled by arbitration in accordance with the rules of the American Arbitration Association."
    ],
    "Force Majeure": [
        "Neither party shall be liable for any failure to perform due to causes beyond their reasonable control.",
        "Force majeure events include acts of God, war, or natural disasters.",
        "Performance shall be excused during the period of force majeure."
    ],
    "Assignment": [
        "Neither party may assign this agreement without the prior written consent of the other party.",
        "This agreement may not be assigned by either party without written approval.",
        "No assignment of this agreement shall be valid unless in writing and signed by both parties."
    ],
    "Notices": [
        "All notices required under this agreement shall be in writing and delivered to the addresses specified.",
        "Notice shall be deemed given when delivered personally or sent by certified mail.",
        "Any notice under this agreement must be in writing."
    ],
    "Entire Agreement": [
        "This agreement constitutes the entire agreement between the parties.",
        "This document supersedes all prior agreements and understandings.",
        "No other agreements, promises, or representations shall be binding."
    ],
    "Amendment": [
        "This agreement may only be amended in writing signed by both parties.",
        "No amendment to this agreement shall be effective unless in writing.",
        "Any changes to this agreement must be made in writing."
    ],
    "Waiver": [
        "No waiver of any provision of this agreement shall be effective unless in writing.",
        "Failure to enforce any provision shall not constitute a waiver.",
        "A waiver of any breach shall not be deemed a waiver of any subsequent breach."
    ],
    "Other": [
        "This clause does not fit any specific category.",
        "Miscellaneous provisions apply.",
        "Any other terms and conditions not covered above."
    ]
}

# Pre-compute embeddings for anchors to save time
ANCHOR_EMBEDDINGS = {
    clause: model.encode(texts, convert_to_tensor=True)
    for clause, texts in CLAUSE_ANCHORS.items()
}

import re
from .summary_service import summarizer

def extract_section_number(text: str) -> str:
    """
    Extracts section references like 'Section 1.2', 'Article 3', '2.1', etc.
    """
    # Regex for common section patterns
    # Matches: Section 1, Section 1.2, Article 3, 4.1 Termination
    patterns = [
        r"(?:Section|Article|Clause)\s+(\d+(?:\.\d+)*)",  # Explicit labels
        r"^(\d+(?:\.\d+)+)",                             # Number start (e.g., 4.1)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Reconstruct the label if it was a number-only match, or return identifying string
            matched_val = match.group(1)
            # If we matched just a number at start, prepending "Section" looks nicer in UI
            if match.re.pattern.startswith("^"):
                return f"Section {matched_val}"
            return f"Section {matched_val}" # simplifying to just "Section X"
            
    # Fallback: look for just a number at the very start of the text if it looks like a section header
    # e.g. "4. Confidentiality"
    simple_num_match = re.search(r"^(\d+(\.\d+)*)\.?\s+[A-Z]", text)
    if simple_num_match:
         return f"Section {simple_num_match.group(1)}"
         
    return ""

def extract_clauses(text: str, threshold: float = 0.4):
    """
    Extracts clauses from text using semantic similarity to anchor sentences.
    Returns structured data with summaries and section numbers.
    """
    # 1. Split text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50] 
    
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]

    if not paragraphs:
        return []

    # 2. Encode all paragraphs at once
    try:
        paragraph_embeddings = model.encode(paragraphs, convert_to_tensor=True)
    except Exception as e:
        print(f"Error encoding paragraphs: {e}")
        return []

    results = []

    # Define risk levels for each clause type
    RISK_LEVELS = {
        "Confidentiality": "medium", # Lowercase to match UI style usually? User image showed lowercase "low", "high" tags
        "Termination": "low",       # User image showed "low" for 30 days notice
        "Indemnification": "high",
        "Governing Law": "low",
        "Payment Terms": "medium",
        "Liability": "high",
        "Non-Compete": "high",
        "Severance": "low",
        "IP Assignment": "medium",
        "At-Will Employment": "low" # Added from usage in screenshots if needed, mapping to termination anchors?
    }

    # 3. Compare each paragraph to each clause type
    matched_paragraphs = set()
    for clause_type, anchor_embs in ANCHOR_EMBEDDINGS.items():
        cosine_scores = util.cos_sim(paragraph_embeddings, anchor_embs)
        max_scores_per_paragraph, _ = torch.max(cosine_scores, dim=1)

        for idx, score in enumerate(max_scores_per_paragraph):
            if score.item() > threshold:
                original_text = paragraphs[idx]
                matched_paragraphs.add(idx)
                # Generate very short summary
                try:
                    summary_result = summarizer(original_text, max_length=60, min_length=10, do_sample=False, truncation=True)
                    summary_text = summary_result[0]['summary_text']
                except Exception:
                    summary_text = original_text[:100] + "..."

                section_ref = extract_section_number(original_text)

                results.append({
                    "title": clause_type,
                    "description": summary_text,
                    "risk": RISK_LEVELS.get(clause_type, "medium"),
                    "section": section_ref,
                    "confidence": round(score.item(), 3),
                    "original_text": original_text
                })

    # Add unmatched paragraphs as 'Other' clauses
    for idx, original_text in enumerate(paragraphs):
        if idx not in matched_paragraphs:
            try:
                summary_result = summarizer(original_text, max_length=60, min_length=10, do_sample=False, truncation=True)
                summary_text = summary_result[0]['summary_text']
            except Exception:
                summary_text = original_text[:100] + "..."
            section_ref = extract_section_number(original_text)
            results.append({
                "title": "Other",
                "description": summary_text,
                "risk": "low",
                "section": section_ref,
                "confidence": 0.4,
                "original_text": original_text
            })

    # Deduplicate: prefer higher confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    unique_results = []
    seen_titlesss_sections = set() # Heuristic: Don't show same section for same clause type multiple times
    
    for res in results:
        # Key to identify uniqueness: tuple of (title, section) is probably good. 
        # But if section is empty, relying on title? 
        # A single paragraph might match multiple types. We pick the highest confidence one (already sorted)
        # But we also don't want the same paragraph appearing as mulitple types.
        
        # We also need to avoid duplicates of the exact same text matching the same thing.
        # Let's use (title, section) as uniq key if section exists, otherwise just title + hash of text?
        # Actually, simpler: unique by text. A paragraph is one clause.
        
        # Wait, if a single paragraph covers both termination and liability, we might want to split it?
        # For now, let's just allow the highest confidence match for a given paragraph.
        pass # Logic handled below

    # Better deduplication strategy:
    # 1. Map each paragraph index to its best matching clause type.
    # But wait, `results` doesn't track paragraph index explicitly anymore.
    # Let's rely on text unique check.
    
    seen_texts = set()
    final_output = []
    
    for res in results:
        txt = res['original_text']
        if txt not in seen_texts:
            seen_texts.add(txt)
            # Cleanup keys for frontend
            frontend_obj = {
                "title": res['title'],
                "description": res['description'],
                "risk": res['risk'],
                "section": res['section']
            }
            final_output.append(frontend_obj)

    return final_output
