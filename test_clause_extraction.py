from services.clause_service import extract_clauses
import json

sample_text = """
SERVICE AGREEMENT

This Service Agreement is entered into on January 1st, 2026.

1. Confidentiality
The parties agree to keep all information confidential and not disclose it to third parties. Recipient shall not disclose the Disclosing Party’s Confidential Information.

2. Termination
This agreement may be terminated by either party with 30 days written notice.

3. Payment
Client shall pay Provider the fees set forth in the attached Statement of Work within 30 days.

4. Governing Law
This Agreement shall be governed by and construed in accordance with the laws of India.
"""

print("Extracting clauses...")
results = extract_clauses(sample_text)
print(json.dumps(results, indent=2))
