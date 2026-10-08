import re
import spacy

nlp = spacy.load("en_core_web_sm")

def mask_pii(text: str) -> dict:
    """
    Detects and masks PII/PCI entities in the text without using LLMs.
    Returns the masked email string and a structured list of masked entities with positions.
    """
    if not isinstance(text, str):
        return {"masked_email": "", "list_of_masked_entities": []}

    entities = []
    
    # Define regex patterns for structured PII/PCI fields
    patterns = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone_number": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10}\b',
        "dob": r'\b(?:0[1-9]|[12][0-9]|3[01])[-/.](?:0[1-9]|1[012])[-/.](?:19|20)\d{2}\b',
        "aadhar_num": r'\b\d{4}\s?\d{4}\s?\d{4}\b',
        "credit_debit_no": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "cvv_no": r'\b\d{3,4}\b',
        "expiry_no": r'\b(0[1-9]|1[0-2])\/([0-9]{2}|[0-9]{4})\b'
    }

    # 1. Apply Regex Masking
    for entity_type, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            start, end = match.span()
            entity_value = match.group()
            entities.append({
                "position": [start, end],
                "classification": entity_type,
                "entity": entity_value
            })

    # 2. Apply spaCy NER for Full Names
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            entities.append({
                "position": [start, end],
                "classification": "full_name",
                "entity": ent.text
            })

    # Sort entities by start position and filter out overlapping matches
    entities = sorted(entities, key=lambda x: x["position"][0])
    filtered_entities = []
    last_end = -1
    for e in entities:
        if e["position"][0] >= last_end:
            filtered_entities.append(e)
            last_end = e["position"][1]

    # Construct masked string (sorting descending for string slicing)
    masked_email = text
    desc_entities = sorted(filtered_entities, key=lambda x: x["position"][0], reverse=True)
    for e in desc_entities:
        start, end = e["position"]
        tag = f"[{e['classification']}]"
        masked_email = masked_email[:start] + tag + masked_email[end:]

    return {
        "masked_email": masked_email,
        "list_of_masked_entities": filtered_entities
    }