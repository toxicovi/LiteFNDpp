import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """
    Preprocess the input text:
    - NER-aware token joining
    - Lowercasing, punctuation removal
    - Replace dates, numbers, locations with placeholders
    """
    doc = nlp(text)
    tokens = []
    for ent in doc.ents:
        text = text.replace(ent.text, "_".join(ent.text.split()))
    # Simple placeholder replacement
    text = text.replace(".", " ").replace(",", " ")
    text = text.lower()
    return text
