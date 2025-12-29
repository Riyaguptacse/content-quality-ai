import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - lowercase
    - normalize whitespace
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
