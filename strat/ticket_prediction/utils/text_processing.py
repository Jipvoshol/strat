import unicodedata
from typing import Any

def normalize_text(text: Any) -> str:
    """
    Normaliseert tekst door accenten en speciale tekens te verwijderen en te converteren naar kleine letters.

    Parameters:
        text (Any): Input tekst.

    Returns:
        str: Genormaliseerde tekst.
    """
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower().strip() 