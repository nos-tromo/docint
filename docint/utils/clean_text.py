import re


def basic_clean(text: str) -> str:
    """
    Cleans the input text by normalizing whitespace and line breaks.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # normalize newlines
    text = re.sub(r"\s+\n", "\n", text)  # remove spaces before newlines
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse multiple newlines
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse multiple spaces/tabs
    return text.strip()
