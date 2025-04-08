import re

def preprocess_text(text: str) -> str:
    """
    Clean and normalize text data by lowercasing,
    stripping extra spaces, and removing unwanted characters.
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

if __name__ == "__main__":
    sample = "  This is A Sample TEXT, with   EXTRA spaces!  "
    print("Processed Text:", preprocess_text(sample))