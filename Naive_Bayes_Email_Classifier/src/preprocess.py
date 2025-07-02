import re
import string
from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# One-time NLTK downloads (run only once)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Initialize reusable components
stop_words = set(stopwords.words('english'))

# --- Modular Preprocessing Steps ---

def clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation, numbers, and normalize whitespace.
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """
    Tokenize the cleaned text into individual words.
    """
    return word_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove common English stopwords.
    """
    return [t for t in tokens if t not in stop_words]

def preprocess_text(text: str) -> List[str]:
    """
    Full pipeline: clean, tokenize, remove stopwords.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    return remove_stopwords(tokens)




