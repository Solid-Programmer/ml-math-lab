import json
import re
import pandas as pd
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import unicodedata
import os

# NLTK stopwords setup
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load JSON sections
json_path = os.path.join(os.path.dirname(__file__), "book_sections.json")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"book_sections.json not found at {json_path}")
with open(json_path, "r", encoding="utf-8") as f:
    sections = json.load(f)

# Build DataFrame
for item in sections:
    item["combined"] = item["title"] + " " + item["text"]
    item["cleaned"] = clean(item["combined"])

df = pd.DataFrame(sections)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=2,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=10000  # limit vocab size for large books
)
doc_vectors = vectorizer.fit_transform(df["cleaned"])

# Save everything into a pickle (absolute path for robustness)
pkl_path = os.path.join(os.path.dirname(__file__), "tfidf_vector_store.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "doc_vectors": doc_vectors,
        "df": df
    }, f)

print(f"✅ TF-IDF vector store saved as {pkl_path}")
