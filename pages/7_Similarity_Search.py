import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from symspellpy.symspellpy import SymSpell, Verbosity

# ---------- Spell Correction Setup ----------
@st.cache_resource
def load_symspell(dict_path=None):
    if dict_path is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dict_path = os.path.join(BASE_DIR, "Cosine_Similarity", "frequency_dictionary_en_82_765.txt")
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    if os.path.exists(dict_path):
        sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        # Add custom domain-specific words
        sym_spell.create_dictionary_entry("supervised", 1000)
        sym_spell.create_dictionary_entry("unsupervised", 1000)
        sym_spell.create_dictionary_entry("machine learning", 1000)
    return sym_spell

def spell_correct_query(query, sym_spell):
    suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
    return suggestions[0].term if suggestions else query

# ---------- Load vector store ----------
@st.cache_resource
def load_vector_store(path=None):
    if path is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(BASE_DIR, "Cosine_Similarity", "tfidf_vector_store.pkl")
    with open(path, "rb") as f:
        store = pickle.load(f)
    return store["vectorizer"], store["doc_vectors"], store["df"]

vectorizer, doc_vectors, df = load_vector_store()
sym_spell = load_symspell()

# ---------- Page config ----------
st.set_page_config(page_title="Similarity Search with Cosine Distance", layout="centered",
                   initial_sidebar_state="expanded",)

# ---------- Title and explanation ----------
st.title("Similarity Search with Cosine Distance")
st.subheader("Using Cosine Similarity on PDF-based content")

st.markdown("""
Welcome! This app demonstrates how you can semantically search a PDF using **TF-IDF vectorization** and **cosine similarity**.

---

#### **Note for Learners:**
- This app uses **basic cosine similarity** on TF-IDF vectors. This is a classic and interpretable approach, but it has limitations:
    - It only matches on exact or similar words, not on meaning. For example, "car" and "automobile" are not considered similar.
    - It does not understand context, synonyms, or word order.
    - It is sensitive to spelling and phrasing (though we add spell correction for queries).
- **Modern semantic search** uses transformer-based models (like BERT, Sentence Transformers) that capture much deeper meaning and context. These models are much better for natural language queries, but require more resources and setup.

---

### What is a Vector?
In machine learning and NLP, a **vector** is a numeric representation of text. We convert each document or query into a vector so we can compare them mathematically.

For example, the sentence "machine learning is fun" might become a vector like `[0.1, 0.5, 0.2, ...]` based on word frequencies or other features.

### What is TF-IDF?
TF-IDF stands for **Term Frequency – Inverse Document Frequency**.

- **TF (Term Frequency):** Looks at how often a word appears in a single document (one section of the PDF). This is calculated **within a row**.
- **IDF (Inverse Document Frequency):** Looks at how rare that word is across all documents. This is calculated **across all rows**.

TF-IDF combines both, so you get high scores for words that are frequent in one section, but rare in others.

**In simple words:** TF tells you how important a word is in a single section, and IDF tells you how useful that word is for finding unique topics across the whole PDF.

### What is Cosine Similarity?
Cosine similarity measures how **similar** two vectors are by computing the cosine of the angle between them:

```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

- Score = 1 → texts are very similar (angle = 0°)
- Score = 0 → texts are unrelated (angle = 90°)

It works well for high-dimensional vector spaces like TF-IDF.

**How this app uses Cosine Similarity:**
- We convert each section and your query into a vector using TF-IDF.
- We compare your query vector to every section vector using cosine similarity.
- The sections with the highest scores are shown as the most relevant results.
- This helps you find the most similar or related content in the book, based on your question.

""")

# ---------- Cosine Similarity Demo ----------
st.markdown("""
---
### Cosine Similarity Demo
Type two short sentences below to see their vector representations and cosine similarity.
""")

from sklearn.feature_extraction.text import TfidfVectorizer
with st.form(key="similarity_form"):
    col1, col2 = st.columns(2)
    with col1:
        sentence1 = st.text_input("Sentence 1", "machine learning is fun")
    with col2:
        sentence2 = st.text_input("Sentence 2", "deep learning is interesting")
    submitted = st.form_submit_button("Calculate")

if submitted and sentence1.strip() and sentence2.strip():
    demo_vectorizer = TfidfVectorizer(norm='l2')  # Explicit normalization
    demo_vectors = demo_vectorizer.fit_transform([sentence1, sentence2])
    arr1 = demo_vectors[0].toarray().flatten()
    arr2 = demo_vectors[1].toarray().flatten()
    cos_sim = cosine_similarity([arr1], [arr2])[0, 0]

    st.markdown("**Vector for Sentence 1:**")
    st.code(arr1, language="plaintext")

    st.markdown("**Vector for Sentence 2:**")
    st.code(arr2, language="plaintext")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(arr1[:20], label="Sentence 1", marker='o')
    ax.plot(arr2[:20], label="Sentence 2", marker='x')
    ax.set_title("First 20 Dimensions of TF-IDF Vectors (Independent Demo)")
    ax.legend()
    st.pyplot(fig)
    st.success(f"Cosine Similarity: {cos_sim:.4f}")


# ---------- Search Box ----------
st.markdown("""
---
## Search the PDF

Type a **natural language question** below. We'll convert your question into a TF-IDF vector and compare it with every section in the PDF to show the most relevant results.

### About the Data Source
- The PDF used here is [Machine Learning Yearning by Andrew Ng](https://nessie.ilab.sztaki.hu/~kornai/2020/AdvancedMachineLearning/Ng_MachineLearningYearning.pdf)
- Try exploring the book by asking questions like:
  - `What is supervised learning?`
  - `Difference between training and testing?`
  - `What is the bias-variance tradeoff?`
---
""")
query = st.text_input("Enter your query:", "")

if query.strip():
    # Spell correction
    corrected_query = spell_correct_query(query, sym_spell)
    # Clean query (remove punctuation, lowercase)
    query_clean = re.sub(r"[^\w\s]", "", corrected_query.lower())
    query_vector = vectorizer.transform([query_clean])
    cosine_scores = cosine_similarity(query_vector, doc_vectors).flatten()

    top_n = 3
    top_indices = cosine_scores.argsort()[::-1][:top_n]

    st.markdown(f"### Top {top_n} Matching Sections")

    for idx in top_indices:
        score = cosine_scores[idx]
        st.markdown(f"#### 🔹 {df['title'][idx]}")
        st.markdown(f"**Relevance Score:** `{score * 100:.2f}%`")
        st.markdown(f"<div style='background-color:#f9f9f9;padding:10px;border-radius:5px'>{df['text'][idx][:500]}...</div>", unsafe_allow_html=True)
        st.markdown("---")
else:
    st.info("Type a query to see matching sections from the PDF.")

st.markdown("---")
st.markdown("""
<div class="footer">
    For questions or feedback, contact the project maintainer at 
    <a href="mailto:12bce1006@gmail.com" style="color:#1976d2; text-decoration: none; font-weight: 500;">
        this email
    </a>, or visit 
    <a href="https://samkhai.com/" target="_blank" style="color:#1976d2; text-decoration: none; font-weight: 500;">
        my portfolio
    </a>.<br>
    ML Math Lab &copy; 2025
</div>
""", unsafe_allow_html=True)