# naive_bayes_educational_app.py

import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Naive_Bayes_Email_Classifier')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.main import predict, load_model_components, get_word_contributions, get_top_words, evaluate_model_metrics

# --- Title and Goal ---
st.set_page_config(page_title="Naive Bayes Email Classifier", layout="centered")
st.title("Naive Bayes Email Classifier")
st.markdown("<h5>Learn how we use probabilities to classify emails as spam or ham</h5>", unsafe_allow_html=True)

# --- What is an Email Classifier? ---
st.header("What is an Email Classifier?")
st.markdown("""
An email classifier is a machine learning model that categorizes emails into two classes:
- **Spam**: Unwanted or junk email
- **Ham**: Legitimate, non-spam email

This helps users automatically filter out junk mail and prioritize important communication.
""")

# --- What is Naive Bayes? ---
st.header("What is Naive Bayes?")
st.markdown(r'''
Naive Bayes is a **probabilistic classification algorithm** based on **Bayes’ Theorem**, which assumes that features (words in a message) are conditionally independent given the class.

The core formula is:

$$
P(\text{Spam} \mid \text{Words}) \propto P(\text{Words} \mid \text{Spam}) \cdot P(\text{Spam})
$$

It's widely used for spam filtering due to its:
- Simplicity
- Speed
- Surprisingly strong performance
''')

# --- Limitations ---
st.markdown("""
**Limitations of Naive Bayes:**
- Assumes all words are independent (which is rarely true in natural language)
- Struggles with context, sarcasm, and advanced semantics
""")

# --- Step-by-Step Approach ---
st.header("Step-by-Step Implementation")

## 1. Data Collection
st.markdown("<h5>1. Data Collection</h5>", unsafe_allow_html=True)
st.markdown("""
- We use a labeled dataset of emails, each tagged as **spam** or **ham**.
- The dataset used here is the Enron spam corpus with thousands of labeled messages.
""")

## 2. Data Splitting
st.markdown("<h5>2. Data Splitting</h5>", unsafe_allow_html=True)
st.markdown("""
- A **stratified sampling technique** is used to split the dataset into **80% training** and **20% testing** sets.
- This ensures the spam-to-ham ratio is preserved in both subsets.
- Only the **training set** is used for vocabulary building and probability estimation.
- The **test set** remains unseen during training and is used strictly for evaluating the model's performance.
""")

## 3. Data Preprocessing
st.markdown("<h5>3. Data Preprocessing</h5>", unsafe_allow_html=True)
st.markdown("""
- Preprocessing is performed **separately** on training and test data to avoid data leakage.
- Each email’s **subject and body are combined**, converted to lowercase, punctuation is removed, and the text is tokenized.
- **Stop words** (like *the*, *is*, *and*) are removed to retain only the most informative words.
- The resulting tokens are used to build class-wise word frequencies and train the Naive Bayes model on the **training set only**.
""")


## 3. Model Training
st.markdown("<h5>4. Model Training</h5>", unsafe_allow_html=True)
st.markdown(r'''
To understand how the model is trained, let’s walk through a simple example with just 5 emails:

**Sample Input**:
- spam1: "Win money now"
- spam2: "Limited offer, win big prizes"
- spam3: "Claim your free cash now"
- ham1: "Meeting schedule attached"
- ham2: "Project deadline extended"

**Step 1: Preprocessing Output**
- spam1: [win, money, now]
- spam2: [limited, offer, win, big, prizes]
- spam3: [claim, free, cash, now]
- ham1: [meeting, schedule, attached]
- ham2: [project, deadline, extended]

**Step 2: Prior Calculation**
- \( P(\text{spam}) = \frac{3}{5} = 0.6 \)
- \( P(\text{ham}) = \frac{2}{5} = 0.4 \)

**Step 3: Word Likelihood Calculation (with Laplace Smoothing)**
- Vocabulary size (V) = 14 unique words
- Total words in spam emails = 11
- Total words in ham emails = 6

**Why Laplace Smoothing?**
To prevent zero probabilities for unseen words. Without smoothing, a single zero likelihood can nullify the entire probability product.

**Example**:
- For a word like "win" that appears 2 times in spam:

$$
P(\text{win} \mid \text{spam}) = \frac{2 + 1}{11 + 14} = \frac{3}{25} = 0.12
$$

- For a word like "win" that appears 0 times in ham:

$$
P(\text{win} \mid \text{ham}) = \frac{0 + 1}{6 + 14} = \frac{1}{20} = 0.05
$$
''')

## 4. Model Testing and Prediction
st.markdown("<h5>5. Model Testing</h5>", unsafe_allow_html=True)
st.markdown("""
- Use the trained model to classify new messages.
- Calculate log-probability of a message being spam or ham.
- Assign the label with the higher probability.
""")

# --- Interactive Demo ---
st.header("Try It Yourself")
components = load_model_components()
user_input = st.text_area("Enter your email message here:", "Free money now! Click here to claim.")

if st.button("Classify Email"):
    log_spam, log_ham = predict(user_input, *components, return_logprobs=True)
    label = "Spam" if log_spam > log_ham else "Ham"
    st.success(f"Prediction: **{label}**")

    # Explanation of model output
    st.subheader("Prediction Probabilities")
    st.write(f"Log( P(Spam | Message) ) = {log_spam:.4f}")
    st.write(f"Log( P(Ham | Message) ) = {log_ham:.4f}")

    # Show word contributions
    st.subheader("Word Contributions to Prediction")
    contributions = get_word_contributions(user_input, *components)
    if contributions:
        contrib_df = pd.DataFrame(contributions, columns=["Word", "Log(P(w|Spam))", "Log(P(w|Ham))"])
        st.dataframe(contrib_df)
    else:
        st.info("No informative words found after preprocessing.")

# --- Visualization of Top Words ---
st.header("Most Influential Words")

col1, col2 = st.columns(2)
with col1:
    top_spam = get_top_words("spam")
    spam_df = pd.DataFrame(top_spam, columns=["Word", "Score"])
    spam_df["Score"] = pd.to_numeric(spam_df["Score"], errors="coerce")
    spam_df = spam_df.dropna()
    st.bar_chart(spam_df.set_index("Word"))

with col2:
    top_ham = get_top_words("ham")
    ham_df = pd.DataFrame(top_ham, columns=["Word", "Score"])
    ham_df["Score"] = pd.to_numeric(ham_df["Score"], errors="coerce")
    ham_df = ham_df.dropna()
    st.bar_chart(ham_df.set_index("Word"))

# --- Model Evaluation ---
st.header("Model Evaluation Metrics")
metrics = evaluate_model_metrics()

# Display metric cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
col2.metric("Precision", f"{metrics['precision']:.2%}")
col3.metric("Recall", f"{metrics['recall']:.2%}")
col4.metric("F1 Score", f"{metrics['f1']:.2%}")

# Display confusion matrix
st.subheader("Confusion Matrix")
cm = metrics["confusion_matrix"]
cm_df = pd.DataFrame(cm, index=["Actual Spam", "Actual Ham"], columns=["Predicted Spam", "Predicted Ham"])
st.dataframe(cm_df.style.format(precision=0).set_caption("Confusion Matrix").highlight_max(axis=1, color="lightgreen"))

# --- Handling Edge Cases ---
st.header("Handling Edge Cases")
st.markdown("""
- **Unseen Words**: Handled using Laplace smoothing to avoid zero probabilities.  
- **Empty Messages**: Falls back to prior probabilities when no tokens remain.  
- **Stopword-only Messages**: Treated as empty after stopword removal, handled like empty input.  
- **Very Short/Long Emails**: Log-probabilities keep classification stable regardless of length.  
""")


# --- Conclusion ---
st.header("Conclusion")
st.markdown("""
Naive Bayes remains a fundamental algorithm for text classification — simple, fast, and surprisingly effective for tasks like spam detection.

However, real-world spam filters today use far more advanced techniques — including **deep learning models**, **transformers**, and **neural networks** — which capture complex language patterns, context, and user behavior.
""")

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
