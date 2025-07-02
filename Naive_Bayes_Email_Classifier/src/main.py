# src/naive_bayes_classifier.py

import os
import math
from collections import Counter
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.data_loader import load_dataframe, check_missing_or_invalid_labels, count_spam_ham
from src.preprocess import preprocess_text
import streamlit as st
from src.model_loader import load_model

# ------------------------- Vocabulary and Probability -------------------------
def build_vocab(df):
    spam_vocab = Counter()
    ham_vocab = Counter()

    for _, row in df.iterrows():
        text = f"{row['Subject']} {row['Message']}"
        tokens = preprocess_text(text)
        label = row['Spam/Ham'].strip().lower()
        if label == 'spam':
            spam_vocab.update(tokens)
        elif label == 'ham':
            ham_vocab.update(tokens)

    return spam_vocab, ham_vocab

def compute_priors(df):
    df['label'] = df['Spam/Ham'].str.strip().str.lower()
    num_spam = (df['label'] == 'spam').sum()
    num_ham = (df['label'] == 'ham').sum()
    total = len(df)
    return num_spam / total, num_ham / total, num_spam, num_ham

def compute_word_probs(spam_vocab, ham_vocab):
    total_spam_words = sum(spam_vocab.values())
    total_ham_words = sum(ham_vocab.values())
    vocab = set(spam_vocab.keys()) | set(ham_vocab.keys())
    vocab_size = len(vocab)

    spam_word_probs = {}
    ham_word_probs = {}

    for word in vocab:
        spam_count = spam_vocab.get(word, 0)
        ham_count = ham_vocab.get(word, 0)
        spam_word_probs[word] = (spam_count + 1) / (total_spam_words + vocab_size)
        ham_word_probs[word]  = (ham_count + 1) / (total_ham_words + vocab_size)

    return spam_word_probs, ham_word_probs, vocab_size, total_spam_words, total_ham_words

# ------------------------- Prediction and Evaluation -------------------------
def predict(message, p_spam, p_ham, spam_word_probs, ham_word_probs, total_spam_words, total_ham_words, vocab_size, return_logprobs=False):
    tokens = preprocess_text(message)
    log_spam_prob = math.log(p_spam)
    log_ham_prob = math.log(p_ham)

    for token in tokens:
        pw_spam = spam_word_probs.get(token, 1 / (total_spam_words + vocab_size))
        pw_ham = ham_word_probs.get(token, 1 / (total_ham_words + vocab_size))
        log_spam_prob += math.log(pw_spam)
        log_ham_prob += math.log(pw_ham)

    if return_logprobs:
        return log_spam_prob, log_ham_prob
    return 'spam' if log_spam_prob > log_ham_prob else 'ham'

def split_data(df, test_size=0.2, random_state=42):
    df['label'] = df['Spam/Ham'].str.strip().str.lower()
    return train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )

def evaluate(test_df, p_spam, p_ham, spam_word_probs, ham_word_probs, total_spam_words, total_ham_words, vocab_size):
    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        text = f"{row['Subject']} {row['Message']}"
        label = row['label']
        prediction = predict(text, p_spam, p_ham, spam_word_probs, ham_word_probs, total_spam_words, total_ham_words, vocab_size)
        y_true.append(label)
        y_pred.append(prediction)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='spam')
    rec = recall_score(y_true, y_pred, pos_label='spam')
    f1 = f1_score(y_true, y_pred, pos_label='spam')
    cm = confusion_matrix(y_true, y_pred, labels=['spam', 'ham'])

    return acc, prec, rec, f1, cm

# ------------------------- Model Loader Interface -------------------------
def load_model_components():
    model = load_model()
    return (
        model["p_spam"],
        model["p_ham"],
        model["spam_word_probs"],
        model["ham_word_probs"],
        model["total_spam_words"],
        model["total_ham_words"],
        model["vocab_size"]
    )

# ------------------------- Explainability -------------------------
def get_word_contributions(message, p_spam, p_ham, spam_word_probs, ham_word_probs, total_spam_words, total_ham_words, vocab_size):
    tokens = preprocess_text(message)
    contributions = []
    for token in tokens:
        pw_spam = spam_word_probs.get(token, 1 / (total_spam_words + vocab_size))
        pw_ham = ham_word_probs.get(token, 1 / (total_ham_words + vocab_size))
        contributions.append((token, round(math.log(pw_spam), 4), round(math.log(pw_ham), 4)))
    return contributions

# ------------------------- Top Words from Model -------------------------
def get_top_words(label, n=10):
    model = load_model()
    key = "top_spam_words" if label == "spam" else "top_ham_words"
    return model.get(key, [])[:n]

# ------------------------- Evaluation Metrics -------------------------
def evaluate_model_metrics():
    model = load_model()
    metrics = model.get("metrics", {})
    # Convert confusion_matrix to np.array for compatibility
    if "confusion_matrix" in metrics:
        import numpy as np
        metrics["confusion_matrix"] = np.array(metrics["confusion_matrix"])
    return metrics
