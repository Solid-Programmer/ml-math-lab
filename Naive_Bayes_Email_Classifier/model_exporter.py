# model_exporter.py

import os
import pickle
from src.data_loader import load_dataframe, check_missing_or_invalid_labels
from src.main import (
    build_vocab, compute_priors, compute_word_probs, split_data, evaluate
)

# Load and preprocess data
data_path = os.path.abspath("data/enron_spam_data.csv")
df = load_dataframe(data_path)
check_missing_or_invalid_labels(df)
train_df, test_df = split_data(df)

# Train model components
spam_vocab, ham_vocab = build_vocab(train_df)
p_spam, p_ham, _, _ = compute_priors(train_df)
spam_word_probs, ham_word_probs, vocab_size, total_spam_words, total_ham_words = compute_word_probs(spam_vocab, ham_vocab)

# Evaluation on test set
test_df["label"] = test_df["Spam/Ham"].str.strip().str.lower()
acc, prec, rec, f1, cm = evaluate(test_df, p_spam, p_ham, spam_word_probs, ham_word_probs, total_spam_words, total_ham_words, vocab_size)

# Add top words from vocab
top_spam_words = sorted(spam_vocab.items(), key=lambda x: x[1], reverse=True)[:20]
top_ham_words = sorted(ham_vocab.items(), key=lambda x: x[1], reverse=True)[:20]

# Final model bundle
model = {
    "p_spam": p_spam,
    "p_ham": p_ham,
    "spam_word_probs": spam_word_probs,
    "ham_word_probs": ham_word_probs,
    "total_spam_words": total_spam_words,
    "total_ham_words": total_ham_words,
    "vocab_size": vocab_size,
    "top_spam_words": top_spam_words,
    "top_ham_words": top_ham_words,
    "metrics": {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist()
    }
}

# Save to disk
os.makedirs("model", exist_ok=True)
with open("model/naive_bayes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model exported to model/naive_bayes_model.pkl")
