import streamlit as st
import numpy as np
from Entropy_Explorer.entropy_math import compute_entropy, compute_cross_entropy, compute_kl_divergence
from Entropy_Explorer.entropy_plot import plot_probability_distribution, plot_entropy_contributions, plot_cross_entropy_contributions
import matplotlib.pyplot as plt

st.set_page_config(page_title="Entropy Explorer", layout="centered")
st.title("Entropy and Cross-Entropy Explorer")

st.markdown("""
### What is Entropy?
Entropy is a measure of **uncertainty** or **randomness** in a distribution of outcomes.
It tells us how **unpredictable** or **surprising** the outcomes are.

**Key points:**
- Entropy is highest when all outcomes are equally likely.
- Entropy is lowest (zero) when one outcome is certain.
- Entropy is measured in **bits** (if log base 2 is used).
- Common in decision trees, compression, and information theory.
- Helps quantify the amount of information in a distribution.

It’s defined for a probability distribution as:
""")

st.latex(r"H(p) = -\sum_{i=1}^n p_i \log_2(p_i)")

st.markdown("---")
st.markdown("""
### 🐾 Example: Multi-Class Distribution – Animal Counts

To understand entropy in action, let’s look at a **multi-class distribution** — a scenario where outcomes belong to one of several categories.

**Imagine** you have a group of **100 animals**, where each one is either a **Cat 🐱**, **Dog 🐶**, or **Panda 🐼**.

Use the sliders below to assign how many animals belong to each class.  
We'll convert these counts into probabilities and compute the **entropy** of the resulting distribution — a measure of how uncertain or uniform your animal distribution is.
""")

# --- Controls ---
labels = ["Cat", "Dog", "Panda"]
num_classes = len(labels)

total_animals = 100

cat_count = st.slider("Cat count", 0, total_animals, 33, 1)

# Dynamically limit Dog slider only if valid
dog_max = total_animals - cat_count
if dog_max > 0:
    dog_count = st.slider("Dog count", 0, dog_max, min(dog_max, 33), 1)
else:
    dog_count = 0
    st.write("Dog count: 0 (auto-filled)")

panda_count = total_animals - cat_count - dog_count
st.write(f"Panda count: {panda_count} (auto-filled)")

animal_counts = [cat_count, dog_count, panda_count]


# Convert to probabilities
animal_counts = np.array(animal_counts)
total = animal_counts.sum()
probabilities = animal_counts / total if total > 0 else np.zeros_like(animal_counts)

# --- Compute Entropy ---
entropy = compute_entropy(probabilities)

# --- Show entropy computation ---
st.markdown("### Entropy Calculation")
for i, label in enumerate(labels):
    st.markdown(f"{label}: Count = {animal_counts[i]}, Probability = {probabilities[i]:.2f}, Entropy = {-probabilities[i] * np.log2(probabilities[i] + 1e-10):.4f} bits")

st.markdown(f"""
### Total Entropy = `{entropy:.4f}` bits
""")

# --- Charts ---
fig1 = plot_probability_distribution(probabilities, labels)
fig2 = plot_entropy_contributions(probabilities, labels)

col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1)
with col2:
    st.pyplot(fig2)

st.markdown("""
> **Entropy** quantifies the **uncertainty** in a distribution.  
> **Entropy** is **inversely related** to confidence in probability.  
> The more uniform the distribution, the **higher** the entropy.  
> **Low Entropy** means one class dominates — the model is very confident.  
> **High Entropy** means all classes are equally likely — the model is **less confident**.
""")




# ----------------------------
# Cross-Entropy Section
# ----------------------------

st.markdown("---")
st.markdown("## What is Cross-Entropy?")

st.markdown("""
Cross-Entropy measures the difference between two probability distributions:  
- **True distribution** (actual labels)  
- **Predicted distribution** (model’s output)

It’s widely used as a **loss function** for classification problems — especially in neural networks.

**Key points:**
- Lower Cross-Entropy means the predicted probabilities closely match the true labels.
- It penalizes wrong predictions more strongly when the model is confident.
- It becomes zero only when predicted distribution exactly matches the true distribution.

""")

st.latex(r"H(p, q) = -\sum_{i=1}^n p_i \log_2(q_i)")

st.markdown("**Where:**")
st.latex(r"p_i = \text{true probability of class } i \ (\text{actual distribution})")
st.latex(r"q_i = \text{predicted probability of class } i \ (\text{model output})")
st.markdown("---")



st.markdown("""
### Example: Cross-Entropy Loss for Animal Classification

Let’s suppose:
- The **true distribution** is fixed (based on actual labels).
- You can adjust the **predicted probabilities** using sliders.

We’ll compute the Cross-Entropy loss to show how far your predicted values are from truth.
""")

# --- Fixed true distribution ---
true_dist = np.array([0.7, 0.15, 0.15])  # Cat, Dog, Panda

st.markdown("### True Distribution")

st.markdown("""
In real-world classification, we usually represent the true label using **one-hot encoding** — assigning 100% probability to the correct class and 0% to others.

But here, we intentionally use a **soft true distribution**:

- Cat: 70%  
- Dog: 15%  
- Panda: 15%

This helps us visualize **how each class contributes** to the overall **Cross-Entropy loss**, even when the true label isn’t fully confident or when there's class ambiguity.
""")

# --- Predicted Probabilities ---
st.markdown("**Adjust Predicted Probabilities** (must sum to 1):")
pred_cat = st.slider("Predicted Probability for Cat", 0.0, 1.0, 0.6, 0.01)
pred_dog = st.slider("Predicted Probability for Dog", 0.0, 1.0 - pred_cat, 0.25, 0.01)
pred_panda = 1.0 - pred_cat - pred_dog
pred_dist = np.array([pred_cat, pred_dog, pred_panda])

st.markdown(f"Predicted Probability for Panda: `{pred_panda:.2f}` (auto-calculated)")

# --- Cross-Entropy Calculation ---
ce_loss = compute_cross_entropy(true_dist, pred_dist)

st.markdown("### Cross-Entropy Calculation")
for i, label in enumerate(labels):
    ce_term = -true_dist[i] * np.log2(pred_dist[i] + 1e-10)
    st.markdown(f"{label}: true = {true_dist[i]:.2f}, predicted = {pred_dist[i]:.2f}, contribution = {ce_term:.4f} bits")

st.markdown(f"### Total Cross-Entropy Loss = `{ce_loss:.4f}` bits")

# --- Visualize Cross-Entropy loss contributions ---
ce_contributions = -true_dist * np.log2(pred_dist + 1e-10)
fig_loss = plot_cross_entropy_contributions(ce_contributions, labels)
st.pyplot(fig_loss)



# Footer
st.markdown("""
> Cross-Entropy is **low** when the model assigns a high probability to the correct class — meaning it is confident and right.  
> It becomes **high** when the model is confident but assigns that confidence to the wrong class — a highly penalized mistake.  
> This makes Cross-Entropy a powerful and sensitive loss function for training classification models, especially when calibrated probabilities are important.
""")

st.markdown("---")
st.markdown("## KL Divergence")

# Reuse true_dist = np.array([1.0, 0.0, 0.0]) from earlier
# Compute entropy of true_dist only once
true_entropy = compute_entropy(true_dist)

st.markdown("""
KL Divergence (Kullback–Leibler divergence) measures how much one probability distribution **diverges** from a second, expected probability distribution.

It compares the **true distribution** with the **predicted distribution**, quantifying the extra information (in bits) needed when using the predicted distribution to approximate the true one.

The formula is:
""")

st.latex(r"D_{\mathrm{KL}}(P || Q) = \sum_{i=1}^n p_i \log_2\left(\frac{p_i}{q_i}\right)")

st.markdown(r"""
Where:
- \(P = [p_1, p_2, ..., p_n]\) is the **true** distribution  
- \(Q = [q_1, q_2, ..., q_n]\) is the **predicted** distribution
""")


# --- Compute KL Divergence ---
kl_divergence = compute_kl_divergence(true_dist, pred_dist)

st.markdown(f"### KL Divergence = `{kl_divergence:.4f}` bits")

st.markdown("""
> 💡 In this case, KL Divergence is simply the difference between Cross-Entropy and Entropy:
""")

st.latex(r"D_{\mathrm{KL}}(P || Q) = H(P, Q) - H(P)")

# Show the same numerically
st.markdown(f"""
- Cross-Entropy: `{ce_loss:.4f}`  
- Entropy: `{true_entropy:.4f}`  
- KL Divergence = `{ce_loss - true_entropy:.4f}` bits
""")

st.markdown("---")
st.markdown("<small>🔢 This demo is part of the ML Math Lab - exploring foundational concepts interactively.</small>", unsafe_allow_html=True)
