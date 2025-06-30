import numpy as np
import matplotlib.pyplot as plt

def plot_probability_distribution(probabilities, labels):
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, probabilities, color="skyblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probability")
    ax.set_title("Probability Distribution")
    return fig

def plot_entropy_contributions(probabilities, labels):
    x = np.arange(len(labels))
    entropy_contributions = -probabilities * np.log2(probabilities + 1e-10)
    fig, ax = plt.subplots()
    ax.bar(x, entropy_contributions, color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Entropy Contribution (bits)")
    ax.set_title("Entropy by Category")
    return fig

def plot_cross_entropy_contributions(ce_contributions, labels):
    fig, ax = plt.subplots()
    ax.bar(labels, ce_contributions, color=['green', 'orange', 'gray'])
    ax.set_ylabel("CE Loss (bits)")
    ax.set_title("Cross-Entropy Loss Contribution per Class")
    return fig
