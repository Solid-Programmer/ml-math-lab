import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd

def generate_2d_classification_data(n_samples=200, centers=2, random_state=42):
    """
    Generates a 2D binary classification dataset using Gaussian blobs.
    Returns features X and labels y.
    """
    X, y = make_blobs(n_samples=n_samples,
                      centers=centers,
                      n_features=2,
                      cluster_std=1.5,
                      random_state=random_state)

    # Return as DataFrame for easier plotting
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    return df

