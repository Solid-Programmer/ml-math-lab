import numpy as np

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def predict(X, w1, w2, b):
    """
    Computes predicted probabilities using logistic regression.
    
    Parameters:
    - X: DataFrame with 'x1' and 'x2' columns
    - w1, w2: weights
    - b: bias

    Returns:
    - Predicted probabilities (numpy array)
    """
    z = w1 * X["x1"].values + w2 * X["x2"].values + b
    return sigmoid(z)

def compute_loss(y_true, y_pred):
    """
    Computes binary cross-entropy loss.
    
    Parameters:
    - y_true: Ground truth labels (0 or 1)
    - y_pred: Predicted probabilities

    Returns:
    - Total loss (scalar)
    """
    eps = 1e-9  # for numerical stability
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
