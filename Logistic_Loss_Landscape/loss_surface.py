import numpy as np
from .logistic_model import predict, compute_loss

def compute_loss_surface(df, w1_range, w2_range, b=0.0, resolution=50):
    """
    Computes the logistic loss over a meshgrid of (w1, w2) values.

    Parameters:
    - df: DataFrame with 'x1', 'x2', and 'label'
    - w1_range: tuple (min, max) for weight 1
    - w2_range: tuple (min, max) for weight 2
    - b: fixed bias term
    - resolution: number of points along each axis

    Returns:
    - W1, W2: meshgrid arrays
    - Z: loss values at each (w1, w2) point
    """
    w1_vals = np.linspace(w1_range[0], w1_range[1], resolution)
    w2_vals = np.linspace(w2_range[0], w2_range[1], resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = np.zeros_like(W1)

    y_true = df["label"].values

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w1 = W1[i, j]
            w2 = W2[i, j]
            y_pred = predict(df, w1, w2, b)
            Z[i, j] = compute_loss(y_true, y_pred)
    
    i_min, j_min = np.unravel_index(np.argmin(Z), Z.shape)
    min_point = (W1[i_min, j_min], W2[i_min, j_min], Z[i_min, j_min])

    return W1, W2, Z, min_point
