import numpy as np

def compute_entropy(p):
    p = np.array(p)
    return -np.sum(p * np.log2(p + 1e-10))  # Add epsilon to avoid log(0)

def compute_cross_entropy(p_true, p_pred):
    p_pred = np.clip(p_pred, 1e-10, 1.0)
    return -np.sum(p_true * np.log2(p_pred))

def compute_kl_divergence(p_true, p_pred):
    p_true = np.array(p_true)
    p_pred = np.clip(p_pred, 1e-10, 1.0)
    return np.sum(p_true * np.log2((p_true + 1e-10) / p_pred))
