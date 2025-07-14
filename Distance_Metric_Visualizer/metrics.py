import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def cosine_distance(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0  # Max dissimilarity if one is zero vector
    return 1 - np.dot(a, b) / (norm_a * norm_b)

def minkowski_distance(a, b, p=2):
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

def get_distance_fn(metric_name: str, p_value: float = 2.0):
    if metric_name == "Euclidean":
        return euclidean_distance
    elif metric_name == "Manhattan":
        return manhattan_distance
    elif metric_name == "Cosine":
        return cosine_distance
    elif metric_name == "Minkowski":
        return lambda a, b: minkowski_distance(a, b, p=p_value)
    else:
        raise ValueError(f"Unknown distance metric: {metric_name}")
