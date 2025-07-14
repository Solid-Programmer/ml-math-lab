from sklearn.datasets import make_blobs

def generate_data(n_samples: int = 300, random_state: int = 42):
    """
    Generate a clean 2D dataset with 3 clusters.
    Suitable for visualizing different distance metrics.

    Returns:
        X: 2D feature coordinates
        y: Corresponding class labels
    """
    X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.2, random_state=random_state)
    return X, y

def generate_smart_factory_data(n_samples: int = 300, random_state: int = 42):
    """
    Generate labeled 2D data for sensor events in a smart factory.
    X-axis: Temperature spike intensity
    Y-axis: Vibration level
    Labels:
        0 → Normal
        1 → Mechanical Fault
        2 → Overheating

    Returns:
        X: 2D numpy array of shape (n_samples, 2)
        y: Labels as integers (0, 1, 2)
        label_names: Dictionary mapping labels to class names
    """
    centers = [
        [1, 1],   # Normal
        [1, 5],   # Mechanical Fault
        [5, 1],   # Overheating
    ]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.8, random_state=random_state)

    label_names = {
        0: "Normal",
        1: "Mechanical Fault",
        2: "Overheating"
    }

    return X, y, label_names
