import numpy as np

def standardize_data(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_centered = X - X_mean
    X_standardized = X_centered / X_std
    return X_standardized, X_mean, X_std

def covariance_matrix(X_standardized):
    return np.cov(X_standardized, rowvar=False)

def pca_eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    return eigenvalues_sorted, eigenvectors_sorted

def explained_variance_ratio(eigenvalues_sorted):
    return eigenvalues_sorted / np.sum(eigenvalues_sorted)

def project_data(X_standardized, eigenvectors_sorted, k=2):
    return X_standardized @ eigenvectors_sorted[:, :k]
