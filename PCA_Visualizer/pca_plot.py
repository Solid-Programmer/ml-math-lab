import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def plot_2d_scatter(X, labels, feat_names, title, annotate_coords=True):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], color='royalblue', s=60)
    for i, label in enumerate(labels):
        if annotate_coords:
            coords = f"({X[i, 0]}, {X[i, 1]})"
            ax.text(X[i, 0] + 10, X[i, 1] + 0.1, f"{label} {coords}", fontsize=9)
        else:
            ax.text(X[i, 0] + 0.1, X[i, 1], label, fontsize=9)
    ax.set_xlabel(feat_names[0])
    ax.set_ylabel(feat_names[1])
    ax.set_title(title)
    ax.grid(False)
    st.pyplot(fig)

def plot_3d_scatter(X, labels, feat_names, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='royalblue', s=60)
    for i, label in enumerate(labels):
        coords = f"({X[i, 0]}, {X[i, 1]}, {X[i, 2]})"
        ax.text(X[i, 0] + 10, X[i, 1], X[i, 2], f"{label} {coords}", fontsize=9)
    ax.set_xlabel(feat_names[0])
    ax.set_ylabel(feat_names[1])
    ax.set_zlabel(feat_names[2])
    ax.set_title(title)
    st.pyplot(fig)

def plot_pca_scatter(U_k, labels, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(U_k[:, 0], U_k[:, 1], color='royalblue', s=80)
    for i, label in enumerate(labels):
        ax.text(U_k[i, 0] + 0.1, U_k[i, 1], label, fontsize=9)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title(title)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)

def plot_bar_loadings(feature_names, pca1_loadings, pca2_loadings):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.barh(feature_names, pca1_loadings, color='royalblue')
    ax1.set_title("PCA1 Feature Contributions")
    ax1.set_xlabel("Loading Weight")
    ax2.barh(feature_names, pca2_loadings, color='seagreen')
    ax2.set_title("PCA2 Feature Contributions")
    ax2.set_xlabel("Loading Weight")
    plt.tight_layout()
    st.pyplot(fig)

def plot_scree(explained_var):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(explained_var)+1), explained_var, marker='o')
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Scree Plot")
    st.pyplot(fig)
