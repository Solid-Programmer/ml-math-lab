import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def show_metric_visualization(X, metric, p=None):
    # Fixed reference point
    ref_point = np.array([1.0, 1.0])  # or [1, 0] for horizontal reference

    # Select logic per metric
    if metric == "Manhattan":
        ord_val = 1
        contour_shape = "diamond"
        r = st.slider("Select Distance Radius (r)", min_value=1, max_value=20, value=5, step=1)
    elif metric == "Euclidean":
        ord_val = 2
        contour_shape = "circle"
        r = st.slider("Select Distance Radius (r)", min_value=1, max_value=20, value=5, step=1)
    elif metric == "Cosine":
        contour_shape = "angle"
        angle_deg = st.slider("Select Cosine Sector Angle (degrees)", 5, 180, 45, step=1)
        angle_rad = np.deg2rad(angle_deg / 2)  # half-angle for symmetric sector
        r = 10  # fixed visual radius for ray lines
    else:
        st.error("Only 'Manhattan', 'Euclidean', and 'Cosine' are supported.")
        return

    # Compute distances
    if metric in ["Manhattan", "Euclidean"]:
        distances = np.array([np.linalg.norm(x - ref_point, ord=ord_val) for x in X])
    elif metric == "Cosine":
        def cosine_distance(x):
            x_norm = np.linalg.norm(x)
            if x_norm == 0:
                return 0
            return 1 - np.dot(x, ref_point) / (x_norm * np.linalg.norm(ref_point) + 1e-10)
        distances = np.array([cosine_distance(x) for x in X])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter points
    sc = ax.scatter(X[:, 0], X[:, 1], c=distances, cmap='viridis', s=60, edgecolors='k', alpha=0.8)

    # Reference point
    ax.scatter(*ref_point, color='red', s=100, edgecolor='black', label="Reference Point (1, 1)")
    ax.annotate("Reference Point\n(0, 0)", xy=ref_point, xytext=(0.5, -0.5),
                textcoords='offset points', ha='left', fontsize=9, color='red')

    # Draw contour or angular sector
    if contour_shape == "diamond":
        shape = np.array([[ r, 0], [0,  r], [-r, 0], [0, -r], [ r, 0]])
        ax.plot(shape[:, 0], shape[:, 1], 'r--', label=f'Equal Distance Contour (L1 = {r})')

    elif contour_shape == "circle":
        theta = np.linspace(0, 2 * np.pi, 200)
        circle = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        ax.plot(circle[:, 0], circle[:, 1], 'r--', label=f'Equal Distance Contour (L2 = {r})')

    elif contour_shape == "angle":
        # Draw angle sector using selected angle
        angle1 = angle_rad
        angle2 = -angle_rad
        x1, y1 = r * np.cos(angle1), r * np.sin(angle1)
        x2, y2 = r * np.cos(angle2), r * np.sin(angle2)
        ax.plot([0, x1], [0, y1], 'r--', label=f'Angle ±{angle_deg/2:.0f}°')
        ax.plot([0, x2], [0, y2], 'r--')
        ax.fill([0, x1, x2], [0, y1, y2], color='red', alpha=0.1, label='Cosine Angle Sector')

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f'{metric} Distance from Reference Point', rotation=270, labelpad=15)

    # Labels and grid
    ax.set_title(f"{metric} Distance Visualization", fontsize=14)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', 'box')

    st.pyplot(fig)
