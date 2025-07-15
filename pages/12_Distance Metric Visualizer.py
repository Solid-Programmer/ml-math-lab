import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Distance_Metric_Visualizer.generator import generate_data, generate_smart_factory_data
from Distance_Metric_Visualizer.visualizer import show_metric_visualization

st.set_page_config(page_title="Distance Metric Visualizer", layout="centered")
st.title("Distance Metric Visualizer")

st.markdown("""
---

#### What is Distance?

In mathematics, distance is a measure of how far apart two points are in space.  
The most common example is the straight-line distance between two points.

#### What Does Distance Mean in Machine Learning?

In machine learning, distance helps us measure how similar or different two data points are.  
We usually compute the distance between a test point and known training points.  
Smaller distances indicate higher similarity. This is used to classify, cluster, or compare data.


#### Where Is Distance Used?

| Use Case                 | How Distance Is Used                                 |
|--------------------------|------------------------------------------------------|
| k-NN Classification      | To find nearest neighbors for prediction             |
| Clustering (e.g., K-Means) | To assign points to the closest cluster center     |
| Anomaly Detection        | To detect outliers far from normal data points       |
| Text Similarity / Search | To rank similar documents using vector distances     |
| Image Recognition        | To compare feature vectors of images                 |
| Recommendation Systems   | To suggest similar items or users                    |
| Regularization           | To control model complexity using L1 or L2 norms     |
| Embeddings               | To represent and compare data in vector space        |

---
""")

st.markdown("""
#### Types of Distance Metrics

**1. Euclidean Distance (L2)**  
Measures the shortest straight-line distance between two points.  
Example: Walking diagonally across a park from point A to B.

**2. Manhattan Distance (L1)**  
Measures distance along horizontal and vertical paths (like a city grid).  
Example: Walking along blocks in a city, turning at corners to reach B.

**3. Cosine Distance**  
Measures the angle between two vectors, ignoring their length.  
Example: Two reviews with similar opinions but different word counts are considered close.

**4. Minkowski Distance**  
A general form of distance that becomes L1 when p = 1 and L2 when p = 2.  
Example: You can tune the 'p' value to switch between block-style and straight-line behavior.
""")

# Controls Section
st.markdown("---")
st.subheader("1. Explore Distance Metrics")

col1, col2 = st.columns(2)

with col1:
    metric = st.selectbox("Distance Metric", ["Euclidean", "Manhattan", "Cosine", "Minkowski"])

with col2:
    p_value = st.slider("Minkowski p", 1.0, 4.0, 2.0, 0.1) if metric == "Minkowski" else None

st.markdown("#### Generate 2D Points")

n_points = st.slider("Number of points", 10, 500, 30)
X, y = generate_data(n_samples=n_points)

# Visualize distance geometry
show_metric_visualization(X, metric=metric, p=p_value)

st.markdown("---")
st.subheader("2. Understand Norm Geometry (Where Distance Metrics Come From)")

st.markdown("""
### What Is a Norm?

A **norm** is a mathematical function that measures the length (magnitude) of a vector.  
In 2D, it tells us how far a point is from the origin using a particular rule.

Many distance metrics are actually **based on norms**:

| Distance Metric       | Norm Used        |
|------------------------|------------------|
| Manhattan Distance     | L1 norm           |
| Euclidean Distance     | L2 norm           |
| Maximum Distance       | L∞ norm           |
| Minkowski Distance     | L<sub>p</sub> norm (generalized form) |

This part of the tool visualizes how different **norms shape space** — by plotting **unit balls** that represent all points with norm = 1.
""")

# --- Select norm type and p ---
norm_options = ["L1 (Manhattan)", "L2 (Euclidean)", "L∞ (Max Norm)", "Lp (Custom)"]
norm_choice = st.selectbox("Choose Norm", norm_options)

# Slider for p value (only visible for Lp)
p = None
if norm_choice == "Lp (Custom)":
    p = st.slider("Select p for Lp Norm", min_value=0.5, max_value=10.0, value=2.0, step=0.1)

# --- Generate unit circle/diamond/square/Lp ball ---
theta = np.linspace(0, 2 * np.pi, 400)
unit_points = []

for angle in theta:
    x = np.cos(angle)
    y = np.sin(angle)
    if norm_choice == "L1 (Manhattan)":
        scale = 1 / (abs(x) + abs(y))
    elif norm_choice == "L2 (Euclidean)":
        scale = 1 / np.sqrt(x ** 2 + y ** 2)
    elif norm_choice == "L∞ (Max Norm)":
        scale = 1 / max(abs(x), abs(y))
    elif norm_choice == "Lp (Custom)":
        scale = 1 / ((abs(x) ** p + abs(y) ** p) ** (1 / p))
    else:
        scale = 1
    unit_points.append((scale * x, scale * y))

unit_points = np.array(unit_points)

# --- Optional point inclusion check ---
st.markdown("### Optional: Check point inclusion")
x0 = st.number_input("Test Point X", value=0.5, step=0.1)
y0 = st.number_input("Test Point Y", value=0.5, step=0.1)
test_point = np.array([x0, y0])

if norm_choice == "L1 (Manhattan)":
    distance = np.linalg.norm(test_point, ord=1)
elif norm_choice == "L2 (Euclidean)":
    distance = np.linalg.norm(test_point, ord=2)
elif norm_choice == "L∞ (Max Norm)":
    distance = np.linalg.norm(test_point, ord=np.inf)
elif norm_choice == "Lp (Custom)":
    distance = np.linalg.norm(test_point, ord=p)

inside = distance <= 1

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(unit_points[:, 0], unit_points[:, 1], 'r--', label=f"Unit Ball ({norm_choice})")
ax.scatter(0, 0, color='black', label="Origin")
ax.scatter(x0, y0, color='blue', s=80, edgecolor='k', label=f"Test Point (inside? {'Yes' if inside else 'No'})")

ax.set_title(f"Norm Contour: {norm_choice}")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_aspect('equal', 'box')
ax.grid(True)
ax.legend()

st.pyplot(fig)

# --- Explanation panel ---
st.markdown("#### Norm Formulas")

st.latex(r"\|x\|_1 = |x_1| + |x_2|")
st.latex(r"\|x\|_2 = \sqrt{x_1^2 + x_2^2}")
st.latex(r"\|x\|_\infty = \max(|x_1|, |x_2|)")
st.latex(r"\|x\|_p = \left( |x_1|^p + |x_2|^p \right)^{1/p}")

st.markdown("---")
st.markdown("#### ML Insights")
st.markdown("""
- **L1 norm** promotes sparsity (used in LASSO).  
- **L2 norm** is smooth and isotropic (used in Ridge).  
- **L∞ norm** defines bounding box-like distance.  
- Changing norms changes model geometry and sensitivity.
""")

# Generate Smart Factory data with mild overlap
def generate_smart_factory_data():
    X, y = make_blobs(
        n_samples=300,
        centers=[[-2, -2], [2, 2], [-1, 3]],
        cluster_std=[1.8, 1.5, 1.5],
        random_state=42
    )
    return X, y

st.markdown("---")
st.subheader("3. How Distance Affects Classification")

st.markdown("""
This demo shows how the **decision boundary** in k-NN classification changes based on the **distance metric**.

### Key Insight
Changing the distance alters which neighbors are chosen, which may change the class prediction.
""")

# Load data
X, y = generate_smart_factory_data()

# User controls
col1, col2 = st.columns(2)
with col1:
    metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"], key="knn_metric")
with col2:
    k = st.slider("Number of Neighbors (k)", min_value=1, max_value=10, value=3)

# Train k-NN model
model = KNeighborsClassifier(n_neighbors=k, metric=metric)
model.fit(X, y)

# Create meshgrid for decision surface
h = 0.1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap="Accent")
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Accent", edgecolor="k")
ax.set_title(f"k-NN (k={k}) using {metric.title()} Distance")
ax.set_xlabel("Temperature")
ax.set_ylabel("Vibration")
ax.grid(True)

st.pyplot(fig)

# ------------------------------------------------------------
# Final Notes and Conclusion
# ------------------------------------------------------------
st.markdown("---")
# ------------------------------------------------------------
# Final Notes
# ------------------------------------------------------------
st.markdown("---")
st.subheader("Final Notes")

st.markdown("""
- Different distance metrics change how "closeness" is measured.
- This affects which neighbors k-NN selects — and thus the predicted class.
- **Euclidean**: smooth, circular boundaries (good for geometric closeness).
- **Manhattan**: boxy, axis-aligned boundaries (robust to individual outliers).
- **Cosine**: compares angle, not length (great for text or direction-based data).
- Changing the metric can shift a point from one class to another.
- No one-size-fits-all — pick based on your data and problem.
""")


st.markdown("""
##### Practical Tips

- Always **standardize or normalize** your data before applying distance-based models.
- Try multiple distance metrics during experimentation to observe changes in decision boundaries.
- Visualizations like these help develop **intuition** about how models "see" space.

Choosing the right distance metric isn't just a technical decision — it's a modeling decision that directly affects performance, interpretability, and generalization.
""")

st.markdown("---")
st.markdown("""
<div class="footer">
    For questions or feedback, contact the project maintainer at 
    <a href="mailto:12bce1006@gmail.com" style="color:#1976d2; text-decoration: none; font-weight: 500;">
        this email
    </a>, or visit 
    <a href="https://samkhai.com/" target="_blank" style="color:#1976d2; text-decoration: none; font-weight: 500;">
        my portfolio
    </a>.<br>
    ML Math Lab &copy; 2025
</div>
""", unsafe_allow_html=True)