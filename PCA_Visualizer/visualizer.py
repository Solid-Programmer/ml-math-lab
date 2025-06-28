import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pca_math import standardize_data, covariance_matrix, pca_eigen_decomposition, explained_variance_ratio, project_data
from pca_plot import plot_2d_scatter, plot_3d_scatter, plot_pca_scatter, plot_bar_loadings, plot_scree

st.set_page_config(page_title="PCA Visualizer", layout="centered")

st.title("PCA Visualizer")
st.subheader("Understand Dimensionality Reduction with Eigen Decomposition or SVD")

# --- PCA Introduction ---
st.markdown("---")

st.markdown("""
<div style='font-size:18px; text-align: justify;'>
<b>What is PCA?</b><br><br>
Principal Component Analysis (PCA) is a powerful linear transformation technique used to reduce the number of features (dimensions) in a dataset while retaining the most significant patterns.
<br><br>
It transforms the original correlated variables into a new set of uncorrelated axes called <b>principal components</b>.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='font-size:18px; text-align: justify;'>
<b>What does PCA do?</b><br><br>
PCA simplifies complex, high-dimensional data into lower dimensions, allowing us to visualize hidden relationships, trade-offs, or clusters between features that are otherwise hard to interpret.
<br><br>
The top principal components capture the directions of <b>maximum variance</b>, helping reduce noise and redundancy in the data while preserving the structure.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='font-size:18px; text-align: justify;'>
<b>How does PCA work?</b><br><br>
PCA operates on a matrix with <b>k observations</b> (rows) and <b>n features</b> (columns). It focuses on the relationships between features — the number of observations doesn’t matter much (in fact, more is better).
<br><br>
It helps us identify the most important <i>directions</i> in feature space that explain the variability in the data — enabling us to project the data to 2D or 3D for visualization.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='font-size:18px;'>
<b>Common Techniques to Perform PCA:</b><br><br>
<ol style='margin-left: 20px; font-size: 17px;'>
  <li><b>Eigen Decomposition</b> of the Covariance Matrix</li>
  <li><b>Singular Value Decomposition (SVD)</b> of the Standardized Data Matrix</li>
</ol>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='font-size:18px; text-align: justify;'>
PCA is widely used in data science and machine learning pipelines for:
<ul style='margin-left: 20px; font-size: 17px;'>
  <li>Feature reduction</li>
  <li>Noise filtering</li>
  <li>Data visualization (2D/3D)</li>
  <li>Preprocessing before clustering or classification</li>
</ul>
It is especially useful when working with high-dimensional tabular, image, or biological data.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Transition to Problem Understanding ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("### Let's First Understand: What Problem is PCA Trying to Solve?")
st.markdown("""
Before we jump into PCA, it's important to see that when we only have **2 or 3 features**, we can already plot and understand our data easily.

But as dimensions increase, it becomes hard to:
- Visualize relationships between features
- Spot patterns or clusters
- Identify trade-offs across many specs

That’s when PCA becomes helpful — to reduce dimensions **while preserving structure**.

Let’s start with simple 2D and 3D plots to build intuition.
""")
st.markdown("---")



# --- 2D Plot Section ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.header("Visualizing Data with Just 2 Features")
st.markdown("When data has only two features, it's easy to visualize patterns without PCA.")

# 2D Data: Storage vs Battery
X_2d = np.array([
    [512, 10],   # ASUS
    [1024, 11],  # Dell
    [512, 9],    # Lenovo
    [256, 8]     # HP
])
obs_2d = ["ASUS", "Dell", "Lenovo", "HP"]
feat_2d = ["Storage (GB)", "Battery (hrs)"]

# Table header and rows
headers = ["Brand", "Storage (GB)", "Battery (hrs)"]
rows = [
    ["ASUS", 512, 10],
    ["Dell", 1024, 11],
    ["Lenovo", 512, 9],
    ["HP", 256, 8]
]

# Combine into HTML table for better control (no index)
table_html = "<table><thead><tr>{}</tr></thead><tbody>{}</tbody></table>".format(
    "".join([f"<th style='padding: 6px 12px; text-align: left;'>{h}</th>" for h in headers]),
    "".join([
        "<tr>{}</tr>".format("".join(
            [f"<td style='padding: 6px 12px;'>{cell}</td>" for cell in row]
        )) for row in rows
    ])
)

# Title and display
st.markdown("#### Laptop Specs Sample (Only 2 Features)")
st.markdown(table_html, unsafe_allow_html=True)

# 2D Scatter Plot
plot_2d_scatter(X_2d, obs_2d, feat_2d, "2D Plot: Storage vs Battery (No PCA Needed)")

# Insights
st.markdown("### What We Observe from the 2D Plot:")
st.markdown("""
- Dell stands out with both highest storage and battery — clearly a premium laptop.
- HP offers the lowest on both — likely a budget-friendly model.
- ASUS and Lenovo share the same storage, but ASUS gives better battery life.
- When only two features are involved, we don’t need PCA — the relationship is visually clear.
""")


# --- 3D Plot Section ---
# --- 3D Plot Section ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.header("Visualizing Data with 3 Features")
st.markdown("When we move to 3 features, visualizing becomes harder — but still doable!")

# 3D Data: Storage, Battery, Price
X_3d = np.array([
    [512, 10, 85],   # ASUS
    [1024, 11, 110], # Dell
    [512, 9, 80],    # Lenovo
    [256, 8, 60]     # HP
])
obs_3d = ["ASUS", "Dell", "Lenovo", "HP"]
feat_3d = ["Storage (GB)", "Battery (hrs)", "Price (k INR)"]

# Table header and rows
headers_3d = ["Brand"] + feat_3d
rows_3d = [
    ["ASUS", 512, 10, 85],
    ["Dell", 1024, 11, 110],
    ["Lenovo", 512, 9, 80],
    ["HP", 256, 8, 60]
]

# HTML table for display
table_html_3d = "<table><thead><tr>{}</tr></thead><tbody>{}</tbody></table>".format(
    "".join([f"<th style='padding: 6px 12px; text-align: left;'>{h}</th>" for h in headers_3d]),
    "".join([
        "<tr>{}</tr>".format("".join(
            [f"<td style='padding: 6px 12px;'>{cell}</td>" for cell in row]
        )) for row in rows_3d
    ])
)

# Display Table
st.markdown("#### Laptop Specs Sample (Now 3 Features)")
st.markdown(table_html_3d, unsafe_allow_html=True)

# 3D Scatter Plot
plot_3d_scatter(X_3d, obs_3d, feat_3d, "3D Plot: Storage vs Battery vs Price (No PCA Needed)")

# Insights
st.markdown("### What We Observe from the 3D Plot:")
st.markdown("""
- Dell is the most powerful and expensive — ideal for performance users.
- HP clearly targets budget-conscious buyers.
- ASUS and Lenovo have same storage but differ in price and battery.
- Visualization in 3D is possible — but starts to get cluttered as features grow.
""")


# --- PCA Section ---

# --- PCA Section Starts ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.header("PCA: When Features Grow Beyond 3")
st.markdown("With more than 3 features, direct visualization becomes impractical — PCA helps reduce dimensions while preserving key patterns")

# 📦 Laptop specs with 5 features each
X = np.array([
    [3.2, 16, 512, 10, 85],   # Asus 1
    [2.8, 8, 256, 9, 60],     # Asus 2
    [3.4, 16, 1024, 11, 110], # Dell 1
    [2.6, 8, 256, 7, 55],     # Dell 2
    [3.0, 12, 512, 9, 80],    # Lenovo 1
    [2.9, 16, 512, 10, 90],   # Lenovo 2
    [3.1, 8, 256, 8, 65],     # HP 1
    [2.7, 12, 512, 9, 75]     # HP 2
])
feature_names = ["CPU Speed (GHz)", "RAM (GB)", "Storage (GB)", "Battery (hrs)", "Price (k INR)"]
observation_names = [
    "Asus 1", "Asus 2", "Dell 1", "Dell 2",
    "Lenovo 1", "Lenovo 2", "HP 1", "HP 2"
]

# Display raw data as a DataFrame
st.markdown("### Raw Laptop Feature Matrix (5 Features)")
st.dataframe(
    {"Laptop": observation_names, **{fn: X[:, i] for i, fn in enumerate(feature_names)}},
    use_container_width=True
)

st.markdown("### PCA Steps: Eigen Decomposition Approach")
st.markdown("""
We will use **Eigen Decomposition of the Covariance Matrix** to perform PCA.

#### Step-by-step Process:

1. **Standardize the Data**  
 Subtract the mean and divide by standard deviation for each feature, so that all features contribute equally.

2. **Compute the Covariance Matrix**  
 Construct a matrix that captures how standardized features vary with each other.

3. **Perform Eigen Decomposition & Select Principal Components**  
 Get eigenvalues and corresponding eigenvectors of the covariance matrix.  
 Sort the eigenvalues in descending order and reorder eigenvectors accordingly.  
 Select the top `k` eigenvectors (those explaining most variance) as principal components.

4. **Understand PCA Loadings**  
 Multiply each eigenvector by the square root of its eigenvalue to get the **loadings**.  
 Loadings show how much each original feature contributes to each principal component.

5. **Project the Data**  
 Multiply the standardized data matrix with the selected top `k` eigenvectors to transform it into reduced dimensional space.
""")


# --- PCA Execution: Step-by-Step with Math and Explanation ---

st.markdown("### PCA Step-by-Step using Eigen Decomposition")

# Step 1: Standardize the Data
st.markdown("#### Step 1: Standardize the Data")
st.markdown("We subtract the mean and divide by standard deviation for each feature to bring all features to the same scale.")
st.markdown(r"Mathematically:  $X_{standardized} = \frac{X - \mu}{\sigma}$")
X_standardized, X_mean, X_std = standardize_data(X)
st.dataframe(
    {"Laptop": observation_names, **{fn: np.round(X_standardized[:, i], 3) for i, fn in enumerate(feature_names)}},
    use_container_width=True
)

# Step 2: Compute Covariance Matrix
st.markdown("#### Step 2: Compute the Covariance Matrix")
st.markdown("We calculate the covariance matrix to capture how pairs of features vary together. Note: we use the standardized matrix, not the original raw data.")
st.markdown(r"Mathematically:  $Cov(X) = \frac{1}{n - 1} \cdot X_{standardized}^\top X_{standardized}$")
cov_matrix = covariance_matrix(X_standardized)
st.dataframe(
    np.round(cov_matrix, 3),
    use_container_width=True,
    height=250
)

# Step 3: Eigen Decomposition
eigenvalues_sorted, eigenvectors_sorted = pca_eigen_decomposition(cov_matrix)

st.markdown("#### Step 3: Eigen Decomposition & Selecting Principal Components")

st.markdown("""
We decompose the **covariance matrix** to get:

- **Eigenvectors**: directions (axes) that define new dimensions.
- **Eigenvalues**: how much variance (information) lies along each direction.

We then **sort the eigenvalues in descending order** and **reorder the eigenvectors** to match.  
This ensures the most informative directions (principal components) come first.

> Mathematically:  **Cov(X) · v = λ · v**

We keep the top principal components (like **PCA1** and **PCA2**) that capture the highest variance in the dataset.
""")

# Explained variance ratio table
explained_var = explained_variance_ratio(eigenvalues_sorted)
st.markdown("##### Explained Variance by Principal Components")
st.dataframe({
    "Eigenvalue": np.round(eigenvalues_sorted, 3),
    "Explained Variance Ratio": np.round(explained_var, 3)
})

# Dynamically compute top 2 variance contribution
pca1_var = explained_var[0] * 100
pca2_var = explained_var[1] * 100
total_var = pca1_var + pca2_var
loss = 100 - total_var

# Display example with real data
st.markdown(f"""
**Example:**  
If **PCA1** has an explained variance of **{pca1_var:.1f}%**, and **PCA2** has **{pca2_var:.1f}%**,  
then together they capture **{total_var:.1f}%** of the dataset’s structure —  meaning we can reduce dimensionality from 5 to 2 with minimal loss of information.
""")


# --- Step 4: Understand PCA Loadings ---
st.markdown("#### Step 4: Understand PCA Loadings Before Projection")

st.markdown(r"""
Each **Principal Component** is a linear combination of the original features.  
We now examine the **loadings** — weights that show how much each feature contributes to a component.

> **PCA loadings are the correlation-like weights that show how much each original feature contributes to a principal component.**

Mathematically (for standardized data):  
- $\text{Loadings} = \text{Eigenvectors} \times \sqrt{\text{Eigenvalue}}$  
- $\text{PCA}_1 = w_{11} \cdot x_1 + w_{12} \cdot x_2 + \dots + w_{1n} \cdot x_n$
""", unsafe_allow_html=True)

# Calculate loadings (scaled eigenvectors)
sqrt_eigenvalues = np.sqrt(eigenvalues_sorted[:2])
pca1_loadings = eigenvectors_sorted[:, 0] * sqrt_eigenvalues[0]
pca2_loadings = eigenvectors_sorted[:, 1] * sqrt_eigenvalues[1]

# Tables
st.markdown("**PCA1 Loadings (Most Important Direction)**")
st.dataframe(
    {"Feature": feature_names, "PCA1 Loading": np.round(pca1_loadings, 3)},
    use_container_width=True
)

st.markdown("**PCA2 Loadings (Second Important Direction)**")
st.dataframe(
    {"Feature": feature_names, "PCA2 Loading": np.round(pca2_loadings, 3)},
    use_container_width=True
)

# Show PCA equations
pca1_eq = " + ".join([f"{round(w, 2)} × {f}" for w, f in zip(pca1_loadings, feature_names)])
pca2_eq = " + ".join([f"{round(w, 2)} × {f}" for w, f in zip(pca2_loadings, feature_names)])

st.markdown("##### PCA1 Equation:")
st.markdown(f"`PCA1 = {pca1_eq}`")
st.markdown("##### PCA2 Equation:")
st.markdown(f"`PCA2 = {pca2_eq}`")

# Bar Plots
plot_bar_loadings(feature_names, pca1_loadings, pca2_loadings)

# --- Step 5: Project Data ---
st.markdown("#### Step 5: Project the Data onto Principal Components (k = 2)")

st.markdown("""
We now transform the original data into a lower-dimensional space using the top 2 eigenvectors.

Mathematically:  
  $Z = X_{standardized} \cdot W_k$  
Where:  
- $X_{standardized}$ shape = **(8 × 5)**  
- $W_k$ (top 2 eigenvectors) shape = **(5 × 2)**  
- $Z$ = Projected data in PCA space = **(8 × 2)**

<b>What are PCA1 and PCA2?</b><br>
- <b>PCA1</b> (Principal Component 1) is the direction (linear combination of original features) that captures the maximum variance in the data. It is the most important axis found by PCA.  
- <b>PCA2</b> (Principal Component 2) is the next most important direction, orthogonal to PCA1, capturing the next highest variance.
""", unsafe_allow_html=True)

U_k = project_data(X_standardized, eigenvectors_sorted, k=2)
st.dataframe(
    {"Laptop": observation_names, "PCA1": np.round(U_k[:, 0], 3), "PCA2": np.round(U_k[:, 1], 3)},
    use_container_width=True
)

# --- Step 6: Visualize the Projected Data in PCA Space ---
st.markdown("---")
st.header("Visualize 2D PCA")

st.markdown("""
Now that we’ve reduced the dataset to two dimensions (PCA1 and PCA2),  
we can plot the data to explore patterns and relationships.

This 2D PCA plot reveals clusters, similarities, or separations between the laptops.

**Suggestions for PCA Plot:**
- Use color or marker style to highlight different brands or categories if available.
- Add tooltips or hover text for more interactive exploration (Streamlit supports this with Plotly or Altair if desired).
""")

# 2D PCA Scatter Plot
plot_pca_scatter(U_k, observation_names, "2D Projection After PCA (Top 2 Principal Components)")

# --- Step 7: Insights from PCA Projection ---
st.header("Insights from PCA Projection and Loadings")

st.markdown("""
The PCA projection reveals clear patterns when we interpret the **principal components** along with their **feature loadings**:

1. **PCA1** captures the overall *spec strength* — it has high positive weights across all features:  
   `PCA1 = 0.86 × CPU + 0.97 × RAM + 0.99 × Storage + 1.01 × Battery + 1.06 × Price`  
   - Laptops with high PCA1 (e.g., **Dell 1**) have better specs and are high-end.  
   - Low PCA1 values (e.g., **Dell 2**) indicate budget or low-spec devices.

2. **PCA2** contrasts *CPU* against other specs:  
   `PCA2 = 0.62 × CPU − 0.33 × RAM − 0.05 × Storage − 0.1 × Battery − 0.06 × Price`  
   - High PCA2 scores (e.g., **HP 1**) suggest strong CPU but relatively lower RAM, storage, or price.  
   - Low PCA2 scores (e.g., **Lenovo 2**, **HP 2**) suggest weaker CPU performance.

3. **Dell 1** stands out on the far right (high PCA1) — highest specs and premium build.

4. **Dell 2** appears on the far left — low on all fronts, likely a basic model.

5. **HP 1**, at the top of the plot, is CPU-focused but may be weaker in RAM or price.

6. **Lenovo 2** and **HP 2** cluster in the bottom quadrant — they score low on both components, possibly balanced but lower-tier models.

7. **ASUS 1**, **ASUS 2**, and **Lenovo 1** sit around the middle — they offer balanced configurations, neither budget nor flagship.

These insights, driven by PCA and loadings, allow us to interpret the data structure, compare models visually, and reduce dimensionality with minimal loss of information.
""")

# --- Scree Plot Section ---
st.markdown("---")
st.header("Scree Plot: Explained Variance by All Principal Components")
fig_scree, ax_scree = plt.subplots()
ax_scree.plot(np.arange(1, len(explained_var)+1), explained_var, marker='o')
ax_scree.set_xlabel("Principal Component")
ax_scree.set_ylabel("Explained Variance Ratio")
ax_scree.set_title("Scree Plot")
st.pyplot(fig_scree)
