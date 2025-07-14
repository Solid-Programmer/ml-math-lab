import streamlit as st
import sys
import os

# Make sure we can import from MLE_Distribution_Fitting
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Logistic_Loss_Landscape.generate_data import generate_2d_classification_data
from Logistic_Loss_Landscape.loss_surface import compute_loss_surface
from Logistic_Loss_Landscape.plot_surface import plot_loss_surface_3d, plot_loss_contour

# --- Page Config ---
st.set_page_config(page_title="Logistic Loss Landscape", layout="centered")
st.title("Logistic Loss Landscape")

# --- Intro Section ---
st.markdown("""
### What is Logistic Regression?

**Logistic Regression** is a supervised learning algorithm used primarily for **binary classification**, but it can also be extended to **multiclass classification**.

#### Binary vs. Multiclass Logistic Regression

- **Binary Logistic Regression** predicts the probability of one of two classes (e.g., spam or not spam) using the sigmoid function.
- **Multiclass Logistic Regression** is handled by:
  - **One-vs-Rest (OvR):** One binary classifier per class.
  - **Softmax Regression:** A generalization using the softmax function to model multiple exclusive classes.

#### Key Differences from Linear Regression

| Logistic Regression                    | Linear Regression                      |
|----------------------------------------|----------------------------------------|
| Output is a probability (0 to 1)       | Output is a continuous value           |
| Used for classification problems       | Used for regression problems           |
| Applies sigmoid/softmax to linear sum  | Uses raw linear output                 |
| Trained using cross-entropy loss       | Trained using mean squared error (MSE) |

#### Why Logistic Regression?

- Provides **interpretable model weights**.
- Outputs **probabilities** for decision-making.
- Has a **convex loss function**, making optimization efficient.
- Serves as a strong **baseline classifier** in many ML pipelines.

This app visualizes how the **logistic loss changes** with different weight values (w₁, w₂), building intuition for model behavior in parameter space.
""")

# --- Intuition Section: Explain Weights, Bias, and Resolution ---
st.markdown("""
---

### How Does Logistic Regression Work?

We’re teaching a model to predict if a fruit is an **apple (1)** or **not (0)**.

1. **Features as Numbers**  
   If the fruit is red → x₁ = 1, otherwise 0  
   If the fruit is heavy → x₂ = 1, otherwise 0

2. **Assign Importance (Weights)**  
   w₁ = 3.0 (color), w₂ = 1.0 (weight), b = -2.0

3. **Compute a Score**  
   _Formula:_  
""", unsafe_allow_html=True)

st.latex(r"z = w_1 \cdot x_1 + w_2 \cdot x_2 + b")
st.latex(r"z = 3 \cdot 1 + 1 \cdot 1 - 2 = 2")

st.markdown("""
4. **Make It a Probability**  
   _Sigmoid function:_  
""")

st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
st.latex(r"\sigma(2) \approx 0.88")

st.markdown("So, the model is **88% sure** this is an apple!")

# --- Sigmoid and Bias Explanation ---
st.markdown("---")
st.markdown("### What Happens After Calculating the Score?")

st.markdown("After we compute the weighted sum of the inputs and add the bias:")

st.latex(r"z = w_1 \cdot x_1 + w_2 \cdot x_2 + b")

st.markdown("We pass this value to the **sigmoid function** to squash the score into a probability:")

st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")

st.markdown("This output is always between **0 and 1**, which makes it perfect for predicting probabilities.")

st.markdown("#### Why Do We Use Sigmoid?")
st.markdown("""
- It converts any number into a range between **0 and 1**.
- The output tells us **how confident** the model is.
- If \\( sigma(z) > 0.5 \\), we usually predict **class 1**.
- If \\( sigma(z) < 0.5 \\), we predict **class 0**.
""")

st.markdown("#### What Is the Bias Term?")
st.markdown("""
- The **bias** \\( b \\) is a constant added to the score before applying sigmoid.
- It helps the model shift the output when features are weak or missing.
- Think of it like a starting adjustment — like tilting the scale slightly.
- Without bias, the model could always be wrong when all input features are 0.
""")

st.markdown("#### Full Prediction Flow")
st.markdown("""
1. Multiply features by weights to compute the score  
2. Add bias to adjust the score  
3. Apply sigmoid to get a probability  
4. Predict class based on that probability  
""")

st.markdown("---")

st.markdown("Now we are ready to explore how different values of weights and bias affect the model's predictions!")


# --- Layout Controls in Columns with Spacing ---
col1, spacer1, col2 = st.columns([2, 0.5, 2])
with col1:
    w1_range = st.slider("w₁ Range", -10.0, 10.0, (-5.0, 5.0), step=0.5)
with col2:
    w2_range = st.slider("w₂ Range", -10.0, 10.0, (-5.0, 5.0), step=0.5)

col3 = st.columns([1, 2])
with col3[0]:
    bias = st.slider("Bias (b)", -10.0, 10.0, 0.0, step=0.5)

# --- Generate Data ---
df = generate_2d_classification_data(n_samples=200, centers=2, random_state=42)

# --- Compute Loss Surface ---
W1, W2, Z, min_point = compute_loss_surface(df, w1_range, w2_range, b=bias)


# --- Plot Tabs ---
tab1, tab2 = st.tabs(["3D Loss Surface", "2D Contour Map"])
with tab1:
    fig3d = plot_loss_surface_3d(W1, W2, Z, min_point=min_point)
    st.plotly_chart(fig3d, use_container_width=True)
with tab2:
    fig2d = plot_loss_contour(W1, W2, Z, min_point=min_point)
    st.plotly_chart(fig2d, use_container_width=True)

st.markdown("### Loss Function Used (Cross-Entropy)")
st.latex(r"\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]")
st.markdown("""
- This measures how well the predicted probabilities match the actual labels.
- Lower loss means better predictions.
- Our plots show how this loss changes with different weights.
""")


st.markdown("---")

st.markdown("""
### What Are We Visualizing?

We're plotting how the **logistic regression loss** changes across different combinations of weights w_1 and w_2, while keeping the bias - b fixed.

- You set a range for w_1 and w_2 — this defines a **grid of weight values**.
- For each weight pair, we calculate how wrong the model is (cross-entropy loss).
- These loss values are used to plot:

  - **3D Loss Surface**: A landscape showing how the model's error changes with weights  
  - **2D Contour Plot**: A top-down view showing curves of equal loss

---

### Why This Helps

- Shows **where the loss is lowest** — the best weight combination
- Helps visualize **how sensitive** the model is to changes in weights
- Lets you see **why logistic regression has a single, clear minimum**
- Reveals what optimizers like **gradient descent** are trying to find

In short: this helps us to **see** the math behind training — and find the weights that make our model perform best.
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
