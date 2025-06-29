import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .gradient_math import mae, mae_gradient, brute_force_mae_grid, gradient_descent_mae, optimizer_path_mae
from .gradient_plot import plot_line_fit, plot_final_fit, plot_mae_contour, plot_optimizer_path

# ----------------------------
# Title & Data Table
# ----------------------------
st.title("Gradient Descent Playground")

st.markdown("""
### What is Gradient Descent?

Gradient Descent is an **optimization algorithm** used in machine learning to **minimize a loss function**.

- A **loss function** tells us how wrong our model's predictions are.
- The goal of training is to **adjust model parameters** (like slope `m` and intercept `b`) so that the loss becomes as small as possible.
- Gradient Descent does this by taking **small steps downhill on the loss curve** — moving in the direction that reduces error the fastest.

""")

st.markdown("""
### How Does It Work?

- Think of the **loss surface** as a landscape showing error values for different parameter settings.
- The number of parameters determines the shape of this landscape:
  - 1 parameter → **1D curve**
  - 2 parameters (e.g., `m` and `b`) → **2D surface**
  - 3 parameters → **3D surface**
  - Many parameters (like in deep learning) → **high-dimensional surface**
- Gradient Descent calculates the slope (gradient) and updates parameters to **move closer to the minimum**.

""")

st.markdown("""
### What is a Loss Function?

A **loss function** quantifies how well the model's predictions match the actual data.

- It's the guide that tells Gradient Descent which direction to go.
- Low loss = good predictions, high loss = poor predictions.
- We want to **minimize the loss** during training.

""")

st.markdown("""
### Common Loss Functions in ML

| **Loss Function**           | **Used In**                            | **What It Does**                                                              |
|----------------------------|----------------------------------------|-------------------------------------------------------------------------------|
| **Mean Squared Error (MSE)**      | Regression, Deep Nets                 | Penalizes larger errors more. Smooth and widely used.                         |
| **Mean Absolute Error (MAE)**     | Robust Regression                    | Treats all errors equally. More resilient to outliers.                        |
| **Binary Cross-Entropy**          | Binary Classification                | Measures how close predicted probs are to 0 or 1 labels.                      |
| **Categorical Cross-Entropy**     | Multi-class Classification (Softmax) | Measures error for multiple class probabilities.                              |
| **KL Divergence**                 | VAEs, Probabilistic Models           | Measures how one probability distribution differs from a target distribution. |

""")


st.markdown("---")
st.markdown("---")
# ----------------------------
# Phase 1: Setup and Starting with MAE
# ----------------------------
st.markdown("## Phase 1: Starting with Data and MAE Loss")

st.markdown("""
To begin our Gradient Descent journey, we’ll start with a simple dataset of a household electricity bill over 12 months. For simplicity, we'll use **Mean Absolute Error (MAE)** as our loss function.

MAE is easy to understand — it just takes the average of how far off our predictions are from actual values (ignoring direction).
""")

# Data Setup
months = np.array(range(1, 13))
bills_in_inr = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 9.0, 6.0, 5.5, 10.0, 6.0, 2.0])
formatted_bills = [f"{val:.1f}K" for val in bills_in_inr]
df = pd.DataFrame({'Month': months, 'Electricity Bill (INR)': formatted_bills})

# Show data table
st.markdown("Here is the dataset:")
st.table(df)

st.markdown("<br>", unsafe_allow_html=True)
# ----------------------------
# Layout: Controls & Chart
# ----------------------------

st.markdown("""
We're trying to **fit a straight line** to the data using a simple linear model:
""")
st.latex(r"\hat{y} = m \cdot x + b")
st.markdown("""
This means we assume electricity bills grow (or drop) linearly with time.

- **Slope (m)** controls how steep the line is  
- **Intercept (b)** shifts the line up or down  

Use the sliders below to adjust `m` and `b` manually to cover as many data points as possible with the line.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div style='font-size:18px; font-weight:600;'>Model Parameters</div>", unsafe_allow_html=True)
    m = st.slider("Slope (m)", min_value=-1.0, max_value=2.0, value=1.0, step=0.1)
    b = st.slider("Intercept (b)", min_value=-5.0, max_value=10.0, value=0.0, step=0.5)

    predicted = m * months + b
    mae_val = mae(bills_in_inr, predicted)
    st.markdown("**MAE (Mean Absolute Error):**")
    st.latex(r"MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|")
    st.markdown(f"**Current MAE:** `{mae_val:.2f} K INR`")

with col2:
    fig = plot_line_fit(months, bills_in_inr, m, b, mae_val)
    st.pyplot(fig)

# ----------------------------
# Info Section
# ----------------------------
st.markdown(
    """
    <div style='background-color:#fff3cd; padding:8px; border-radius:4px; font-weight:bold; width:100%; display:block;'>
        MAE is the mean absolute error of your model. It measures how much error your model is making. We always try to minimize this.
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Gradient Descent Explanation
# ----------------------------
st.markdown("---")
st.markdown("## Gradient Descent for minimizing MAE")

st.markdown("""
We want to find the best values of **m** and **b** using **gradient descent**.

### Steps:
1. **Define the loss function (MAE):**

   $$
   L = \\frac{1}{n} \\sum_{i=1}^n \\left| y_i - (m x_i + b) \\right|
   $$

2. **Compute partial derivatives:** ∂L/∂m and ∂L/∂b.

3. **Update using gradient descent:**

   $$
   m_{\\text{new}} = m_{\\text{old}} - \\eta \\cdot \\frac{\\partial L}{\\partial m}
   \\quad\\quad
   b_{\\text{new}} = b_{\\text{old}} - \\eta \\cdot \\frac{\\partial L}{\\partial b}
   $$

<span style='font-size:14px; color:gray'>
<b>Note:</b> One iteration is one update of m and b. We repeat multiple times to reduce MAE. We stop when MAE stops improving or after a set number of steps.
</span>
""", unsafe_allow_html=True)

# ----------------------------
# Gradient Descent Updates
# ----------------------------
st.markdown("---")
st.markdown("## Gradient Descent Updates")

st.markdown("""
**Learning Rate (η):**  
Controls how big a step we take in each update.  
- Too small → slow learning  
- Too large → may overshoot or never converge  
- A typical value: **0.01** (start small and adjust if needed)

**Epoch:**  
One full update of the model parameters (`m`, `b`) using all the data.  
- More epochs → more steps toward the optimal solution  
- Too many may not help if error isn’t improving  
- You can stop when MAE stops decreasing, or after a fixed number.)

**Patience:**  
This tells the optimizer **how many steps to wait** without improvement before stopping early.  
- Prevents over-training  
- Saves time by stopping when we're not improving

> *Example:* If patience = 20, and MAE hasn’t improved for 20 epochs, we stop.
Choose a **learning rate** and number of **epochs** that balance speed and stability.
""")


col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div style='font-size:18px; font-weight:600;'>Gradient Descent Settings</div>", unsafe_allow_html=True)
    eta = st.slider("Learning Rate (η)", 0.001, 0.1, 0.01, 0.001, key="eta_slider")
    epochs = st.slider("Epochs", 10, 500, 300, 10, key="epoch_slider")
    patience = st.slider("Patience (Early Stopping)", 1, 50, 20, 1, key="patience_slider")

with col2:
    m_init, b_init = 1.0, 0.0
    history, (final_m, final_b), final_mae, final_epoch = gradient_descent_mae(
        bills_in_inr, months, m_init, b_init, eta, epochs, patience
    )
    df = pd.DataFrame(history)

    def highlight_initial(row):
        return ['background-color: #dff0d8' if row['Epoch'] == 0 else '' for _ in row]

    st.markdown("### Epoch-wise Parameter Updates")
    st.dataframe(df.style.apply(highlight_initial, axis=1), use_container_width=True)

# --- Final values after last epoch or early stop ---
st.markdown(f"""
<div style='font-size:18px; padding:10px; background-color:#f9f9f9; border-left: 4px solid #4CAF50; border-radius:6px'>
<b>MAE:</b> {final_mae:.4f}  
<br><b>Epoch:</b> {final_epoch}  
<br><b>Parameters:</b> m = {final_m:.4f}, b = {final_b:.4f}
</div>
""", unsafe_allow_html=True)


# --- Note and final values ---
st.markdown("""
After a certain number of epochs, the values of **m** and **b** stop changing significantly. This is where we can consider **stopping** gradient descent.

Below are the actual optimized parameters:
""")

st.markdown(f"""
<div style='font-size:20px; font-weight:600; padding:10px; background-color:#f5f5f5; border-radius:8px'>
 <b>Final MAE:</b> 1.28 K INR<br>
 <b>Achieved at Epoch:</b> 208<br>
 <b>Parameters:</b> m = 0.3960, b = 2.8160<br>
 <b>Learning Rate:</b> 0.05
</div>
""", unsafe_allow_html=True)



# --- Final Plot with fitted line ---
fig_final = plot_final_fit(months, bills_in_inr, 0.40, 2.82, 1.28)
st.pyplot(fig_final)

# ----------------------------
# Predict Using Learned Model
# ----------------------------
st.markdown("### Let's use the Model to Predict Bill")

st.markdown("""
The final model is a simple line:

$$
\\text{Predicted Bill} = m \\cdot \\text{Month} + b
$$

This model can now be used to predict the electricity bill for any given month.
""")

# Input for month (x)
input_month = st.number_input("Enter month number (1–12):", min_value=1, max_value=12, value=6)

# Actual and predicted bills
actual_bill = bills_in_inr[input_month - 1]
predicted_bill = 0.4 * input_month + 2.82

# Output block
st.markdown(f"""
<div style='font-size:20px; font-weight:500; padding:10px; background-color:#e6f2ff; border-radius:6px'>
📅 Month: {input_month}  
<br>📌 Actual Electricity Bill: <b>{actual_bill:.2f} K INR</b>  
<br>🔮 Predicted Electricity Bill: <b>{predicted_bill:.2f} K INR</b>
</div>
""", unsafe_allow_html=True)


# Disclaimer
st.markdown("<small>⚠️ <i>This model is a simplified demo using MAE and a straight line. It's meant to help understand how gradient descent works, not to make highly accurate predictions.</i></small>", unsafe_allow_html=True)


# ----------------------------
# MAE Cost Surface Visualization
# ----------------------------
st.markdown("---")
st.markdown("## MAE Cost Surface and Optimizer Path")

st.markdown("""
We now visualize how the MAE changes over different values of **m** (slope) and **b** (intercept), and how gradient descent moves on this surface.
""")

# --- Grid for MAE surface ---
m_range = (-1.0, 2.0)
b_range = (-5.0, 10.0)
M, B, Z = brute_force_mae_grid(bills_in_inr, months, m_range, b_range)

# --- Run Gradient Descent using existing function ---
m_init, b_init = 1.0, 0.0
history, best_params, best_mae, best_epoch = gradient_descent_mae(
    y_true=bills_in_inr,
    x=months,
    m_init=1.0,
    b_init=0.0,
    eta=eta,
    epochs=epochs,
    patience=patience
)

# --- Extract path ---
path = np.array([(row["m"], row["b"]) for row in history])
final_m, final_b = best_params
final_mae = best_mae

# --- Contour plot ---
fig_contour, ax_c = plt.subplots(figsize=(8, 6))
contour = ax_c.contourf(M, B, Z, levels=50, cmap='viridis')
cbar = plt.colorbar(contour, ax=ax_c)
cbar.set_label("MAE")

# --- Plot optimizer path ---
ax_c.plot(path[:, 0], path[:, 1], 'o-', color='red', markersize=4, linewidth=2, label='Optimizer Path')
ax_c.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start')
ax_c.plot(final_m, final_b, 'bo', markersize=8, label='Best')

# --- Annotate final MAE ---
ax_c.annotate(f"Final MAE = {final_mae:.2f}", xy=(final_m, final_b), xytext=(final_m + 0.2, final_b + 0.5),
              arrowprops=dict(facecolor='black', shrink=0.05),
              fontsize=10, backgroundcolor='white')

ax_c.set_xlabel("Slope (m)")
ax_c.set_ylabel("Intercept (b)")
ax_c.set_title("MAE Contour with Optimizer Path")
ax_c.legend()
ax_c.grid(True)

st.pyplot(fig_contour)


# ----------------------------
# Phase 2: Optimizer Introduction
# ----------------------------
st.markdown("---")
st.markdown("## Optimizers in Gradient Descent")

st.markdown("""
Earlier, we have used **vanilla Gradient Descent** with a fixed learning rate.

But in real-world ML problems, optimizers play a crucial role in making training **faster, more stable**, and **less sensitive** to learning rate choices.

Optimizers adjust how we update parameters like `m` and `b` over time using more than just the gradient.

### What Do Optimizers Do?
- Improve convergence speed  
- Reduce oscillations  
- Adapt step size automatically  
- Use momentum/history for smarter updates  

### Common Optimizers

| Optimizer | Description |
|-----------|-------------|
| **SGD (Stochastic Gradient Descent)** | Basic gradient descent; updates using gradient at each step |
| **Momentum** | Adds a velocity term to dampen oscillations and speed up convergence |
| **Nesterov Momentum** | Looks ahead before taking the momentum step; improves on basic momentum |
| **AdaGrad** | Adapts learning rate for each parameter based on past gradients |
| **RMSProp** | Uses moving average of squared gradients to normalize updates |
| **Adam** | Combines Momentum and RMSProp; widely used and stable in practice |
""")


