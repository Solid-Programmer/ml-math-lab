import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bias-Variance Tradeoff", layout="centered")

st.title("Bias-Variance Tradeoff")

st.markdown("""
## What is Bias?

**Bias** refers to the error caused by a model's overly simplistic assumptions about the true relationship between inputs and outputs.

A **high-bias** model:
- Makes strong assumptions
- Cannot capture the complexity of the data
- Often performs poorly on both training and test sets
- Leads to **underfitting**

## What is Variance?

**Variance** refers to the error caused by the model's sensitivity to small changes in the training dataset.

A **high-variance** model:
- Fits the training data very closely (possibly including noise)
- Performs poorly on unseen data
- Leads to **overfitting**

---

## Approximating Bias and Variance

In practice, we can use a simple formula to approximate:

Bias ≈ Training Error
\nVariance ≈ Dev Error − Training Error
            

- **Bias** tells us how well the model fits the training data.
- **Variance** tells us how much worse the model performs on the dev (validation) data compared to training data.

> These are rough estimates used to understand how the model is behaving. They are not exact mathematical definitions.

---

#### Example 1: High Bias (Underfitting)

Suppose your classifier produces:
- Training error = 15%
- Dev (validation) error = 16%

Using our formula:
- Bias ≈ 15%
- Variance ≈ 16% − 15% = 1%

**Interpretation:**
- The model performs poorly even on the training data.
- Its dev error is only slightly higher, so it is not very sensitive to new data.
- This suggests the model is too simple and is **underfitting**.

----

#### Example 2: High Variance (Overfitting)

Suppose your classifier produces:
- Training error = 1%
- Dev error = 11%

Using our formula:
- Bias ≈ 1%
- Variance ≈ 11% − 1% = 10%

**Interpretation:**
- The model performs very well on training data.
- But the dev error is much higher, indicating poor generalization.
- This is a case of **overfitting**, where the model has **low bias** but **high variance**.
            
----

#### Example 3: High Bias and High Variance

Suppose your classifier produces:
- Training error = 15%
- Dev error = 30%

Using our formula:
- Bias ≈ 15%
- Variance ≈ 30% − 15% = 15%

**Interpretation:**
- The model performs poorly on training data (high bias).
- It performs even worse on dev data (high variance).
- This is a case where the model both **underfits and fails to generalize**.

---

## Summary

| Scenario                  | Training Error | Dev Error | Bias   | Variance | Problem Type     |
|---------------------------|----------------|-----------|--------|----------|------------------|
| High Bias                 | 15%            | 16%       | 15%    | 1%       | Underfitting     |
| High Variance             | 1%             | 11%       | 1%     | 10%      | Overfitting      |
| High Bias & High Variance | 15%            | 30%       | 15%    | 15%      | Under + Overfit  |

---

""")

st.header("Demo: Polynomial Fit vs. True Function")

st.markdown("""
Imagine you're building a cat vs. dog image classifier.

- A **simple model** may only look at color and miss shapes → underfitting (**high bias**)
- A **complex model** may memorize the training images → overfitting (**high variance**)

In this demo, we simulate a similar idea:

- The **true function** is like the hidden pattern in real data
- We generate **noisy samples** like real-world observations
- We use **polynomial models** to see how complexity affects performance
""")

# Description of sliders
st.markdown("""
**Adjust the settings below:**
- **Number of samples**: Size of the training dataset
- **Noise std deviation**: Randomness added to the true function
- **Polynomial degree**: Controls model complexity
""")

st.markdown("---")

# True function
def f_true(x):
    return np.sin(x)

# --- Controls ---
col1, col2, col3 = st.columns(3)
with col1:
    n_samples = st.slider("Number of samples", 10, 200, 50, step=10)
with col2:
    noise_std = st.slider("Noise std deviation", 0.0, 1.0, 0.2, step=0.05)
with col3:
    degree = st.slider("Polynomial degree", 0, 15, 3)

# --- Generate data ---
np.random.seed(0)
x = np.random.uniform(-3, 3, size=n_samples)
y = f_true(x) + np.random.normal(0, noise_std, size=n_samples)

x = x.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Fit polynomial regression ---
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(x_train)
X_test_poly = poly.transform(x_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predictions
x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
X_plot_poly = poly.transform(x_plot)
y_pred_plot = model.predict(X_plot_poly)

# --- Evaluate error ---
train_pred = model.predict(X_train_poly)
test_pred = model.predict(X_test_poly)

train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_plot, f_true(x_plot), label="True Function (sin(x))", color="green", linewidth=2)
ax.plot(x_plot, y_pred_plot, label=f"Polynomial Fit (deg={degree})", color="blue")
ax.scatter(x_train, y_train, color="black", label="Train Data", alpha=0.6)
ax.scatter(x_test, y_test, color="red", label="Test Data", alpha=0.6)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Polynomial Regression Fit vs. True Function")
ax.legend()
st.pyplot(fig)

# --- Show errors ---
st.markdown(f"**Training MSE:** {train_mse:.3f}  **Test MSE:** {test_mse:.3f}")
# --- Interpret model behavior ---
delta = test_mse - train_mse

if train_mse < 0.1 and test_mse > 50:
    message = "The model is severely overfitting — it fits the training data too tightly but fails completely on unseen data."
elif train_mse < 0.1 and delta > 0.1:
    message = "The model is overfitting — it performs well on training data but generalization is poor."
elif train_mse > 0.15 and test_mse > 0.15 and delta < 0.1:
    message = "The model is underfitting — it's too simple and performs poorly on both training and test data."
elif train_mse < 0.1 and test_mse < 0.1 and abs(delta) < 0.05:
    message = "Great! The model fits both training and test data well — this looks like a good level of complexity."
elif delta < -0.05:
    message = "Interesting! Test error is lower than training error — possibly due to random variation in the test split."
else:
    message = "The model is learning, but there may still be room for improvement. Consider adjusting the polynomial degree or noise level."

st.info(message)

st.markdown("---")
st.subheader("Train vs. Test Error Curve")

degrees = range(0, 16)
train_errors = []
test_errors = []

for d in degrees:
    # Fit polynomial of degree d
    poly_d = PolynomialFeatures(degree=d)
    X_train_d = poly_d.fit_transform(x_train)
    X_test_d = poly_d.transform(x_test)

    model_d = LinearRegression()
    model_d.fit(X_train_d, y_train)

    y_train_pred = model_d.predict(X_train_d)
    y_test_pred = model_d.predict(X_test_d)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# Plot error vs. degree
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(degrees, train_errors, label="Training Error", marker='o')
ax2.plot(degrees, test_errors, label="Test Error", marker='o')
# Identify regions
optimal_degree = np.argmin(test_errors)
ax2.axvline(x=optimal_degree, linestyle='--', color='gray', alpha=0.6)
ax2.text(optimal_degree + 0.3, max(test_errors)*0.9, 'Optimal', color='gray')
# Annotate underfitting zone
ax2.text(1, max(test_errors)*0.85, "Underfitting\n(high bias)", fontsize=9, color='blue')
# Annotate overfitting zone
ax2.text(12, max(test_errors)*0.85, "Overfitting\n(high variance)", fontsize=9, color='orange')

ax2.set_xlabel("Model Complexity (Polynomial Degree)")
ax2.set_ylabel("Mean Squared Error")
ax2.set_title("Bias-Variance Tradeoff")
ax2.legend()
st.pyplot(fig2)
st.markdown("""
- At **low degrees**, both errors are high → model is too simple (**high bias**).
- At **high degrees**, training error drops but test error rises → model is too complex (**high variance**).
- The **optimal degree** lies in the middle — where test error is minimized.
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

