import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Make sure we can import from MLE_Distribution_Fitting
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MLE_Distribution_Fitting.distribution_fitter import fit_normal, fit_poisson, fit_exponential
from MLE_Distribution_Fitting.mle_plot import plot_distribution_fit

st.set_page_config(page_title="MLE Distribution Fitting Tool", layout="centered")
st.title("MLE Distribution Fitting Tool")

# --- Introduction ---
st.markdown("""
### What is Maximum Likelihood Estimation (MLE)?
Maximum Likelihood Estimation (MLE) is a way to find the best values (parameters) for a probability distribution that could have generated the given data.  
In simple terms, it looks for the values that make the data we observed most likely.

---

### What is a Probability Distribution?
A probability distribution tells us how values are spread or how likely different outcomes are in a dataset.  
Some common types are:

- **Normal distribution**: Looks like a bell curve, used when values are centered and spread evenly.
- **Poisson distribution**: Used for counting how often something happens (e.g., emails per hour).
- **Exponential distribution**: Used to model how much time passes between events (e.g., time between breakdowns).

---

### Where It Helps and How

Fitting a distribution using MLE is useful in many real-life situations, such as:

- **Predicting system failures**: Exponential distribution can help model time between failures in machines.
- **Modeling customer arrivals**: Poisson distribution is used to model the number of customers arriving at a store or calls to a helpline.
- **Estimating height patterns**: Normal distribution can be used to understand variation in human height or weight across a population.

This tool helps you find the best-fit distribution and its parameters, so you can better understand patterns, make predictions, or simulate data based on real-world behavior.

---
""")

# --- User Input ---
st.markdown("### Select Distribution and Sample Size")
distribution = st.selectbox("Distribution", options=["Normal", "Poisson", "Exponential"])
sample_size = st.slider("Select number of data points to generate", min_value=10, max_value=2000, value=500, step=100)

if st.button("Generate Data"):
    # --- Simulate Data ---
    if distribution == "Normal":
        data = np.random.normal(loc=0, scale=1, size=sample_size)
        result = fit_normal(data)

    elif distribution == "Poisson":
        data = np.random.poisson(lam=5, size=sample_size)
        result = fit_poisson(data)

    elif distribution == "Exponential":
        data = np.random.exponential(scale=1.0, size=sample_size)
        result = fit_exponential(data)

    else:
        st.error("Unsupported distribution.")
        st.stop()

    if distribution == "Normal":
        st.info("Note: Poisson and Exponential may not fit well because Normal data can be negative or non-integer.")


    # --- Show Parameter Estimates ---
    st.success(f"Generated {sample_size} samples from {distribution} distribution.")
    st.markdown("### MLE Estimated Parameters")
    for key, value in result.items():
        if key != "log_likelihood":
            st.write(f"**{key}**: {value:.4f}")
    st.write(f"Log-Likelihood: **{result['log_likelihood']:.2f}**")

        # --- Explain Parameters ---
    st.markdown("### What These Parameters Mean")

    if distribution == "Normal":
        st.markdown("""
        - **μ (mean)** tells us where the center of the data is — the most common or average value.
        - **σ (standard deviation)** shows how spread out the data is. A larger σ means more variation.
        """)

    elif distribution == "Poisson":
        st.markdown("""
        - **λ (lambda)** is the average number of times an event happens in a fixed interval (e.g., 5 calls/hour).
        - It represents both the mean and variance for a Poisson process.
        """)

    elif distribution == "Exponential":
        st.markdown("""
        - **λ (lambda)** is the **rate** at which events happen (e.g., 2 failures/hour).
        - The **mean time between events** is `1 / λ`, so a higher λ means shorter gaps between events.
        - For example, if λ = 0.5, then on average an event happens every 2 units of time.
        """)


    # --- Plot ---
    st.markdown("### Histogram with Fitted Curve")
    fig = plot_distribution_fit(data, dist_name=distribution.lower(), params=result)
    st.pyplot(fig)
    st.markdown("---")


    # --- Comparison Section ---
    st.markdown("### What Log-Likelihood Tells Us")
    st.markdown("""
    - The **log-likelihood** measures how well the chosen distribution, with the fitted parameters, explains the observed data.
    - A **higher log-likelihood** means the model is more likely to have generated the data.
    - It helps us compare different models (distributions) for the same data.
    """)

    # Fit all three, regardless of validity
    normal_fit = fit_normal(data)
    poisson_fit = fit_poisson(data)
    exponential_fit = fit_exponential(data)

    fits = {
        "Normal": normal_fit["log_likelihood"],
        "Poisson": poisson_fit["log_likelihood"],
        "Exponential": exponential_fit["log_likelihood"]
    }

    best_fit = max(fits.items(), key=lambda x: x[1])

    # Display log-likelihoods
    for dist, ll in fits.items():
        label = "✅ Best Fit" if dist == best_fit[0] else ""
        status = "❌ Invalid" if ll == -np.inf else ""
        st.write(f"- **{dist}**: {ll:.2f} {label} {status}")

    # Plot bar chart
    fig2, ax = plt.subplots()
    colors = ["blue" if k == best_fit[0] else "gray" for k in fits.keys()]
    ax.bar(fits.keys(), fits.values(), color=colors)
    ax.set_title("Log-Likelihoods for All Distributions")
    ax.set_ylabel("Log-Likelihood")
    ax.grid(True, axis="y")
    st.pyplot(fig2)
