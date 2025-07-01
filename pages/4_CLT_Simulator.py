import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# Set page config
st.set_page_config(page_title="Central Limit Theorem Simulator", layout="centered")

# Title
st.title("Central Limit Theorem Simulator")

# Description
st.markdown(r"""
Welcome to the **Central Limit Theorem (CLT) Simulator**!

This tool helps you understand how the **distribution of sample means** begins to resemble a **normal distribution** as:
- The **sample size** per trial increases, and
- The **number of trials** grows.
""")

# CLT explanation
with st.expander("What is the Central Limit Theorem (CLT)?", expanded=True):
    st.markdown(r"""
    The **Central Limit Theorem (CLT)** is a foundational idea in probability and statistics.

    > If you repeatedly take random samples from a population, and compute their **means**,  
    > then as the **sample size** increases, the distribution of those sample means  
    > will tend to form a **normal distribution**, regardless of the original population's shape.

    This means even if your population is skewed or uniform (like dice rolls),  
    the **average of many such samples** will behave normally.

    ---
    **Why it matters in Machine Learning:**  
    CLT helps explain **why so many algorithms assume normality** and **why averages are reliable.**
    """)

# Simple explanation of i.i.d.
st.markdown(r"""
---
### Let’s understand it in a simple way — What does 'Independent and Identically Distributed' mean?

To apply the Central Limit Theorem correctly, the variables we sample must be:

**Independent:**  
- The outcome of one observation does not influence the others.  
- Example: Rolling a die multiple times — each roll is independent of the last.

**Identically Distributed:**  
- All the variables are drawn from the same probability distribution.  
- Example: Each dice roll has the same uniform distribution over {1, 2, 3, 4, 5, 6}.

So when we say **i.i.d. random variables**, we mean:  
"Each value is random, drawn the same way, and not affected by the others."

These conditions are essential for the Central Limit Theorem to work.
""")

st.markdown("""
---
### How Sample Means Are Computed in This Simulation

- In each **trial**, we draw a **sample of size N** from the selected distribution  
- We compute the **mean of that sample** — this gives us **1 sample mean per trial**  
- We repeat this process for **T trials**, collecting **T sample means**  
- Finally, we **plot the distribution** of these T sample means to visualize the Central Limit Theorem
""")

# Example section
st.markdown(r"""
---
### **Example: Unfair Dice and the Central Limit Theorem**

**Note:** CLT does not require the original distribution to be fair, symmetric, or normal.  
It only needs independent and identically distributed data with finite mean and variance.

**1. Random Variable:** The outcome of rolling a biased 6-sided die  
- Support: Values range from 1 to 6 (e.g., P(6) = 0.4, others = 0.12)

**2. One Observation:** Roll once → get one value, like [6]  
- A single value does not show the behavior of means

**3. Sample of Size 2:** Roll twice in one trial: [3, 6]  
- Sample mean = (3 + 6) / 2 = 4.5

**4. Repeat Trials:** Repeat sampling many times (e.g., 1000 trials)  
- Collect sample means: [4.5, 4.8, 4.2, ...]

**Note:**  
Even though the die is biased, the distribution of sample means begins to look normal.  
This is the effect of the Central Limit Theorem at work.
""")

# Layout: sliders on the left, plot on the right
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Simulation Controls")
    sample_size = st.slider("Sample size per trial (n)", min_value=1, max_value=50, value=5, step=1)
    num_trials = st.slider("Number of trials", min_value=1, max_value=2000, value=1000, step=100)

with col2:
    # Define unfair die probabilities
    die_faces = np.array([1, 2, 3, 4, 5, 6])
    die_probs = np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.4])  # Biased toward 6

    # Run simulation
    sample_means = []
    for _ in range(num_trials):
        sample = np.random.choice(die_faces, size=sample_size, p=die_probs)
        sample_means.append(np.mean(sample))
    sample_means = np.array(sample_means)

    # Calculate statistics
    mean = np.mean(sample_means)
    std = np.std(sample_means)

    # Plotting
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(sample_means, bins=40, density=True, color="skyblue", edgecolor="black", alpha=0.7)

    # Overlay normal distribution curve
    
    x = np.linspace(min(sample_means), max(sample_means), 200)
    ax.plot(x, norm.pdf(x, loc=mean, scale=std), color="darkred", lw=2, label="Normal Fit")

    # Plot vertical line at mean
    ax.axvline(mean, color="green", linestyle="--", linewidth=2, label=f"Mean = {mean:.2f}")

    # Labels and title
    ax.set_title("Distribution of Sample Means")
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()

    # Show plot
    st.pyplot(fig)

st.success(f"With sample size {sample_size} and {num_trials} trials, the sample means are approximately normally distributed.\nThis visually confirms the Central Limit Theorem.")

st.markdown("""
---
### Distributions Where CLT Applies

The **Central Limit Theorem (CLT)** works on a wide variety of distributions — as long as they meet **all three** of the following conditions:

1. **Independent observations**  
2. **Identically distributed (i.i.d.)**  
3. **Finite mean and finite variance**

Below are few examples of such distributions:
""")

dist_data = pd.DataFrame({
    "Distribution": [
        "Uniform (Discrete)",
        "Bernoulli",
        "Binomial",
        "Poisson",
        "Categorical (Discrete)",
        "Uniform (Continuous)",
        "Exponential",
        "Normal (Gaussian)"
    ],
    "Type": [
        "**Discrete**",
        "**Discrete**",
        "**Discrete**",
        "**Discrete**",
        "**Discrete**",
        "**Continuous**",
        "**Continuous**",
        "**Continuous**"
    ],
    "Example": [
        "Fair die roll (1–6 equal chance)",
        "Single coin flip (0 or 1)",
        "Number of heads in 10 flips",
        "Number of calls per minute",
        "Biased die (P(6)=0.4, others=0.12)",
        "Random number between 0 and 1",
        "Time between random events",
        "Human heights, test scores"
    ]
})

st.table(dist_data)

# Unified section: Choose and simulate
st.markdown("---")
st.markdown("### Choose a Distribution and Simulate Central Limit Theorem")

# Distribution selector
dist_choice = st.selectbox("Select a distribution to simulate", [
    "Uniform (Discrete)",
    "Bernoulli",
    "Binomial",
    "Poisson",
    "Categorical (Discrete)",
    "Uniform (Continuous)",
    "Exponential",
    "Normal (Gaussian)"
])

# One-sample description
if dist_choice == "Uniform (Discrete)":
    sample = np.random.randint(1, 7)
    st.markdown(f"""
    **Uniform (Discrete): Fair Die Roll**  
    - Random variable: outcome of rolling a fair 6-sided die  
    - Possible values: 1 to 6
    """)
elif dist_choice == "Bernoulli":
    sample = np.random.binomial(n=1, p=0.5)
    st.markdown(f"""
    **Bernoulli: Coin Flip**  
    - Random variable: outcome of a coin toss  
    - Possible values: 0 (Tails), 1 (Heads) 
    """)
elif dist_choice == "Binomial":
    sample = np.random.binomial(n=10, p=0.4)
    st.markdown(f"""
    **Binomial: Successes in Trials**  
    - Random variable: number of successes in 10 trials (p = 0.4)  
    - Possible values: 0 to 10 
    """)
elif dist_choice == "Poisson":
    sample = np.random.poisson(lam=3)
    st.markdown(f"""
    **Poisson: Event Count in Interval**  
    - Random variable: events per unit interval (λ = 3)  
    - Possible values: 0, 1, 2, ...
    """)
elif dist_choice == "Categorical (Discrete)":
    outcomes = [1, 2, 3, 4, 5, 6]
    probs = [0.12, 0.12, 0.12, 0.12, 0.12, 0.4]
    sample = np.random.choice(outcomes, p=probs)
    st.markdown(f"""
    **Categorical: Biased Die Roll**  
    - Random variable: outcome from a biased 6-sided die  
    - Probabilities: P(6) = 0.4, P(others) = 0.12  
    - Possible values: 1 to 6
    """)
elif dist_choice == "Uniform (Continuous)":
    sample = np.random.uniform(0, 1)
    st.markdown(f"""
    **Uniform (Continuous)**  
    - Random variable: real number between 0 and 1  
    - Possible values: any value in [0, 1]
    """)
elif dist_choice == "Exponential":
    sample = np.random.exponential(scale=1.0)
    st.markdown(f"""
    **Exponential: Time Between Events**  
    - Random variable: time until next event (λ = 1)  
    - Possible values: any non-negative real number
    """)
elif dist_choice == "Normal (Gaussian)":
    sample = np.random.normal(loc=0, scale=1)
    st.markdown(f"""
    **Normal (Gaussian)**  
    - Random variable: standard normal distribution (μ = 0, σ = 1)  
    - Possible values: any real number
    """)

# CLT simulation
st.markdown("---")
st.markdown(f"### Central Limit Theorem Simulation (**{dist_choice}**)")

# Sliders for sample size and number of trials
sample_size = st.slider("Sample size per trial (n)", min_value=1, max_value=50, value=5)
num_trials = st.slider("Number of trials", min_value=1, max_value=2000, value=100, step=100)

# Generate sample means
sample_means = []
for _ in range(num_trials):
    if dist_choice == "Uniform (Discrete)":
        sample = np.random.randint(1, 7, size=sample_size)
    elif dist_choice == "Bernoulli":
        sample = np.random.binomial(n=1, p=0.5, size=sample_size)
    elif dist_choice == "Binomial":
        sample = np.random.binomial(n=10, p=0.4, size=sample_size)
    elif dist_choice == "Poisson":
        sample = np.random.poisson(lam=3, size=sample_size)
    elif dist_choice == "Categorical (Discrete)":
        outcomes = [1, 2, 3, 4, 5, 6]
        probs = [0.12, 0.12, 0.12, 0.12, 0.12, 0.4]
        sample = np.random.choice(outcomes, size=sample_size, p=probs)
    elif dist_choice == "Uniform (Continuous)":
        sample = np.random.uniform(0, 1, size=sample_size)
    elif dist_choice == "Exponential":
        sample = np.random.exponential(scale=1.0, size=sample_size)
    elif dist_choice == "Normal (Gaussian)":
        sample = np.random.normal(loc=0, scale=1, size=sample_size)

    sample_means.append(np.mean(sample))

# Convert and compute stats
sample_means = np.array(sample_means)
mean = np.mean(sample_means)
std = np.std(sample_means)

# Plotting
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.hist(sample_means, bins=40, density=True, color="skyblue", edgecolor="black", alpha=0.7)
x = np.linspace(min(sample_means), max(sample_means), 200)
ax.plot(x, norm.pdf(x, loc=mean, scale=std), color="darkred", lw=2, label="Normal Fit")
ax.axvline(mean, color="green", linestyle="--", linewidth=2, label=f"Mean = {mean:.2f}")
ax.set_title("Distribution of Sample Means")
ax.set_xlabel("Sample Mean (Average of n observations per trial)")
ax.set_ylabel("Density")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax.legend()

st.pyplot(fig)

# Summary stats
st.markdown(f"**Estimated Mean of Sample Means:** {mean:.4f} &nbsp;&nbsp; | &nbsp;&nbsp; **Standard Deviation:** {std:.4f}")

# Explanation
st.markdown("""
This plot shows how the **sample means**, computed from repeated sampling of the selected distribution,  
begin to form a **normal distribution** as sample size and number of trials increase.  
This is the core behavior predicted by the Central Limit Theorem.
""")