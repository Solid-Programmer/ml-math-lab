import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from AB_Test_Analyzer.ab_math import z_test_proportions, t_test_means, confidence_interval, cohens_h, cohens_d, power_z, power_t
from AB_Test_Analyzer.ab_common import show_confidence_interval, show_effect_size, show_power, plot_bar_with_ci

if "results_ready" not in st.session_state:
    st.session_state.results_ready = False


st.set_page_config(page_title="A/B Test Analyzer", layout="centered")

# Title
st.title("🔍 A/B Test Analyzer")

# Introduction
st.markdown("""
## Understanding A/B Testing

A/B testing is a simple and effective method for comparing two versions of something—such as a website, app feature, or marketing campaign—to determine which one performs better.

These two versions are randomly shown to users:
- **Group A**: Sees the original version (often called the control)
- **Group B**: Sees the new or modified version (the variation)

We use **hypothesis testing** to make statistically sound decisions:

### Hypotheses
- **Null Hypothesis (H₀)**: There is **no difference** between Group A and Group B.  
  Example: "Performance of A = Performance of B"
- **Alternative Hypothesis (H₁)**: There **is a difference**.  
  Example: "Performance of A ≠ Performance of B"
            
### Three Key Factors in Hypothesis Testing

1. **Significance Level (α)**: This is the threshold we set before the test — commonly 0.05 — which defines how much chance we're willing to accept of making a wrong decision (Type I error). If p-value < α, we reject the null hypothesis.

2. **Test Statistic (Z or T)**: A value computed from sample data that measures how far your observed result is from what the null hypothesis expects. For large samples or known variance, we use **Z-statistic**. For smaller samples with estimated variance, we use **T-statistic**.

3. **p-value**: The probability of getting your observed result (or something more extreme) **if the null hypothesis were true**. A smaller p-value indicates stronger evidence against H₀.
  Example: "Performance of A ≠ Performance of B"

""")

# Questions that can be answered using A/B Testing
st.markdown("""
---

### Questions that Can Be Answered Using A/B Testing

A/B Testing helps answer important data-driven questions like:

- **Does the new landing page convert more users than the current one?**
- **Does changing the button color improve the click-through rate?**
- **Does a new feature increase average session time or engagement?**
- **Is the new pricing model resulting in higher revenue per user?**

These are real-world use cases where measuring **statistical significance** is critical before rolling out changes.
""")

st.markdown("## Enter Your Experiment Data")

# Test type selection
test_type = st.radio("What type of metric are you testing?", ["Conversion Rate / Proportion", "Average Value / Mean"])

# Reset session state if test type changes
def reset_results():
    st.session_state.pop('ab_result', None)
    st.session_state.pop('mean_result', None)
    st.session_state.advanced_ready = False
    st.session_state.results_ready = False

if 'last_test_type' not in st.session_state:
    st.session_state.last_test_type = test_type
elif st.session_state.last_test_type != test_type:
    reset_results()
    st.session_state.last_test_type = test_type

# Significance level input
alpha = st.number_input("Significance Level (α)", min_value=0.001, max_value=0.2, value=0.05, step=0.01)

st.markdown("### 📥 Group A and Group B Inputs")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Group A**")
    n_a = st.number_input("Sample Size (nₐ)", min_value=1, value=1000, key="n_a")
    if test_type == "Conversion Rate / Proportion":
        x_a = st.number_input("Number of Successes (xₐ)", min_value=0, max_value=n_a, value=230, key="x_a")
    else:
        mean_a = st.number_input("Mean (μₐ)", value=4.3, key="mean_a")
        std_a = st.number_input("Standard Deviation (σₐ)", value=1.2, min_value=0.01, key="std_a")

with col2:
    st.markdown("**Group B**")
    n_b = st.number_input("Sample Size (n_b)", min_value=1, value=950, key="n_b")
    if test_type == "Conversion Rate / Proportion":
        x_b = st.number_input("Number of Successes (x_b)", min_value=0, max_value=n_b, value=270, key="x_b")
    else:
        mean_b = st.number_input("Mean (μ_b)", value=4.7, key="mean_b")
        std_b = st.number_input("Standard Deviation (σ_b)", value=1.0, min_value=0.01, key="std_b")


# Calculate and display results
if st.button("Analyze A/B Test"):
    if test_type == "Conversion Rate / Proportion":
        st.info("""
        **Z-test for Proportions** was used.  
        - Suitable for binary outcomes (e.g., success/failure, conversion, click/no-click).  
        - Assumes large enough sample sizes such that np and n(1-p) ≥ 10.  
        - Computes whether the difference in proportions is statistically significant.
        """)
        p_a, p_b, se, z_stat, p_val = z_test_proportions(x_a, n_a, x_b, n_b)
        st.latex(r"Z = \frac{\hat{p}_A - \hat{p}_B}{\sqrt{p(1 - p)(\frac{1}{n_A} + \frac{1}{n_B})}}")
        st.markdown(f"**Z-statistic** = `{z_stat:.4f}`  \n**p-value** = `{p_val:.4f}`")
        st.session_state.ab_result = {
            'p_a': p_a, 'p_b': p_b, 'se': se, 'z_stat': z_stat, 'alpha': alpha, 'p_val': p_val
        }
        st.session_state.advanced_ready = True
        # --- Significance message and summary table for proportions ---
        if p_val < alpha:
            st.success("✅ Reject the null hypothesis: The difference is statistically significant.")
        else:
            st.info("ℹ️ Fail to reject the null hypothesis: The difference is not statistically significant.")
        data = {
            "Group": ["A", "B"],
            "Sample Size": [n_a, n_b],
            "Successes": [x_a, x_b],
            "Conversion Rate": [p_a, p_b],
        }
        df_summary = pd.DataFrame(data)
        st.markdown("### 📋 Summary Table")
        st.dataframe(df_summary, use_container_width=True)
    else:
        st.info("""
        **Welch’s T-test for Means** was used.  
        - Suitable for continuous metrics like session time, rating, or spend.  
        - Makes no assumption about equal variances between groups.  
        - Robust for both small and large sample sizes.  
        - Ideal when population standard deviations are unknown (which is common in A/B testing).
        """)
        se, t_stat, df, p_val = t_test_means(mean_a, std_a, n_a, mean_b, std_b, n_b)
        st.latex(r"T = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}")
        st.markdown(f"**T-statistic** = `{t_stat:.4f}`  \n**Degrees of freedom** = `{df:.2f}`  \n**p-value** = `{p_val:.4f}`")
        st.session_state.mean_result = {
            'mean_a': mean_a, 'mean_b': mean_b, 'std_a': std_a, 'std_b': std_b, 'se': se, 't_stat': t_stat, 'df': df, 'alpha': alpha, 'n_a': n_a, 'n_b': n_b, 'p_val': p_val
        }
        st.session_state.advanced_ready = True
        # --- Significance message and summary table for means ---
        if p_val < alpha:
            st.success("✅ Reject the null hypothesis: The difference is statistically significant.")
        else:
            st.info("ℹ️ Fail to reject the null hypothesis: The difference is not statistically significant.")
        data = {
            "Group": ["A", "B"],
            "Sample Size": [n_a, n_b],
            "Mean": [mean_a, mean_b],
            "Std Dev": [std_a, std_b],
        }
        df_summary = pd.DataFrame(data)
        st.markdown("### 📋 Summary Table")
        st.dataframe(df_summary, use_container_width=True)
    st.session_state.results_ready = True

# Advanced metrics for Conversion Rate
if (
    st.session_state.get('advanced_ready', False)
    and test_type == "Conversion Rate / Proportion"
    and 'ab_result' in st.session_state
):
    if st.button("Show Additional Insights"):
        ab = st.session_state.ab_result
        p_a = ab['p_a']
        p_b = ab['p_b']
        se = ab['se']
        z_stat = ab['z_stat']
        alpha = ab['alpha']
        diff = p_a - p_b
        ci_lower, ci_upper, z_crit = confidence_interval(diff, se, alpha, 'z')
        effect_size = cohens_h(p_b, p_a)
        power = power_z(z_stat, z_crit)
        show_confidence_interval("Proportion Difference", ci_lower, ci_upper)
        show_effect_size("Cohen's h", effect_size)
        show_power(power)
        plot_bar_with_ci(
            groups=['Group A', 'Group B'],
            values=[p_a, p_b],
            errors=[z_crit*se]*2,
            ylabel="Conversion Rate",
            title="Conversion Rate with 95% Confidence Interval",
            colors=['skyblue', 'lightgreen'],
            p_val=ab['p_val'],
            alpha=alpha
        )

# Advanced metrics for Mean Comparison
if (
    st.session_state.get('advanced_ready', False)
    and test_type == "Average Value / Mean"
    and 'mean_result' in st.session_state
):
    if st.button("Show Additional Insights for Means"):
        mean = st.session_state.mean_result
        mean_a = mean['mean_a']
        mean_b = mean['mean_b']
        std_a = mean['std_a']
        std_b = mean['std_b']
        n_a = mean['n_a']
        n_b = mean['n_b']
        se = mean['se']
        t_stat = mean['t_stat']
        df = mean['df']
        alpha = mean['alpha']
        diff = mean_a - mean_b
        ci_lower, ci_upper, t_crit = confidence_interval(diff, se, alpha, 't', df)
        effect_size = cohens_d(mean_a, mean_b, std_a, std_b, n_a, n_b)
        power = power_t(t_stat, t_crit)
        show_confidence_interval("Mean Difference", ci_lower, ci_upper)
        show_effect_size("Cohen's d", effect_size)
        show_power(power)
        errors = [t_crit * std_a / np.sqrt(n_a), t_crit * std_b / np.sqrt(n_b)]
        plot_bar_with_ci(
            groups=['Group A', 'Group B'],
            values=[mean_a, mean_b],
            errors=errors,
            ylabel='Mean Value',
            title='Group Means with 95% Confidence Interval',
            colors=['orange', 'lightblue'],
            p_val=mean['p_val'],
            alpha=alpha
        )
