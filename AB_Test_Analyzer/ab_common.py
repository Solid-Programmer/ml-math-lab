import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show_confidence_interval(label, ci_lower, ci_upper):
    st.markdown("### Confidence Interval for Difference")
    st.markdown("The confidence interval gives a range of values that likely contain the true difference between groups.")
    st.markdown(f"**95% CI**: ({ci_lower:.4f}, {ci_upper:.4f})")

def show_effect_size(label, effect_size):
    st.markdown(f"### Effect Size ({label})")
    st.markdown("Effect size quantifies the magnitude of the difference, independent of sample size.")

    # Determine interpretation (based on Cohen's thresholds)
    if label == "Cohen's d":
        if abs(effect_size) < 0.2:
            interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            interpretation = "small"
        elif abs(effect_size) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
    elif label == "Cohen's h":
        if abs(effect_size) < 0.2:
            interpretation = "small"
        elif abs(effect_size) < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
    else:
        interpretation = "N/A"

    st.markdown(f"**{label}**: `{effect_size:.4f}` ({interpretation} effect)")

def show_power(power):
    st.markdown("### Statistical Power")
    st.markdown("Power is the probability of detecting a true effect (i.e., rejecting H₀ when it is false).")
    st.markdown(f"**Power of the test**: `{power:.4f}`")


def plot_bar_with_ci(groups, values, errors, ylabel, title, colors, p_val=None, alpha=0.05):
    fig, ax = plt.subplots()

    # Draw bars with error bars
    ax.bar(groups, values, yerr=errors, capsize=10, color=colors, edgecolor='black', alpha=0.8)

    # Add p-value annotation
    if p_val is not None:
        annotation = f"p-value = {p_val:.3e}" if p_val < 0.001 else f"p-value = {p_val:.3f}"
        if p_val < alpha:
            annotation += " → Significant"
            color = "green"
        else:
            annotation += " → Not Significant"
            color = "red"
        ax.text(0.5, max(values) + max(errors) * 1.2, annotation, ha='center', fontsize=12, color=color)

    # Final labels
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.markdown("### 📊 Visualization")
    st.pyplot(fig)


def plot_bar_simple(groups, values, ylabel, title, colors, yerr=None):
    fig, ax = plt.subplots()
    ax.bar(groups, values, color=colors, edgecolor='black')
    if yerr is not None:
        ax.errorbar(groups, values, yerr=yerr, fmt='o', color='black', capsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.markdown("### Visualization")
    st.pyplot(fig)

def show_summary_table(data, columns, title="Summary Table"):
    df = pd.DataFrame([data], columns=columns)
    st.markdown(f"### {title}")
    st.dataframe(df)
