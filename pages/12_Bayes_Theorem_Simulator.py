import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Bayes Theorem Simulator", layout="centered")
st.markdown("### Bayes Theorem Simulator")  # Replaces st.title()

st.markdown("#### What is Bayes' Theorem?")
st.markdown("""
Bayes' Theorem is a fundamental principle in probability theory that describes how to **update our belief** about an event based on new evidence.

It answers questions like:
> *"Given a positive test result, what is the probability that the person actually has the disease?"*
""")

st.markdown("#### Bayes' Theorem Formula")
st.latex(r"""
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
""")

st.markdown("""
Where:
- **P(A|B)**: Posterior probability → probability of event A given evidence B  
- **P(B|A)**: Likelihood → probability of evidence B assuming A is true  
- **P(A)**: Prior → initial belief about event A  
- **P(B)**: Marginal probability → overall probability of evidence B  
""")

st.markdown("#### Key Terminologies")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Prior (P(A))**")
    st.markdown("Initial belief before seeing evidence.")

    st.markdown("**Likelihood (P(B|A))**")
    st.markdown("How likely is the evidence if A is true?")

with col2:
    st.markdown("**Marginal (P(B))**")
    st.markdown("Overall probability of observing B.")

    st.markdown("**Posterior (P(A|B))**")
    st.markdown("Updated belief after seeing the evidence.")

st.markdown("---")
st.markdown("#### How is it different from Conditional Probability?")
st.markdown("""
Conditional probability **P(A|B)** tells us the probability of A given B — but doesn’t explain *how* to compute it when only indirect information is available.

Bayes' Theorem provides a structured way to compute this using the **prior knowledge** and the **likelihood** of evidence.

For example:
- Conditional probability: "What's P(Cancer | Positive test)?"
- Bayes' Theorem: Uses test sensitivity, specificity, and prevalence to **calculate** this.
""")

st.info("Bayes' Theorem is not just about probability — it's about updating beliefs in light of new evidence.")

# --- Section: Interactive Demo Intro ---
st.markdown("---")
st.markdown("#### Understanding Bayes' Theorem Through an Interactive Demo")
st.markdown("""
Now that we understand the formula and core ideas, let's explore how Bayes' Theorem works **in practice** using a real-world example.

We'll walk through a **breast cancer screening** scenario and interactively adjust parameters like:
- **Prevalence** (how common the disease is),
- **Sensitivity** (how well the test detects positives), and
- **Specificity** (how well the test avoids false alarms).

This will help you build intuition for how prior beliefs and test accuracy affect the final **posterior probability**.
""")

st.markdown("**Example: Breast Cancer Screening**")

st.markdown("""
Imagine 1,000 women go through a routine breast cancer screening test.

- **Prevalence**: 1% (so 10 women out of 1000 actually have cancer)  
- **Sensitivity**: 90% (the test correctly detects 9 out of 10 cancer cases)  
- **Specificity**: 91% (the test correctly gives a negative result to 91% of women without cancer)

Results:  
- 9 women with cancer test positive (**True Positives**)  
- 1 woman with cancer tests negative (**False Negative**)  
- 89 women without cancer test positive (**False Positives**)  
- 901 women without cancer test negative (**True Negatives**)

So if a woman tests positive, she could be one of the 9 **true positives** or one of the 89 **false positives**.
""")

st.latex(r"""
P(\text{Cancer} \mid \text{Positive}) = \frac{9}{9 + 89} \approx 9.2\%
""")

st.info("Even though the test is 90% sensitive and 91% specific, a person who tests positive has only about a 9% chance of actually having cancer.")

st.markdown("""
This low probability might feel surprising, but it's because:
- The disease is **rare** (only 1% of the population has it),
- And **false positives** (89 cases) greatly outnumber true positives (9 cases).

So among all positive results, most are actually false alarms.

This shows the **power of Bayes' Theorem** — it helps us combine test accuracy with real-world prevalence to make better sense of what a positive result really means.
""")

st.info("An accurate test doesn't always mean a predictive test. Let's explore why using a visual simulator.")

st.markdown("---")
st.markdown("### Interactive Simulator")

st.markdown("""
Use the sliders below to adjust:
- **Prevalence**: How common the disease is in the population
- **Sensitivity**: How well the test detects the disease (true positives)
- **Specificity**: How well the test rules out healthy cases (true negatives)
""")

# --- Slider Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    prevalence = st.slider("Prevalence (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

with col2:
    sensitivity = st.slider("Sensitivity (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.5)

with col3:
    specificity = st.slider("Specificity (%)", min_value=0.0, max_value=100.0, value=91.0, step=0.5)

# --- Calculations ---
total_population = 1000
diseased = int(total_population * (prevalence / 100))
healthy = total_population - diseased

tp = int(diseased * (sensitivity / 100))        # True positives
fn = diseased - tp                              # False negatives

tn = int(healthy * (specificity / 100))         # True negatives
fp = healthy - tn                               # False positives

# Posterior calculation
total_positives = tp + fp
posterior_prob = (tp / total_positives) * 100 if total_positives > 0 else 0.0

# Bayes Factor (sensitivity / false positive rate)
false_positive_rate = 1 - (specificity / 100)
bayes_factor = (sensitivity / 100) / false_positive_rate if false_positive_rate > 0 else float('inf')

# --- Output Display ---
st.markdown("### Simulation Results for 1000 People")
st.markdown(f"""
- **True Positives**: {tp}  
- **False Positives**: {fp}  
- **True Negatives**: {tn}  
- **False Negatives**: {fn}  
- **Posterior Probability (P(Disease | Positive Test))**: {posterior_prob:.2f}%  
- **Bayes Factor**: {bayes_factor:.2f}
""")

st.info("**Note:** Bayes Factor tells us how much a positive test result increases the odds of having the disease.")

# --- Optional: Toggle to analyze negative test results ---
with st.expander("What if the test result is Negative?"):
    total_negatives = fn + tn
    if total_negatives == 0:
        st.warning("No one tested negative. Try changing the inputs.")
    else:
        posterior_if_negative = (fn / total_negatives) * 100  # P(Disease | Negative)
        st.markdown(f"""
        Out of **{total_negatives}** people who tested negative:
        - **{fn}** actually have the disease (false negatives)
        - **{tn}** are truly healthy

        So if someone tests **negative**, the chance they still have the disease is **{posterior_if_negative:.2f}%**.
        """)

        st.info("This shows the importance of test sensitivity — a less sensitive test can miss actual cases (false negatives).")

# --- Visualization in Two Columns ---
col_pie, col_bar = st.columns(2)

# --- Pie Chart (Positive Test Breakdown) ---
with col_pie:
    st.markdown("**Positive Test Result Breakdown**")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # smaller size
    labels = ['True Positives', 'False Positives']
    sizes = [tp, fp]
    colors = ['#4CAF50', '#F44336']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# --- Bar Chart (All Outcomes) ---
with col_bar:
    st.markdown("**All Test Outcomes**")
    fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))  # smaller width, fixed height
    categories = ['TP', 'FP', 'TN', 'FN']
    counts = [tp, fp, tn, fn]
    colors_bar = ['#4CAF50', '#F44336', '#2196F3', '#FFC107']
    bars = ax2.bar(categories, counts, color=colors_bar)
    ax2.set_ylabel("People")
    ax2.set_ylim(0, max(counts) + 50)

    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, height + 5, f'{int(height)}', ha='center', va='bottom')

    st.pyplot(fig2)


# --- Conclusion Summary ---
st.markdown("### Summary in Plain Language")

if total_positives > 0:
    st.success(f"""
**{tp} out of {total_positives}** people who tested **positive** actually have the disease.

That's about **{posterior_prob:.1f}%**
""")
else:
    st.info("No one tested positive, so there's no way to estimate disease chance from positive results.")



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