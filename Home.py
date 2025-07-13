import streamlit as st

st.set_page_config(page_title="ML Math Lab", layout="wide")

# Custom CSS for professional look and full-width layout
st.markdown("""
<style>
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1a237e;
    margin-bottom: 0.5rem;
    letter-spacing: 1px;
}
.subtitle {
    font-size: 1.3rem;
    color: #374151;
    margin-bottom: 1.5rem;
}
.card-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}
.app-card {
    background: #f5f7fa;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    min-width: 300px;
    flex: 1 1 0;
    box-shadow: 0 2px 8px rgba(30, 41, 59, 0.07);
    border-left: 5px solid #1976d2;
    margin-bottom: 0;
    max-width: 32%;
}
.instructions {
    background: #e3f2fd;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-top: 1.5rem;
    font-size: 1.08rem;
    color: #1a237e;
    border-left: 4px solid #1976d2;
}
.footer {
    color: #607d8b;
    font-size: 0.98rem;
    margin-top: 2.5rem;
    text-align: right;
}
@media (max-width: 1200px) {
    .app-card { max-width: 48%; }
}
@media (max-width: 800px) {
    .card-row { flex-direction: column; }
    .app-card { max-width: 100%; }
}
section.main > div:first-child {
    padding-top: 0rem !important;
    margin-top: 0 !important;
}
            .block-container {
    padding-top: 1rem !important;
    margin-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">ML Math Lab: Interactive Math Visualizations</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">A professional suite for exploring and visualizing foundational mathematics in Artificial Intelligence and Machine Learning</div>', unsafe_allow_html=True)

st.markdown('''<div class="card-row">
<div class="app-card"><b>PCA Visualizer</b><br>Explore Principal Component Analysis interactively. Visualize dimensionality reduction and variance explained.</div>
<div class="app-card"><b>Gradient Descent Playground</b><br>Visualize and experiment with gradient descent, loss surfaces, and optimization paths.</div>
<div class="app-card"><b>Entropy and Cross-Entropy Explorer</b><br>Understand entropy, cross-entropy, and KL divergence with interactive probability distributions.</div>
</div>
<div class="card-row">
<div class="app-card"><b>Central Limit Theorem Simulator</b><br>See how the distribution of sample means approaches normality for various distributions.</div>
<div class="app-card"><b>Naive Bayes Email Classifier</b><br>Classify emails as spam or ham using a transparent, interactive Naive Bayes model. Explore word contributions and model metrics.</div>
<div class="app-card"><b>A/B Test Analyzer</b><br>Compare conversion rates or averages with Z and T tests. Visualize confidence intervals, p-values, and statistical power for clear decision-making.</div>
</div>
<div class="card-row">
<div class="app-card"><b>Similarity Search with Cosine Distance</b><br>Find and compare similar items using cosine similarity. Explore vector spaces and high-dimensional geometry.</div>
<div class="app-card"><b>MLE Distribution Fitting Tool</b><br>Fit distributions (Normal, Poisson, Exponential) to data using Maximum Likelihood Estimation and visualize the best-fit curve.</div>
<div class="app-card"><b>Bias-Variance Tradeoff Demo</b><br>Explore model complexity vs generalization by plotting training and test errors across polynomial degrees.</div>
</div>
<div class="card-row">
<div class="app-card"><b>Visual Loss Surface for Logistic Regression</b><br>Visualize the 3D loss surface for logistic regression and animate optimizer paths across the surface.</div>
<div class="app-card"><b>Interactive Bayes' Theorem Simulator</b><br>Adjust priors, sensitivity, and specificity in real-time to simulate medical test outcomes or spam filters.</div>
<div class="app-card"><b>Distance Metric Visualizer</b><br>Compare distance metrics (Euclidean, Manhattan, Cosine, Minkowski) and see how decision boundaries change with each.</div>
</div>
''', unsafe_allow_html=True)

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


