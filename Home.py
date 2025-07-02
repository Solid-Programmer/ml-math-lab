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
</div>''', unsafe_allow_html=True)

st.markdown('<div class="instructions"><b>Instructions:</b><ul><li>Select an app from the sidebar to begin.</li><li>Each app includes concise explanations, interactive controls, and professional visualizations.</li><li>Adjust parameters to see real-time mathematical responses.</li></ul></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">For questions or feedback, contact the project maintainer at <a href="mailto:12bce1006@gmail.com">12bce1006@gmail.com</a>.<br>ML Math Lab &copy; 2025</div>', unsafe_allow_html=True)