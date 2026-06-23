# ML Math Lab

An interactive Streamlit portfolio project that turns core machine-learning and statistical concepts into visual, hands-on learning experiences. Rather than treating models as black boxes, each module exposes the underlying mathematics, calculations, parameters, and visualizations.

The project demonstrates practical implementation skills across numerical computing, statistics, machine learning, natural-language processing, data visualization, and interactive application development.

**Live project:** [samkhai.com/math-for-ml](https://samkhai.com/math-for-ml)

## Highlights

- Build and explain ML concepts from their mathematical foundations.
- Create interactive Streamlit interfaces with adjustable parameters and immediate visual feedback.
- Implement numerical workflows with NumPy, SciPy, pandas, and scikit-learn.
- Visualize results with Matplotlib and Plotly.
- Package reusable topic-specific calculation and plotting modules behind a multi-page application.

## Interactive Modules

| Module | Core skills demonstrated |
| --- | --- |
| PCA Visualizer | Standardization, covariance matrices, eigen-decomposition, explained variance, component loadings, dimensionality reduction |
| Gradient Descent Playground | MAE loss, gradients, optimization paths, SGD, Momentum, Adam, contour plots |
| Entropy Explorer | Entropy, cross-entropy, KL divergence, probability distributions |
| Central Limit Theorem Simulator | Sampling distributions, simulations, normal approximation |
| Naive Bayes Email Classifier | Text preprocessing, tokenization, stop-word removal, Laplace smoothing, log-probability classification, evaluation metrics |
| A/B Test Analyzer | Two-proportion z-tests, Welch's t-test, confidence intervals, effect sizes, statistical power |
| Similarity Search | TF-IDF vectorization, cosine similarity, query ranking, basic spell correction |
| MLE Distribution Fitting | Maximum likelihood estimation for Normal, Poisson, and Exponential distributions |
| Bias-Variance Tradeoff | Polynomial regression, train/test error, underfitting and overfitting |
| Logistic Loss Landscape | Sigmoid activation, binary cross-entropy, parameter-space loss surfaces |
| Bayes' Theorem Simulator | Priors, likelihoods, sensitivity, specificity, posterior probabilities |
| Distance Metric Visualizer | Euclidean, Manhattan, Cosine, and Minkowski distance; norm geometry; k-NN boundaries |

## Technology Stack

- **Application framework:** Streamlit
- **Numerical and data tools:** NumPy, pandas, SciPy
- **Machine learning:** scikit-learn
- **Visualization:** Matplotlib, Plotly
- **NLP:** NLTK, SymSpellPy

## Run Locally

### Prerequisites

- Python 3.11 or newer
- pip

### Installation

```bash
git clone <your-repository-url>
cd ml-math-lab

python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and launch the application:

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Open the local URL Streamlit prints in the terminal, usually `http://localhost:8501`.

## Project Structure

```text
ml-math-lab/
├── Home.py                         # Streamlit landing page
├── pages/                          # Multi-page interactive lessons
├── PCA_Visualizer/                 # PCA mathematics and plotting
├── Gradient_Playground/             # MAE optimization logic and visualizations
├── Entropy_Explorer/                # Information-theory calculations and plots
├── CTL_Simulator/                   # CLT simulation module
├── Naive_Bayes_Email_Classifier/    # Text classifier, model loader, and exporter
├── AB_Test_Analyzer/                # Hypothesis-test calculations and display helpers
├── Cosine_Similarity/               # TF-IDF store builder and PDF parsing helpers
├── MLE_Distribution_Fitting/        # Distribution fitting and chart utilities
├── Logistic_Loss_Landscape/         # Logistic model and loss-surface utilities
├── Distance_Metric_Visualizer/      # Distance functions and geometry visualizations
└── requirements.txt
```

## Included Model Assets

The repository includes two prebuilt assets so the relevant demos can run without retraining:

- `Naive_Bayes_Email_Classifier/model/naive_bayes_model.pkl` — saved Naive Bayes model and evaluation data.
- `Cosine_Similarity/tfidf_vector_store.pkl` — saved TF-IDF vector store for the similarity-search demo.

The scripts used to rebuild these artifacts are included. Rebuilding them requires their original source datasets: the Enron spam CSV for the classifier and the source PDF/parsed sections for similarity search.

## Portfolio Focus

This project is designed to show that I can move from mathematical intuition to working software: derive and implement algorithms, validate them visually, package them into reusable modules, and present them through a clear interactive interface.

## Future Improvements

- Add automated tests for numerical functions and edge cases.
- Pin dependency versions for reproducible environments.
- Add screenshots or a deployed demo link.
- Add downloadable experiment reports and user-provided dataset support.

## Author

Built by **Samar Khan** and hosted for anyone to visualize, explore, and learn: [samkhai.com/math-for-ml](https://samkhai.com/math-for-ml).
