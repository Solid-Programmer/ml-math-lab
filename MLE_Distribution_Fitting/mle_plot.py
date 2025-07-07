import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_distribution_fit(data, dist_name, params, bins=30):
    """
    Plots histogram of the data and overlays the fitted distribution curve.

    Args:
        data: array-like, raw data
        dist_name: 'normal' | 'poisson' | 'exponential'
        params: dict containing MLE parameters
        bins: number of bins for histogram
    """

    data = np.asarray(data)
    
    # --- Basic validation ---
    if len(data) == 0:
        raise ValueError("Cannot plot: data is empty.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Cannot plot: data contains NaNs or infinite values.")

    data_min, data_max = np.min(data), np.max(data)
    if data_min == data_max:
        data_min -= 1
        data_max += 1

    # --- Setup figure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bin_edges, _ = ax.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', label='Data')

    x = np.linspace(data_min, data_max, 1000)

    if dist_name == 'normal':
        mu, sigma = params['mu'], params['sigma']
        pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, pdf, 'r-', lw=2, label=f'Normal Fit\nμ={mu:.2f}, σ={sigma:.2f}')

    elif dist_name == 'exponential':
        lam = params['lambda']
        scale = 1 / lam
        pdf = stats.expon.pdf(x, loc=0, scale=scale)
        ax.plot(x, pdf, 'g-', lw=2, label=f'Exponential Fit\nλ={lam:.2f}')

    elif dist_name == 'poisson':
        lam = params['lambda']
        x_discrete = np.arange(np.floor(data_min), np.ceil(data_max) + 1).astype(int)
        pmf = stats.poisson.pmf(x_discrete, lam)
        ax.vlines(x_discrete, 0, pmf, colors='orange', lw=2, label=f'Poisson Fit\nλ={lam:.2f}')

    else:
        raise ValueError("Unsupported distribution")

    ax.set_title(f"MLE Fit: {dist_name.capitalize()} Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig
