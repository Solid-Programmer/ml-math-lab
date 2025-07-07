import numpy as np
from scipy import stats

def fit_normal(data):
    mu, sigma = stats.norm.fit(data)
    log_likelihood = np.sum(stats.norm.logpdf(data, mu, sigma))
    return {'mu': mu, 'sigma': sigma, 'log_likelihood': log_likelihood}

def fit_poisson(data):
    data = np.array(data)

    # Must be non-negative integers
    if np.any(data < 0) or not np.allclose(data, np.round(data)):
        return {'lambda': np.nan, 'log_likelihood': -np.inf}

    data = np.round(data).astype(int)
    lam = np.mean(data)
    log_likelihood = np.sum(stats.poisson.logpmf(data, lam))
    return {'lambda': lam, 'log_likelihood': log_likelihood}

def fit_exponential(data):
    data = np.array(data)

    # Must be strictly positive
    if np.any(data <= 0):
        return {'lambda': np.nan, 'log_likelihood': -np.inf}

    loc, scale = stats.expon.fit(data, floc=0)
    lam = 1 / scale
    log_likelihood = np.sum(stats.expon.logpdf(data, loc=0, scale=scale))
    return {'lambda': lam, 'log_likelihood': log_likelihood}
