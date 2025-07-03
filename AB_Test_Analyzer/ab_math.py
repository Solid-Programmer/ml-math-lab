import numpy as np
import scipy.stats as stats

def z_test_proportions(x_a, n_a, x_b, n_b):
    p_a = x_a / n_a
    p_b = x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z_stat = (p_a - p_b) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return p_a, p_b, se, z_stat, p_val

def t_test_means(mean_a, std_a, n_a, mean_b, std_b, n_b):
    se = np.sqrt((std_a**2)/n_a + (std_b**2)/n_b)
    t_stat = (mean_a - mean_b) / se
    df = ((std_a**2)/n_a + (std_b**2)/n_b)**2 / (
        ((std_a**2)/n_a)**2 / (n_a - 1) + ((std_b**2)/n_b)**2 / (n_b - 1)
    )
    p_val = 2 * stats.t.sf(abs(t_stat), df)
    return se, t_stat, df, p_val

def confidence_interval(diff, se, alpha, stat_type, df=None):
    if stat_type == 'z':
        crit = stats.norm.ppf(1 - alpha / 2)
    else:
        crit = stats.t.ppf(1 - alpha / 2, df)
    return (diff - crit * se, diff + crit * se, crit)

def cohens_h(p1, p2):
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def cohens_d(mean_a, mean_b, std_a, std_b, n_a, n_b):
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    return (mean_b - mean_a) / pooled_std

def power_z(z_stat, z_crit):
    return stats.norm.cdf(abs(z_stat) - z_crit)

def power_t(t_stat, t_crit):
    return stats.norm.cdf(abs(t_stat) - t_crit)
