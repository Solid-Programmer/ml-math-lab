import numpy as np

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def mae_gradient(y_true, y_pred, x):
    """Gradient of MAE with respect to m and b for y = m*x + b"""
    n = len(y_true)
    errors = y_true - y_pred
    signs = np.sign(errors)
    dL_dm = -np.sum(x * signs) / n
    dL_db = -np.sum(signs) / n
    return dL_dm, dL_db

def brute_force_mae_grid(y_true, x, m_range, b_range, m_points=50, b_points=50):
    """Compute MAE for a grid of m and b values."""
    m_vals = np.linspace(*m_range, m_points)
    b_vals = np.linspace(*b_range, b_points)
    M, B = np.meshgrid(m_vals, b_vals)
    Z = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            preds = M[i, j] * x + B[i, j]
            Z[i, j] = mae(y_true, preds)
    return M, B, Z

def gradient_descent_mae(y_true, x, m_init, b_init, eta=0.01, epochs=1000, patience=20):
    """Gradient Descent for MAE with early stopping using patience. Returns full history and best values."""
    m, b = m_init, b_init
    best_mae = float('inf')
    best_epoch = 0
    best_params = (m, b)
    wait = 0
    history = []

    for epoch in range(epochs + 1):
        preds = m * x + b
        errors = y_true - preds
        current_mae = np.mean(np.abs(errors))
        history.append({'Epoch': epoch, 'm': round(m, 4), 'b': round(b, 4), 'MAE': round(current_mae, 4)})

        if current_mae + 1e-8 < best_mae:
            best_mae = current_mae
            best_epoch = epoch
            best_params = (m, b)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        signs = np.sign(errors)
        dL_dm = -np.sum(x * signs) / len(x)
        dL_db = -np.sum(signs) / len(x)

        m -= eta * dL_dm
        b -= eta * dL_db

    return history, best_params, best_mae, best_epoch


def optimizer_path_mae(y_true, x, m_init, b_init, eta, beta, beta2, epochs, optimizer):
    """Compute optimizer path for different optimizers (SGD, Momentum, Adam) on MAE."""
    m, b = m_init, b_init
    n = len(y_true)
    vm = vb = sm = sb = 0
    eps = 1e-8
    path = [(m, b)]
    for t in range(1, epochs + 1):
        preds = m * x + b
        dL_dm, dL_db = mae_gradient(y_true, preds, x)
        if optimizer == "SGD":
            m -= eta * dL_dm
            b -= eta * dL_db
        elif optimizer == "Momentum":
            vm = beta * vm + dL_dm
            vb = beta * vb + dL_db
            m -= eta * vm
            b -= eta * vb
        elif optimizer == "Adam":
            vm = beta * vm + (1 - beta) * dL_dm
            vb = beta * vb + (1 - beta) * dL_db
            sm = beta2 * sm + (1 - beta2) * (dL_dm ** 2)
            sb = beta2 * sb + (1 - beta2) * (dL_db ** 2)
            vm_hat = vm / (1 - beta ** t)
            vb_hat = vb / (1 - beta ** t)
            sm_hat = sm / (1 - beta2 ** t)
            sb_hat = sb / (1 - beta2 ** t)
            m -= eta * vm_hat / (np.sqrt(sm_hat) + eps)
            b -= eta * vb_hat / (np.sqrt(sb_hat) + eps)
        path.append((m, b))
    return path
