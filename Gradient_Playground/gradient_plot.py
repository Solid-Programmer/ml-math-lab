import matplotlib.pyplot as plt
import numpy as np

def plot_line_fit(months, bills_in_inr, m, b, mae_val):
    x_dense = np.linspace(1, 12, 200)
    y_dense = m * x_dense + b
    fig, ax = plt.subplots()
    ax.scatter(months, bills_in_inr, color='blue', label='Actual Bill')
    ax.plot(x_dense, y_dense, color='red', label=f'Line: y = {m:.2f}x + {b:.2f}')
    ax.set_xlabel("Month")
    ax.set_ylabel("Electricity Bill (K INR)")
    ax.set_title(f"Line Fit with MAE = {mae_val:.2f}")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(1, 12)
    y_margin = (max(bills_in_inr) - min(bills_in_inr)) * 0.1
    ax.set_ylim(min(bills_in_inr) - y_margin, max(bills_in_inr) + y_margin)
    return fig

def plot_final_fit(months, bills_in_inr, final_m, final_b, final_mae):
    x_dense = np.linspace(1, 12, 200)
    y_dense = final_m * x_dense + final_b
    fig_final, ax_final = plt.subplots()
    ax_final.scatter(months, bills_in_inr, color='blue', label='Actual Data')
    ax_final.plot(x_dense, y_dense, color='green', label=f'Fitted Line: y = {final_m:.2f}x + {final_b:.2f}')
    ax_final.set_xlabel("Month")
    ax_final.set_ylabel("Electricity Bill (K INR)")
    ax_final.set_title(f"Final Fitted Line (MAE = {final_mae:.2f})")
    ax_final.legend()
    ax_final.grid(True)
    ax_final.set_xlim(1, 12)
    y_margin = (max(bills_in_inr) - min(bills_in_inr)) * 0.1
    ax_final.set_ylim(min(bills_in_inr) - y_margin, max(bills_in_inr) + y_margin)
    return fig_final

def plot_mae_contour(M, B, Z, path, final_m, final_b, final_mae):
    fig_contour, ax_c = plt.subplots(figsize=(8, 6))
    contour = ax_c.contourf(M, B, Z, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax_c)
    cbar.set_label("MAE")
    # Optimizer Path
    ax_c.plot(path[:, 0], path[:, 1], 'o-', color='red', markersize=4, linewidth=2, label='Optimizer Path')
    ax_c.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start')
    ax_c.plot(path[-1, 0], path[-1, 1], 'bo', markersize=8, label='End')
    # Annotate final MAE
    ax_c.annotate(f"Final MAE = {final_mae:.2f}", xy=(final_m, final_b), xytext=(final_m + 0.2, final_b + 0.5),
                  arrowprops=dict(facecolor='black', shrink=0.05),
                  fontsize=10, backgroundcolor='white')
    ax_c.set_xlabel("Slope (m)")
    ax_c.set_ylabel("Intercept (b)")
    ax_c.set_title("MAE Contour with Optimizer Path")
    ax_c.legend()
    ax_c.grid(True)
    return fig_contour

def plot_optimizer_path(M, B, Z, path, optimizer, step, mae):
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(M, B, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax).set_label("MAE")
    trail = np.array(path[:step + 1])
    ax.plot(trail[:, 0], trail[:, 1], 'o-', color='red', label="Optimizer Path")
    ax.plot(trail[0, 0], trail[0, 1], 'go', label="Start")
    ax.plot(trail[-1, 0], trail[-1, 1], 'bo', label="Current")
    ax.set_xlabel("Slope (m)")
    ax.set_ylabel("Intercept (b)")
    ax.set_title(f"{optimizer} Steps (Step {step}) - MAE = {mae:.3f}")
    ax.legend()
    ax.grid(True)
    return fig
