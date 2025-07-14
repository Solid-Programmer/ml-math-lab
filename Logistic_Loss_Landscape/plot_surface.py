import plotly.graph_objects as go
import numpy as np

def plot_loss_surface_3d(W1, W2, Z, min_point=None):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Surface(z=Z, x=W1, y=W2, colorscale='Viridis')])
    
    if min_point:
        w1, w2, z = min_point
        fig.add_trace(go.Scatter3d(
            x=[w1], y=[w2], z=[z],
            mode='markers',
            marker=dict(size=6, color='red'),
            name='Lowest Loss'
        ))

    fig.update_layout(
        title="3D Loss Surface",
        scene=dict(xaxis_title="w₁", yaxis_title="w₂", zaxis_title="Loss"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig


def plot_loss_contour(W1, W2, Z, min_point=None):
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Contour(
        z=Z,
        x=W1[0],
        y=W2[:, 0],
        colorscale='Viridis',
        contours=dict(showlabels=True)
    ))
    
    if min_point:
        w1, w2, z = min_point
        fig.add_trace(go.Scatter(
            x=[w1], y=[w2],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Lowest Loss'
        ))

    fig.update_layout(
        title="2D Contour Map",
        xaxis_title="w₁",
        yaxis_title="w₂",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

