import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
def ms(x, y, z, radius, resolution=20):
    """
    Return the coordinates for plotting a sphere centered at (x, y, z).

    Args:
        x (float): X-coordinate of the sphere center.
        y (float): Y-coordinate of the sphere center.
        z (float): Z-coordinate of the sphere center.
        radius (float): Radius of the sphere.
        resolution (int): Resolution of the sphere surface mesh.

    Returns:
        tuple: Coordinates (X, Y, Z) for plotting the sphere.
    """
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

def plot_sphere_and_data(sphere_coord, color_value=1, fixed_color='blue'):
    """
    Create a 3D plot of the sphere and data points.

    Args:
        sphere_coord (np.ndarray): Coordinates of data points on the sphere.
        color_value (float): Value for fixed color.
        fixed_color (str): Color for the sphere surface.
    """
    scatter_colors = np.linspace(0, 1, sphere_coord.shape[0])

    x_pns_surface, y_pns_surface, z_pns_surface = ms(0, 0, 0, 1, 100)
    fig = go.Figure(data=go.Surface(
        x=x_pns_surface, 
        y=y_pns_surface, 
        z=z_pns_surface, 
        opacity=0.2, 
        showscale=False,
        colorscale=[[0, fixed_color], [1, fixed_color]], 
        cmin=color_value, 
        cmax=color_value
    ))

    fig.add_trace(go.Scatter3d(
        x=sphere_coord[:,0], 
        y=sphere_coord[:,1], 
        z=sphere_coord[:,2],
        opacity=0.3,
        mode='markers',
        marker=dict(
            size=5, 
            color=scatter_colors,  
            colorscale=[[0, 'red'], [1, 'yellow']]
        )
    ))

    fig.update_layout(autosize=False, height=800, width=1000)
    fig.show()

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses over epochs."""
    # plt.cla()
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_gradients(graddf_in: pd.DataFrame, opacity_in:float):
    """Plot gradients of each layer over training iterations."""
    # plt.cla()
    graddf_in['epoch_iter'] = graddf_in['epoch'].astype(str) + '-' + graddf_in['iteration'].astype(str)
    
    plt.figure()
    plt.plot(graddf_in['fc1_grad'], label='fc1_grad', linestyle='-', marker='o', alpha=opacity_in)
    plt.plot(graddf_in['fc2_grad'], label='fc2_grad', linestyle='-', marker='o', alpha=opacity_in)
    plt.plot(graddf_in['fc3_grad'], label='fc3_grad', linestyle='-', marker='o', alpha=opacity_in)
    plt.plot(graddf_in['fc4_grad'], label='fc4_grad', linestyle='-', marker='o', alpha=opacity_in)
    
    plt.xlabel('Epoch-Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradients of Each Layer Over Training Iterations')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_hist_kde(data, xlabel, title, color):
    # Plot the histogram
    plt.hist(data, bins=30, density=True, color=color, alpha=0.6, edgecolor='black')
    
    # Plot the KDE
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 1000)
    kde_values = kde(x_range)
    plt.plot(x_range, kde_values, color='black')

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(title)

def plot_sphere_and_results(outputs, targets):
    """
    Create a 3D plot of the sphere, outputs, and targets.

    Args:
        sphere_coord (np.ndarray): Coordinates of data points on the sphere (same as targets).
        outputs (np.ndarray): Predicted outputs by the model.
        targets (np.ndarray): Ground truth target values.
    """
    scatter_colors = np.linspace(0, 1, outputs.shape[0])

    # Create sphere surface
    x_pns_surface, y_pns_surface, z_pns_surface = ms(0, 0, 0, 1, 100)
    fig = go.Figure(data=go.Surface(
        x=x_pns_surface, 
        y=y_pns_surface, 
        z=z_pns_surface, 
        opacity=0.2, 
        showscale=False,
        colorscale=[[0, 'blue'], [1, 'blue']],
        cmin=0,
        cmax=1
    ))

    # Plot targets
    fig.add_trace(go.Scatter3d(
        x=targets[:, 0], 
        y=targets[:, 1], 
        z=targets[:, 2],
        opacity=0.6,
        mode='markers',
        marker=dict(
            size=5, 
            color='green',  
            symbol='circle'
        ),
        name='Targets'
    ))

    # Plot outputs
    fig.add_trace(go.Scatter3d(
        x=outputs[:, 0], 
        y=outputs[:, 1], 
        z=outputs[:, 2],
        opacity=0.6,
        mode='markers',
        marker=dict(
            size=5, 
            color='orange',  
            symbol='x'
        ),
        name='Outputs'
    ))
    
    for i in range(len(outputs)):
        fig.add_trace(go.Scatter3d(
            x=[outputs[i, 0], targets[i, 0]],
            y=[outputs[i, 1], targets[i, 1]],
            z=[outputs[i, 2], targets[i, 2]],
            mode='lines',
            line=dict(color='red', width=0.8),
            showlegend=False
        ))
        
    fig.update_layout(autosize=False, height=800, width=1000, title="Outputs, and Targets")
    fig.show()



