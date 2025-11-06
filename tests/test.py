# Import libraries 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import seaborn as sns 

# Function to plot the triangulation
def plot_mesh(triangulation, aspect=1.4, xlabel="Longitude", ylabel="Latitude", show=True):
    """
    Plot the triangulation.
    
    Parameters
    ----------
    triangulation : matplotlib.tri.Triangulation  
        Triangulation object containing the coordinates and triangle connectivity.  
    show : bool  
        If True, displays the plot; otherwise, returns the fig and ax objects.
    """
    fig, ax = plt.subplots()
    ax.triplot(triangulation, color='black', linewidth=0.5)
    ax.set_aspect(aspect)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show:
        plt.show()
    else:
        return fig, ax


# Function to plot the data
def plot_data(locations, values, boundary_nodes=None, cmap="mako", aspect=1.4,
              xlabel="Longitude", ylabel="Latitude", cbar_label="log_rainfall",
              point_size=20, show=True):
    """
    Plots point data (e.g., log_rainfall) at the given coordinates.
    
    Parameters
    ----------
    locations : ndarray (n x 2)
        Coordinates of the points (column 0 = x, column 1 = y).
    values : ndarray (n)
        Values associated with each point (e.g., log_rainfall).
    boundary_nodes : ndarray (m x 2), optional
        Coordinates of boundary nodes to be drawn as a closed line.
    cmap : str
        Name of the colour palette to use (default = "mako").
    aspect : float
        Aspect ratio of the axes (default = 1.4).
    xlabel, ylabel : str
        Axis labels.
    cbar_label : str
        Label for the colour bar.
    point_size : int
        Point size (default = 20).
    show : bool
        If True, displays the plot; otherwise, returns (fig, ax).
    """
    
    fig, ax = plt.subplots()

    # Data scatter
    sc = ax.scatter(
        x=locations[:, 0],
        y=locations[:, 1],
        c=values,
        cmap=sns.color_palette(cmap, as_cmap=True),
        s=point_size,
        edgecolor='none'
    )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03)
    cbar.set_label(cbar_label, labelpad=10)

    # Boundary 
    if boundary_nodes is not None:
        ax.plot(
            np.concatenate((boundary_nodes[:, 0], [boundary_nodes[0, 0]])),
            np.concatenate((boundary_nodes[:, 1], [boundary_nodes[0, 1]])),
            color='black',
            linewidth=1
        )

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect(aspect)

    if show:
        plt.show()
    else:
        return fig, ax


# Function to plot the solution field

def plot_fit(triangulation, f_fit, boundary_nodes=None, cmap="mako",
             aspect=1.4, figsize=(10, 5), show=True):
    """
    Plots the fitted field in two panels:
      - left: log-scale (f_fit)
      - right: original scale (exp(f_fit))

    Parameters
    ----------
    triangulation : matplotlib.tri.Triangulation
        Triangulation object.
    f_fit : ndarray (n)
        Estimated field values (e.g., f_50).
    boundary_nodes : ndarray (m x 2), optional
        Coordinates of boundary nodes to draw the outline.
    cmap : str
        Name of the colour palette (default = "mako").
    aspect : float
        Aspect ratio of the axes (default = 1.4).
    figsize : tuple
        Figure size (default = (10, 5)).
    show : bool
        If True, displays the plot; otherwise, returns (fig, ax).
    """

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ## Left plot: fit in the logarithmic scale
    tpc1 = ax[0].tripcolor(triangulation, f_fit, cmap=sns.color_palette(cmap, as_cmap=True))

    # Boundary
    if boundary_nodes is not None:
        ax[0].plot(
            np.concatenate((boundary_nodes[:, 0], [boundary_nodes[0, 0]])),
            np.concatenate((boundary_nodes[:, 1], [boundary_nodes[0, 1]])),
            color='black',
            linewidth=1
        )

    ax[0].set_xlabel('Longitude')
    ax[0].set_ylabel('Latitude')
    ax[0].set_title("50%-quantile log-rainfall")
    ax[0].set_aspect(aspect)

    # Colorbar 
    cbar1 = fig.colorbar(tpc1, ax=ax[0], fraction=0.03)
    cbar1.set_label("log-rainfall", labelpad=10)

    ## Right plot: fit in the original scale
    tpc2 = ax[1].tripcolor(triangulation, np.exp(f_fit), cmap=sns.color_palette(cmap, as_cmap=True))

    # Boundary
    if boundary_nodes is not None:
        ax[1].plot(
            np.concatenate((boundary_nodes[:, 0], [boundary_nodes[0, 0]])),
            np.concatenate((boundary_nodes[:, 1], [boundary_nodes[0, 1]])),
            color='black',
            linewidth=1
        )

    ax[1].set_xlabel('Longitude')
    ax[1].set_ylabel('Latitude')
    ax[1].set_title("50%-quantile rainfall")
    ax[1].set_aspect(aspect)

    # Colorbar 
    cbar2 = fig.colorbar(tpc2, ax=ax[1], fraction=0.03)
    cbar2.set_label("rainfall", labelpad=10)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        return fig, ax



# Read mesh nodes, boundary nodes and mesh elements
nodes = np.loadtxt("data/points.txt", dtype=float)
boundary_nodes = np.loadtxt("data/boundary_nodes.txt", dtype=float)
triangles = np.loadtxt("data/elements.txt", dtype=int)
triangles = triangles - 1 

# Create the object triangulation 
triangulation = Triangulation(nodes[:,0], nodes[:,1], triangles)

# Plot the mesh 
plot_mesh(triangulation)


# Import locations and data 
locations = np.loadtxt("data/locations.txt", dtype=float)
log_rainfall = np.loadtxt("data/log_rainfall.txt", dtype=float)  # rainfall data in logarithmic scale


# Plot data 
plot_data(
    locations=locations,
    values=log_rainfall,
    boundary_nodes=boundary_nodes,
    cmap="mako",           
    cbar_label="log-rainfall"
)


# Read solution
f_50 = np.loadtxt("data/f_50.txt", dtype=float)   # alpha=50%
f_90 = np.loadtxt("data/f_90.txt", dtype=float)   # alpha=90%


# Plot solution field 
plot_fit(
    triangulation=triangulation,
    f_fit=f_50,
    boundary_nodes=boundary_nodes,
    cmap="mako",  
)


