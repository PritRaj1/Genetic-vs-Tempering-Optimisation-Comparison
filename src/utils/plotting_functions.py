import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

# Set LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=14)
rc('text', usetex=True)

def plot_2D(func, x_range=(0,10)):
    """
    Function for visualising a function in R^2.

    Args:
        func (function): Function to be visualised.
        x_range (tuple): Range of x values to be visualised.
    """

    # Create a meshgrid
    x1 = np.linspace(x_range[0], x_range[1], 100)
    x2 = np.linspace(x_range[0], x_range[1], 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute the function values
    f = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            f[i, j] = func(np.array([X1[i, j], X2[i, j]]))

    # Plot contour
    plt.figure()
    plt.contourf(X1, X2, f, 100, cmap='jet')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(f'{func.__name__} Contour Plot')
    cbar = plt.colorbar()
    cbar.set_label(r'$f(x_1, x_2)$')
    plt.savefig(f'figures/{func.__name__}_contour.png')

    # Plot 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, f, cmap='rainbow', edgecolor='k')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(r'$f(x_1, x_2)$', rotation=90)
    ax.set_title(f'{func.__name__} 3D Plot')
    ax.view_init(elev=20, azim=30)  # Set the view angle
    plt.savefig(f'figures/{func.__name__}_surf.png')

