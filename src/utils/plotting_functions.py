import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import seaborn as sns

# Set LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=14)
rc('text', usetex=True)

def plot_2D(X1, X2, f, name, constraints=False):
    """
    Function for visualising a function in R^2.

    Args:
    - X1 (np.ndarray): Meshgrid of x1 values.
    - X2 (np.ndarray): Meshgrid of x2 values.
    - f (np.ndarray): Function values.
    - name (str): Name of function.
    - constraints (bool): Whether to plot with carved out feasible region or not.
    """

    if constraints == True:
        name_png = f'{name} Feasible'
        angle = -10
        elevation = 30

    else: 
        name_png = name
        angle = 30
        elevation = 20

    # Plot contour
    plt.figure()
    plt.gca().set_facecolor('xkcd:light grey')
    plt.contourf(X1, X2, f, 100, cmap='jet')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(f'{name_png} Contour Plot')
    cbar = plt.colorbar()
    cbar.set_label(r'$f(x_1, x_2)$')
    plt.savefig(f'figures/{name}/{name_png}_contour.png')

    # Plot 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, f, cmap='rainbow', edgecolor='k')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(r'$f(x_1, x_2)$', rotation=90)
    ax.set_title(f'{name_png} 3D Plot')
    ax.view_init(elev=elevation, azim=angle)  # Set the view angle
    plt.savefig(f'figures/{name}/{name_png}_surf.png')

def plot_population(population, plot, best=None):
    """
    Function for overlaying a population from CGA on a function in R^2.

    Args:
    - X1 (np.ndarray): Meshgrid of x1 values.
    - X2 (np.ndarray): Meshgrid of x2 values.
    - f (np.ndarray): Function values for plotting contour.
    - population (np.ndarray): Population to overlay on contour
    - i (int): Iteration number
    - best (np.ndarray): Best individual in population
    - x_range (tuple): Range of x values to display
    - y_range (tuple): Range of y values to display
    """

    # Plot population
    plot.scatter(population[:,0], population[:,1], marker='x', label='Population', color='red')
    
    # Plot circle around best individual
    if best is not None:
        plot.plot(best[0], best[1], marker='x', markersize=10, label='Best', color='blue')
        plot.add_patch(plt.Circle((best[0], best[1]), 0.5, color='blue', fill=False)) 
    
    plot.legend()

def plot_grey_contour(X1, X2, f, plot, x_range=(0,10)):
    """
    Function for visualising a function in R^2.

    Args:
    - X1 (np.ndarray): Meshgrid of x1 values.
    - X2 (np.ndarray): Meshgrid of x2 values.
    - f (np.ndarray): Function values.
    - name (str): Name of function.
    """
    plot.contourf(X1, X2, f, 100, cmap='gray')
    plot.set_facecolor('xkcd:light grey')
    plot.set_xlabel(r'$x_1$')
    plot.set_ylabel(r'$x_2$')
    plot.set_xlim(x_range)
    plot.set_ylim(x_range)

def plot_fitness(avg_fitness, min_fitness, type):
    """
    Function for plotting the evolution of the average and minimum fitness of a population.

    Args:
    - avg_fitness (np.ndarray): Array of average fitness values.
    - min_fitness (np.ndarray): Array of minimum fitness values.
    - name (str): Name of function.
    """
    plt.figure()
    sns.set_style('darkgrid')
    plt.plot(avg_fitness, label='Average Fitness')
    plt.plot(min_fitness, label='Minimum Fitness')
    plt.xlabel('Iteration')
    plt.ylabel(r'Fitness = $-f(x_1, x_2)$')

    plt.title("Evolution of Fitness to " + type[0] + " Function, \n" 
              + r"[Selection: \textbf{" + type[2] 
              + r"}, Mating: \textbf{" + type[3] 
              +  r"}, Mutation Rate: \textbf{" + str(type[4])
                + r"}, Crossover Rate: \textbf{" + str(type[5])
                + r"}]", fontsize=12)
    
    plt.legend()
    plt.savefig(f'figures/{type[0]}/{str(type[1])}_iters/{type[2]}/{type[3]}/{type[4]}_{type[5]}_Fitness.png')





