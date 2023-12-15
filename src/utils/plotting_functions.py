"""
Candidate No : 5730E, Module: 4M17 

Description :
    This file contains various functions used for plotting figures.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import seaborn as sns
from src.utils.helper_functions import evaluate_2D

# Set LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=14)
rc('text', usetex=True)

def plot_2D(X1, X2, f, name, constraints=False):
    """
    Function for visualising a function in R^2. 
    Creates both a contour plot and a 3D surface plot.

    Args:
    - X1 (np.ndarray): Meshgrid of x1 values.
    - X2 (np.ndarray): Meshgrid of x2 values.
    - f (np.ndarray): Function values.
    - name (str): Name of function.
    - constraints (bool): Whether to plot with carved out feasible region or not.
    """

    # Change visualisation depending on whether we're plotting the carved out feasible region or not
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
    ax.plot_surface(X1, X2, f, cmap='rainbow', edgecolor='k')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(r'$f(x_1, x_2)$', rotation=90)
    ax.set_title(f'{name_png} 3D Plot')
    ax.view_init(elev=elevation, azim=angle)  # Set the view angle
    plt.savefig(f'figures/{name}/{name_png}_surf.png')

def plot_population(population, plot, best=None, last=None):
    """
    Function for overlaying a particular generation from the CGA's optimisation on a contour plot in R^2.

    Args:
    - population (np.ndarray): Population of individuals.
    - plot (matplotlib.pyplot): Plot to overlay on.
    - best (np.ndarray): Best individual.
    - last (np.ndarray): Last individual, (for visualising MCMC moves).
    """

    # Plot population
    plot.scatter(population[:,0], population[:,1], marker='x', label='Population', color='red')
    
    # Plot circle around best individual
    if best is not None:
        plot.plot(best[0], best[1], marker='o', markersize=8, label='Best', color='green')
        plot.add_patch(plt.Circle((best[0], best[1]), 0.5, color='green', fill=False)) 

    # Plot circle around last individual
    if last is not None:
        plot.plot(last[0], last[1], marker='o', markersize=8, label='Tracked', color='yellow')
        plot.add_patch(plt.Circle((last[0], last[1]), 0.5, color='yellow', fill=False))
    
    plot.legend()

def plot_sub_contour(X1, X2, f, plot, x_range=(0,10), colour='gray'):
    """
    Function for plotting the grey contour plot which will be overlayed with the population during optimisation.

    Args:
    - X1 (np.ndarray): Meshgrid of x1 values.
    - X2 (np.ndarray): Meshgrid of x2 values.
    - f (np.ndarray): Function values.
    - plot (matplotlib.pyplot): A subplot within the grid of contours to plot the contour on.
    - x_range (tuple): Range of x values.
    """
    plot.contourf(X1, X2, f, 100, cmap=colour)
    plot.set_facecolor('xkcd:light grey')
    plot.set_xlabel(r'$x_1$')
    plot.set_ylabel(r'$x_2$')
    plot.set_xlim(x_range)
    plot.set_ylim(x_range)

def plot_fitness(avg_fitness, min_fitness, type, PT=False):
    """
    Function for plotting the evolution of the average and minimum fitness of a population.

    Args:
    - avg_fitness (np.ndarray): Array of average fitness values.
    - min_fitness (np.ndarray): Array of minimum fitness values.
    - name (str): Name of function.
    - type (list): List of parameters used for the optimisation.
    - PT (bool): Whether the optimisation was performed using parallel tempering or not.
    """
    plt.figure()
    sns.set_style('darkgrid')
    plt.plot(avg_fitness, label='Average Fitness')
    plt.plot(min_fitness, label='Minimum Fitness')
    plt.xlabel('Iteration')
    plt.ylabel(r'Fitness = $-f(x_1, x_2)$')

    # Set naming based on algorithm used
    if PT:
        hyperparams = ['Exchange Procedure', 'Schedule', 'Exchange Param', 'Power Term']

    else:
        hyperparams = ['Selection', 'Mating', 'Mutation Rate', 'Crossover Prob']

    plt.title("Evolution of Fitness to " + type[0] + " Function, \n" 
              + f"[{hyperparams[0]}: " + r"\textbf{" + type[2] 
              + r"}, " + hyperparams[1] + r": \textbf{" + type[3] 
              + r"}, " + hyperparams[2] + r": \textbf{" + str(type[4])
              + r"}, " + hyperparams[3] + r": \textbf{" + str(type[5])
                + r"}]", fontsize=12)
    
    plt.legend()
    plt.savefig(f'figures/{type[0]}/{str(type[1])}_iters/{type[2]}/{type[3]}/{type[4]}_{type[5]}_Fitness.png')

def visualise_schedule(temps, func, x_range, schedule_name, func_name):
    """
    Function for visualising the effect of a temperature schedule on a function's contour plot.

    Args:
    - temps (np.ndarray): Array of temperatures.
    - func (function): Function to visualise.
    - x_range (tuple): Range of x values.
    - schedule_name (str): Name of temperature schedule.
    - func_name (str): Name of function.
    """
    X1, X2, f = evaluate_2D(func, x_range=x_range)

    num_plots = 5
    t_index = len(temps) // num_plots
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 5))
    fig.suptitle(f"Effect of {schedule_name} on {func_name} Function", fontsize=16)

    for i in range(num_plots):

        # Raise function to power of temperature
        f_t = f ** temps[i * t_index]
    
        plot_sub_contour(X1, X2, f_t, axs[i], x_range=x_range, colour='jet')
        axs[i].set_title(f'T = {temps[i * t_index]:.2f}')
        plt.colorbar(axs[i].contourf(X1, X2, f_t, 100, cmap='jet'), ax=axs[i])

    plt.savefig(f'figures/{func_name}/{schedule_name}_tempschedule.png')        


