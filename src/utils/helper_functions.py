import numpy as np
import os

def evaluate_2D(func, x_range=(0,10)): 
    """
    Function for generating a meshgrid and evaluating a function in R^2.
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
        
    return X1, X2, f

def create_figure_directories(name, selection_methods, mating_procedures):
    """
    Function for creating directories for figures.
    """
    # Create parent directory
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Create directory for specific function
    function_dir = os.path.join('figures', name)
    if not os.path.exists(function_dir):
        os.makedirs(function_dir)

    # Create directories for each selection method and mating procedure
    for selection_method in selection_methods:
        selection_dir = os.path.join(function_dir, selection_method)
        if not os.path.exists(selection_dir):
            os.makedirs(selection_dir)
        for mating_procedure in mating_procedures:
            mating_dir = os.path.join(selection_dir, mating_procedure)
            if not os.path.exists(mating_dir):
                os.makedirs(mating_dir)