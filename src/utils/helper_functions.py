"""
Candidate No : 5730E, Module: 4M17 

Description :
    This file contains some helper functions for the project.
"""

import numpy as np
import os

def satisfy_constraints(x):
    """
    Function to check if a given vector x satisfies the constraints of the problem.

    Args:
    - x (np.ndarray): Vector to check.

    Returns:
    - bool: True if x satisfies constraints, False otherwise.
    """

    # List of boolean values for each constraint satisfied
    constraints = [
        np.all(x >= 0) and np.all(x <= 10),
        np.prod(x) > 0.75,
        np.sum(x) < 15 * x.shape[0] / 2,
    ]

    # Return True if all constraints are satisfied, False otherwise
    return all(constraints)

def evaluate_2D(func, x_range=(0,10), constraints=False): 
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

            if constraints == True:
                if not satisfy_constraints(np.array([X1[i, j], X2[i, j]])):
                    f[i, j] = np.nan
        
    return X1, X2, f

def create_figure_directories(name, selection_methods, mating_procedures, iters_list):
    """
    Function for creating directories for figures generated in my simulations.
    """
    # Create parent directory
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Create directory for specific function
    function_dir = os.path.join('figures', name)
    if not os.path.exists(function_dir):
        os.makedirs(function_dir)

    # Create directory for each number of iterations
    for iters in iters_list:
        iters_dir = os.path.join(function_dir, f'{iters}_iters')
        if not os.path.exists(iters_dir):
            os.makedirs(iters_dir)

    # Create directories for each selection method and mating procedure
    for selection_method in selection_methods:
        for mating_procedure in mating_procedures:
            for iters in iters_list:
                selection_dir = os.path.join(function_dir, f'{str(iters)}_iters', selection_method)
                if not os.path.exists(selection_dir):
                    os.makedirs(selection_dir)

                mating_dir = os.path.join(selection_dir, mating_procedure)
                if not os.path.exists(mating_dir):
                    os.makedirs(mating_dir)