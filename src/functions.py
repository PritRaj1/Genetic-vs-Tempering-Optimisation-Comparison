"""
Candidate No : 5730E, Module: 4M17 

Description :
    This file contains the implementation of the Keane's Bump Function and the Rosenbrock function.
"""


import numpy as np

def KBF_function(x, eps=1e-8):
    """
    Function implements Keane's Bump Function for a given input vector x of shape n x 1.

    Args:
        x (np.ndarray): Input vector of shape n x 1.
        eps (float): Small value to avoid division by zero.

    Returns:
        f (float): Function value at x.
    """

    # Get the number of dimensions
    n = x.shape[0]

    # Compute the numerator
    num = np.sum(np.cos(x)**4) - 2*np.prod(np.cos(x)**2)

    # Compute the denominator
    den = np.sqrt(np.sum(np.arange(1, n+1)*x**2))

    # Avoid division by zero
    if den == 0:
        den = eps

    # Compute the function value
    f = np.abs(num/den)

    return f

def Rosenbrock_function(x):
    """
    Function implements the Rosenbrock function for a given input vector x of shape n x 1.
    Used for testing optimisation algorithms.

    Args:
        x (np.ndarray): Input vector of shape n x 1.

    Returns:
        f (float): Function value at x.
    """

    # Get the number of dimensions
    n = x.shape[0]

    # Compute the function value
    f = 0
    for i in range(n-1):
        f += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2

    return f 

