"""
Candidate No : 5730E, Module: 4M17 

Description : This file contains temperature scheduling functions for parallel tempering.
                Each temerature scheduling function returns a list of temperatures ranging from 0 to 1.
"""
import numpy as np

def power_progression(num_replicas, p=1):
    """
    Uniform progression for temperature scheduling. 
    Returns (i / num_replicas)^p for i in [0, num_replicas].

    Parameters:
    - num_replicas (int): Number of replicas

    Returns:
    - schedule (list): List of temperatures
    """
    return np.linspace(0, 1, num_replicas)**p

def geometric_progression(num_replicas, p=1):
    """
    Geometric progression for temperature scheduling. 
    Returns 2^i / 2^num_replicas for i in [0, num_replicas].
    
    Parameters:
    - num_replicas (int): Number of replicas

    Returns:
    - schedule (list): List of temperatures
    """
    return np.logspace(-2, 0, num_replicas)
