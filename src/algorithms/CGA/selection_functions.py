"""
Candidate No : 5730E, Module: 4M17 

This file contains the selection functions for the CGA algorithm.
"""

import numpy as np
import sys; sys.path.append('..')
from src.utils.helper_functions import satisfy_constraints

def proportional_selection(GCA):
    """
    Proportional selection of parents. 

    Args:
    - GCA (CGA): Continuous Genetic Algorithm object passed into this function using self.select_parents(self)

    Returns:
    - selected_individuals (list): List of indices of selected individuals for mating, length = GCA.num_parents
    """
    # Calculate probabilities
    probabilities = GCA.fitness / np.sum(GCA.fitness)

    # Select individuals based on probabilities
    selected_individuals = list(np.random.choice(GCA.population_size, size=GCA.num_parents, p=probabilities))

    # Retry - reject parents that do not satisfy constraints
    if GCA.constraints == True:
        for i in range(GCA.num_parents):
            while not satisfy_constraints(GCA.population[selected_individuals[i]]):
                selected_individuals[i] = np.random.choice(GCA.population_size, p=probabilities)
    
    return selected_individuals

def tournament_selection(GCA):
    """
    Tournament selection of parents. 

    Args:
    - GCA (CGA): Continuous Genetic Algorithm object passed into this function using self.select_parents(self)

    Returns:
    - selected_individuals (list): List of indices of selected individuals for mating, length = GCA.num_parents
    """

    # Initialise list of selected individuals
    selected_individuals = []

    # Select top two parents for each tournament, so need 'GCA.num_parents // 2' tournaments
    for i in range(GCA.num_parents//2):

        # Take subset of population
        subset = np.random.choice(GCA.population_size, size=GCA.tournament_size, replace=False)

        # Take top two parents
        parent1 = subset[np.argmin(GCA.fitness[subset])]
        subset = np.delete(subset, np.argmin(GCA.fitness[subset]))
        parent2 = subset[np.argmin(GCA.fitness[subset])]

        # Retry, reject parents that do not satisfy constraints
        if GCA.constraints == True:
            while not satisfy_constraints(GCA.population[parent1]):
                subset = np.delete(subset, np.argmin(GCA.fitness[subset]))
                parent1 = subset[np.argmin(GCA.fitness[subset])]
            while not satisfy_constraints(GCA.population[parent2]):
                subset = np.delete(subset, np.argmin(GCA.fitness[subset]))
                parent2 = subset[np.argmin(GCA.fitness[subset])]

        # Add parents to list of selected individuals
        selected_individuals += [parent1, parent2]

    return selected_individuals

def SRS_selection(GCA):
    """
    Stochastic Remainder Selection without Replacement (SRS) of parents. 

    Args:
    - GCA (CGA): Continuous Genetic Algorithm object passed into this function using self.select_parents(self)

    Returns:
    - selected_individuals (list): List of indices of selected individuals for mating, length = GCA.num_parents
    """

    # Calculate probabilities
    probabilities = GCA.fitness / np.sum(GCA.fitness)

    # Calculate expected number of copies of each individual
    expected_num_copies = probabilities * GCA.num_parents

    # Calculate integer number of copies of each individual
    num_copies = np.floor(expected_num_copies) 

    # Calculate remainder, which later serves as the probability of further selection
    remainder = expected_num_copies - num_copies

    # Initialise list of selected individuals
    selected_individuals = []

    # Duplicate individuals "num_copies" times
    for i in range(GCA.population_size):

        # Add only feasible individuals
        if satisfy_constraints(GCA.population[i]):
            selected_individuals += [i] * int(num_copies[i]) # Add i to list num_copies[i] times

    # Remainer must satisfy sum(remainder) = 1, since it serves as the probabilities for further selection
    remainder = remainder / np.sum(remainder)

    # Calculate remaining number of individuals that need to be selected
    remaining_number = GCA.num_parents - len(selected_individuals)
    
    # Cannot be negative
    remaining_number = remaining_number if remaining_number > 0 else 0

    # Select individuals using remainder probabilities
    selected_individuals += list(np.random.choice(GCA.population_size, size=remaining_number, p=remainder))

    # Reject parents that do not satisfy constraints
    if GCA.constraints == True:
        for i in range(len(selected_individuals)):
            while not satisfy_constraints(GCA.population[selected_individuals[i]]):
                selected_individuals[i] = np.random.choice(GCA.population_size, p=remainder)
    
    return selected_individuals
    


