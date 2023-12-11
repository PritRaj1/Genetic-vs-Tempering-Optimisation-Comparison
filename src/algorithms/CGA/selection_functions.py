import numpy as np

import sys; sys.path.append('..')
from src.utils.helper_functions import satisfy_constraints

def proportional_selection(GCA):
        """
        Proportional selection of parents. 

        Args:
        - GCA (CGA): Continuous Genetic Algorithm object passed into this function using self.select_parents(self)

        Returns:
        - parent1 (int): Index of first parent
        - parent2 (int): Index of second parent
        """
        # Calculate probabilities
        probabilities = GCA.fitness / np.sum(GCA.fitness)
        
        # Initial selection of parents
        parent1 = np.random.choice(GCA.population_size, p=probabilities)
        parent2 = np.random.choice(GCA.population_size, p=probabilities)

        # Select parents based on probabilities, reject parents that do not satisfy constraints
        if GCA.constraints == True:
            while not satisfy_constraints(GCA.population[parent1]):
                parent1 = np.random.choice(GCA.population_size, p=probabilities)
            while not satisfy_constraints(GCA.population[parent2]):
                parent2 = np.random.choice(GCA.population_size, p=probabilities)

        return parent1, parent2

def tournament_selection(GCA):
    """
    Tournament selection of parents. 

    Args:
    - GCA (CGA): Continuous Genetic Algorithm object passed into this function using self.select_parents(self)

    Returns:
    - parent1 (int): Index of first parent
    - parent2 (int): Index of second parent
    """
    # Take subset of population
    subset = np.random.choice(GCA.population_size, size=GCA.tournament_size, replace=False)

    # Take top two parents
    parent1 = subset[np.argmax(GCA.fitness[subset])]
    subset = np.delete(subset, np.argmax(GCA.fitness[subset]))
    parent2 = subset[np.argmax(GCA.fitness[subset])]

    # Select parents based on probabilities, reject parents that do not satisfy constraints
    if GCA.constraints == True:
        while not satisfy_constraints(GCA.population[parent1]):
            subset = np.delete(subset, np.argmax(GCA.fitness[subset]))
            parent1 = subset[np.argmax(GCA.fitness[subset])]
        while not satisfy_constraints(GCA.population[parent2]):
            subset = np.delete(subset, np.argmax(GCA.fitness[subset]))
            parent2 = subset[np.argmax(GCA.fitness[subset])]
    
    return parent1, parent2
        
def SRS_selection(GCA):
    """
    Stochastic Remainder Selection without Replacement (SRS) of parents. 

    Args:
    - GCA (CGA): Continuous Genetic Algorithm object passed into this function using self.select_parents(self)

    Returns:
    - parent1 (int): Index of first parent
    - parent2 (int): Index of second parent
    """

    # Calculate probabilities
    probabilities = GCA.fitness / np.sum(GCA.fitness)

    # Calculate expected number of copies of each individual
    expected_num_copies = probabilities * GCA.population_size

    # Calculate number of copies of each individual, each individual is selected this number of times
    num_copies = np.floor(expected_num_copies) 

    # Calculate remainder, which serves as the probability of further selection
    remainder = expected_num_copies - num_copies

    # Calculate number of individuals to be selected
    num_selected = int(np.sum(num_copies))

    # Select individuals
    selected_individuals = []

    # Select individuals with num_copies
    for i in range(GCA.population_size):
        selected_individuals += [i] * int(num_copies[i]) # Add i to list num_copies[i] times

    # Remainer must satisfy sum(remainder) = 1
    remainder = remainder / np.sum(remainder)

    # Select individuals with remainder serving as probability
    selected_individuals += list(np.random.choice(GCA.population_size, size=num_selected, p=remainder))

    # Select parents
    parent1 = np.random.choice(selected_individuals)
    parent2 = np.random.choice(selected_individuals)

    # Select parents based on probabilities, reject parents that do not satisfy constraints
    if GCA.constraints == True:
        while not satisfy_constraints(GCA.population[parent1]):
            parent1 = np.random.choice(selected_individuals)
        while not satisfy_constraints(GCA.population[parent2]):
            parent2 = np.random.choice(selected_individuals)
    
    return parent1, parent2

    


