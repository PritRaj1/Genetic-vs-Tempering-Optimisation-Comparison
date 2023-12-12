"""
Candidate No : 5730E, Module: 4M17 

This file contains the mating functions for the CGA algorithm.
"""

import numpy as np

def crossover(CGA):
    """
    Crossover mating procedure. 
    """
    # Select parents
    selected_parents = CGA.select_parents()

    # Reschape them into a set of potential parent pairs
    parent_pairs = np.array(selected_parents).reshape(-1, 2)

    # Initialise new offspring to replace population
    offspring = np.zeros((CGA.population_size, CGA.chromosome_length))

    # Iterate through population
    for i in range(CGA.population_size):

        # Assign random pair of parents from parent_pairs
        parent1, parent2 = parent_pairs[np.random.randint(len(parent_pairs))]

        # Assign random crossover point
        p = np.random.randint(1, CGA.chromosome_length)

        # Swap genes from parents to create offspring
        if np.random.rand() < CGA.crossover_prob:
            offspring[i][:p] = CGA.population[parent1][:p]
            offspring[i][p:] = CGA.population[parent2][p:]
        else:
            offspring[i][:p] = CGA.population[parent2][:p]
            offspring[i][p:] = CGA.population[parent1][p:]

    return offspring

def heuristic_crossover(CGA):
    """
    Blending mating procedure. Inspired by the relevant section in https://doi.org/10.1002/0471671746.ch3
    """

    # Select parents
    selected_parents = CGA.select_parents()

    # Reschape them into a set of potential parent pairs
    parent_pairs = np.array(selected_parents).reshape(-1, 2)

    # Initialise new offspring to replace population
    offspring = np.zeros((CGA.population_size, CGA.chromosome_length))

    # Iterate through population
    for i in range(CGA.population_size):

        # Assign random pair of parents from parent_pairs
        parent1, parent2 = parent_pairs[np.random.randint(len(parent_pairs))]
        
        # Iterate through all the genes of an individual/chromosome
        for j in range(CGA.chromosome_length):
            
            # Heuristic crossover with probability, CGA.crossover_prob
            b = np.random.rand() # b is a random number between 0 and 1

            if np.random.rand() < CGA.crossover_prob:
                # p_new = b * (p1 - p2) + p2 
                offspring[i][j] = b * (CGA.population[parent1][j] - CGA.population[parent2][j]) + CGA.population[parent2][j]     
            else:
                # p_new = b * (p2 - p1) + p1
                offspring[i][j] = b * (CGA.population[parent2][j] - CGA.population[parent1][j]) + CGA.population[parent1][j]
                
    return offspring


