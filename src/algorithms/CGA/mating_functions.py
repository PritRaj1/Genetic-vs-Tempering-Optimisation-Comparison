import numpy as np

def crossover(CGA):
    """
    Crossover mating procedure. 
    """
    # Initialise new offspring
    offspring = np.zeros((CGA.population_size, CGA.chromosome_length))

    # Iterate through population
    for i in range(CGA.population_size):

        # Select parents
        parent1, parent2 = CGA.select_parents()

        # Iterate through all the genes of an individual/chromosome
        for j in range(CGA.chromosome_length):
            
            # Crossover with probability, CGA.crossover_rate
            if np.random.rand() < CGA.crossover_rate:
                offspring[i][j] = CGA.population[parent1][j] # Take gene from parent1
            else:
                offspring[i][j] = CGA.population[parent2][j] # Take gene from parent2

    return offspring  

def blending(CGA):
    """
    Blending mating procedure. Inspired by the relevant section in https://doi.org/10.1002/0471671746.ch3
    """

    # Initialise new offspring
    offspring = np.zeros((CGA.population_size, CGA.chromosome_length))

    # Iterate through population
    for i in range(CGA.population_size):

        # Select parents
        parent1, parent2 = CGA.select_parents()
        
        # Iterate through all the genes of an individual/chromosome
        for j in range(CGA.chromosome_length):
            
            # Crossover with probability, CGA.crossover_rate
            if np.random.rand() < CGA.crossover_rate:
                b = np.random.rand()
                offspring[i][j] = b * CGA.population[parent1][j] + (1-b) * CGA.population[parent2][j] # Take gene from parent1
            else:
                offspring[i][j] = CGA.population[parent2][j] # Take gene from parent2

    return offspring



