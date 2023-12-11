import numpy as np

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

        else:
            parent1 = np.random.choice(GCA.population_size, p=probabilities)
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
        # Initial selection of parents
        parent1 = np.random.randint(GCA.population_size)
        parent2 = np.random.randint(GCA.population_size)

        # Select two random parents, reject parents that do not satisfy constraints
        if GCA.constraints == True:
            while not satisfy_constraints(GCA.population[parent1]):
                parent1 = np.random.randint(GCA.population_size)
            while not satisfy_constraints(GCA.population[parent2]):
                parent2 = np.random.randint(GCA.population_size)
        
        else:
            parent1 = np.random.randint(GCA.population_size)
            parent2 = np.random.randint(GCA.population_size)

        # Select fittest parent
        if GCA.fitness[parent1] > GCA.fitness[parent2]:
            return parent1, parent2
        else:
            return parent2, parent1
        
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

    # Initial selection of parent
    parent1 = np.random.choice(GCA.population_size, p=probabilities)

    # Select the first parent based on probabilities, reject parent if it does not satisfy constraints
    if GCA.constraints == True:
        while not satisfy_constraints(GCA.population[parent1]):
            parent1 = np.random.choice(GCA.population_size, p=probabilities)
    else:
        parent1 = np.random.choice(GCA.population_size, p=probabilities)

    # Update probabilities after selecting the first parent
    updated_probabilities = np.delete(probabilities, parent1)
    updated_probabilities /= np.sum(updated_probabilities)

    # Initial selection of parent
    parent2 = np.random.choice(GCA.population_size - 1, p=updated_probabilities)

    # Select the second parent based on updated probabilities, reject parent if it does not satisfy constraints
    if GCA.constraints == True:
        while not satisfy_constraints(GCA.population[parent2]):
            parent2 = np.random.choice(GCA.population_size - 1, p=updated_probabilities)
    else:
        parent2 = np.random.choice(GCA.population_size - 1, p=updated_probabilities)

    # Adjust the index of parent2 to account for the removed element
    parent2 = parent2 + 1 if parent2 >= parent1 else parent2

    return parent1, parent2

