import numpy as np

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

        # Select parents based on probabilities
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
        # Select parents
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

    # Select the first parent based on probabilities
    parent1 = np.random.choice(GCA.population_size, p=probabilities)

    # Update probabilities after selecting the first parent
    updated_probabilities = np.delete(probabilities, parent1)
    updated_probabilities /= np.sum(updated_probabilities)

    # Select the second parent based on updated probabilities
    parent2 = np.random.choice(GCA.population_size - 1, p=updated_probabilities)

    # Adjust the index of parent2 to account for the removed element
    parent2 = parent2 + 1 if parent2 >= parent1 else parent2

    return parent1, parent2

