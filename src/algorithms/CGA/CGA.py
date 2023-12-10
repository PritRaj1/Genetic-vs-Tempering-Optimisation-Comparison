import numpy as np
import sys; sys.path.append('..')

# Import functions from other files
from src.algorithms.CGA.selection_functions import proportional_selection, tournament_selection, SRS_selection
from src.algorithms.CGA.mating_functions import crossover, blending

class ContinousGeneticAlgorithm():
    """
    Class for continous genetic algorithm.  
    """
    def __init__(self, population_size, chromosome_length, objective_function, range=(0,10), mutation_rate=0.1, crossover_rate=0.8, selection_method='SRS', mating_procedure='crossover',):
        """
        Constructor for continous genetic algorithm.

        Parameters:
        - population_size (int): Number of individuals in population    
        - chromosome_length (int): Size of vector individual, (number of genes), i.e. dimension of solution space
        - objective_function (function): Objective function to optimise
        - range (tuple): Range of values for genes, determined by constraints of problem
        - num_iters (int): Number of iterations
        - mutation_rate (float): Mutation rate
        - crossover_rate (float): Crossover rate
        - selection_method (str): Selection method used for parent selection
        - mating_procedure (str): Mating procedure used for reproduction
        """
        self.population_size = population_size 
        self.chromosome_length = chromosome_length # n in R^n, dimension of the search space
        self.lb = range[0] 
        self.ub = range[1] 
        self.mutation_rate = mutation_rate  
        self.crossover_rate = crossover_rate
        self.func = objective_function # Objective function to optimise

        # Dictionaries to map string to function call. Function imported from directory files
        selection_mapping = {'proportional': proportional_selection, 
                            'tournament': tournament_selection, 
                            'SRS': SRS_selection # SRS = Stochastic Remainder Selection without Replacement
                            } 
        
        mating_mapping = {'crossover': crossover,
                          'blending': blending 
                          }

        if selection_method not in ['proportional', 'tournament', 'SRS']:
            raise ValueError("Invalid selection method")
        else:
            self.selection_process = selection_mapping[selection_method]

        if mating_procedure not in ['crossover', 'blending']:
            raise ValueError("Invalid mating procedure")
        else:
            self.mating_process = mating_mapping[mating_procedure]

        self.initialise_population() # Initialise population
        
    def initialise_population(self):
        """
        Initialise population with random values between lb and ub.
        """
        self.population = np.random.uniform(low=self.lb, 
                                            high=self.ub, 
                                            size=(self.population_size, self.chromosome_length)
                                            )
        
        self.fitness = np.zeros(self.population_size) # Initialise fitness of population
        self.evaluate_fitness() # Update fitness of population with starting individual

    def evaluate_fitness(self):
        """
        Evaluate fitness of population.

        Parameters:
        - fitness_function (function): Fitness function to evaluate fitness of population
        """
        for i in range(self.population_size):
            self.fitness[i] = - self.func(self.population[i])

        # Update best individual
        self.parent_rankings = np.argsort(self.fitness) # Indices of individuals in order of fitness
        self.best_individual = self.population[self.parent_rankings[0]] # Best individual in population
    
    def select_parents(self):
        """
        Select parents for mating.

        Returns:
        - parent1 (int): Index of first parent
        - parent2 (int): Index of second parent
        """
        return self.selection_process(self)

    def mate(self):
        """
        Mate parents to produce offspring.

        Returns:
        - offspring (np.array): Offspring of parents
        """
        return self.mating_process(self)

    def mutate(self):
        """
        Mutate offspring.
        """
        for i in range(self.population_size):
            for j in range(self.chromosome_length):
                if np.random.rand() < self.mutation_rate:
                    self.population[i][j] = np.random.uniform(low=self.lb, high=self.ub)

    def evolve(self):
        """
        Evolve population for one generation.
        """
        offspring = self.mate()
        self.population = offspring
        self.mutate()
        self.evaluate_fitness() 
