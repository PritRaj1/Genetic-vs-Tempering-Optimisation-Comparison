"""
Candidate No : 5730E, Module: 4M17 

Description :
    This file contains the class for the continous genetic algorithm.
"""

import numpy as np
import sys; sys.path.append('..')

# Import selection and mating functions from directory files
from src.algorithms.CGA.selection_functions import proportional_selection, tournament_selection, SRS_selection
from src.algorithms.CGA.mating_functions import crossover, heuristic_crossover
from src.utils.helper_functions import satisfy_constraints

class ContinousGeneticAlgorithm():
    """
    Class for continous genetic algorithm.  
    """
    def __init__(self, population_size, chromosome_length, num_parents, objective_function, tournament_size, range=(0,10), mutation_rate=0.1, crossover_prob=0.8, selection_method='Tournament', mating_procedure='Heuristic Crossover', constraints=True):
        """
        Constructor for continous genetic algorithm.

        Parameters:
        - population_size (int): Number of individuals in population    
        - chromosome_length (int): Size of vector individual, (number of genes), i.e. dimension of solution space
        - num_parents (int): Number of parents to select for mating
        - objective_function (function): Objective function to optimise
        - tournament_size (int): Size of subset of population for tournament selection
        - range (tuple): Range of values for genes, determined by constraints of problem
        - num_iters (int): Number of iterations
        - mutation_rate (float): Mutation rate
        - crossover_prob (float): Crossover rate
        - selection_method (str): Selection method used for parent selection
        - mating_procedure (str): Mating procedure used for reproduction
        - constraints (bool): Whether to satisfy constraints with parent selection or not
        """
        self.population_size = population_size 
        self.chromosome_length = chromosome_length # n in R^n, dimension of the search space
        self.num_parents = num_parents
        self.func = objective_function
        self.tournament_size = tournament_size
        self.lb = range[0] 
        self.ub = range[1] 
        self.mutation_rate = mutation_rate  
        self.crossover_prob = crossover_prob
        self.constraints = constraints
        
        # Dictionaries to map string to function call. Function imported from directory files
        selection_mapping = {'Proportional': proportional_selection, 
                            'Tournament': tournament_selection, 
                            'SRS': SRS_selection # SRS = Stochastic Remainder Selection without Replacement
                            } 
        
        mating_mapping = {'Crossover': crossover,
                          'Heuristic Crossover': heuristic_crossover 
                          }

        if selection_method not in ['Proportional', 'Tournament', 'SRS']:
            raise ValueError(f"Invalid selection method: {selection_method}")
        else:
            self.selection_process = selection_mapping[selection_method]

        if mating_procedure not in ['Crossover', 'Heuristic Crossover']:
            raise ValueError(f"Invalid mating procedure: {mating_procedure}")
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

        # Evaluate rankings of individuals in population by fitness
        self.parent_rankings = np.argsort(self.fitness) # Indices of individuals in order of fitness

        # Update best individual and best fitness
        self.best_individual = self.population[self.parent_rankings[0]]
        self.min_fitness = self.fitness[self.parent_rankings[0]]

        # Best individual must satisfy constraints, if not, find next best individual that does
        i = 1
        while not satisfy_constraints(self.best_individual):
            self.best_individual = self.population[self.parent_rankings[i]]
            self.min_fitness = self.fitness[self.parent_rankings[i]]
            i += 1
            
    def select_parents(self):
        """
        Select parents for mating.

        Returns:
        - parents (np.array): Indices of parents selected for mating
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
    
