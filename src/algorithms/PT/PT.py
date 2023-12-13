"""
Candidate No : 5730E, Module: 4M17 

Description :
    This file contains the class for the parallel tempering algorithm.
"""
import numpy as np

import sys; sys.path.append('..')
from src.algorithms.PT.temp_prog_functions import power_progression, geometric_progression
from src.algorithms.PT.replica_exchange_functions import period_exchange, stochastic_exchange
from src.utils.helper_functions import satisfy_constraints

class ParallelTempering():
    """
    Class for parallel tempering algorithm.  
    """
    def __init__(self, objective_function, x_dim, range=(0,10), alpha=0.1, omega=2.1, num_replicas=10, num_chains=25, exchange_procedure='Periodic', exchange_param=0.2, schedule_type='Geometric', power_term=1, total_iterations=100, constraints=True):
        """
        Constructor for parallel tempering algorithm.

        Parameters:
        - objective_function (function): Objective function to optimise
        - x_dim (int): Dimension of solution space
        - range (tuple): Range of values for x, determined by constraints of problem
        - alpha (float): Dampening constant for max. allowable step size update
        - omega (float): Weighting for max. allowable step size update
        - num_replicas (int): Number of replicas
        - num_chains (int): Number of solutions per replica
        - exchange_param (float between 0 and 1): 
            if exchange_procedure = 'Periodic', this is the percentage of iterations after which to exchange solutions
            if exchange_procedure = 'Stochastic', this is the probability of exchanging solutions between replicas during each iteration
        - progression_type (str): Type of progression for temperature scheduling
        - power_term (float > 1): Temperature progression power term, if using power progression. Schedule is (i/N)^power_term
        - total_iterations (int): Total number of iterations the algorithm will run for
        - constraints (bool): Whether to satisfy constraints with Metropolis criterion or not
        """
        self.func = objective_function
        self.x_dim = x_dim
        self.lb = range[0] 
        self.ub = range[1]
        self.alpha = alpha
        self.omega = omega 
        self.num_replicas = num_replicas
        self.num_chains = num_chains
        self.exchange_param = exchange_param
        self.constraints = constraints
        self.total_iterations = total_iterations

        # The update step suggested by Parks et al. (1990) requires control variables to be scaled to [0, 1]
        self.scale_up = lambda x: x * (self.ub - self.lb) + self.lb

        # Energy difference = (f(x_new) - f(x)) / (k * deltaT)
        k = 1.38064852e-23 # Boltzmann constant
        self.deltaE = lambda x, x_new, T: (self.func(self.scale_up(x_new)) - self.func(self.scale_up(x))) / (k * T)
        
        # Check if temperature schedule type is valid
        if schedule_type not in ['Geometric', 'Power', 'Log']:
            raise ValueError("Invalid progression type")
        
        # Dictionaries to map string to function call. Function imported from directory files
        schedule_mapping = {'Geometric': geometric_progression, 
                               'Power': power_progression
                               }
        
        # Check if power term is valid, if it's needed
        if schedule_type == 'Power':
            if power_term is None or power_term < 1:
                raise ValueError("Power term not specified correctly. Must be greater than 1.")

        self.temperature_schedule = schedule_mapping[schedule_type](num_replicas, power_term)

        # Check if exchange procedure is valid
        if exchange_procedure not in ['Periodic', 'Stochastic']:
            raise ValueError("Invalid exchange procedure")
        
        # Check if exchange parameter is valid
        if exchange_param < 0 or exchange_param > 1:
            raise ValueError("Exchange parameter must be between 0 and 1")
        
        # Dictionaries to map string to function call. Function imported from directory files
        exchange_mapping = {'Periodic': period_exchange,
                            'Stochastic': stochastic_exchange
                            }
        
        self.exchange_procedure = exchange_mapping[exchange_procedure]

        # Initialise solutions
        self.initialise_solutions()

    def initialise_solutions(self):
        """
        Initialise solutions for each replica.
        """
        # self.current_solutions = np.random.uniform(self.lb, self.ub, (self.num_replicas, self.x_dim))

        # # Diagonal matrix of max. allowable step sizes for each dimension of x
        # self.max_change = np.diag(np.random.uniform(0, 1, self.x_dim)) # Initialise stochastically

        # Initialise num_chains solutions per replica
        #self.current_solutions = np.random.uniform(self.lb, self.ub, (self.num_replicas, self.num_chains, self.x_dim))
        #self.current_solutions = np.random.uniform(4.5, 5.5, (self.num_replicas, self.num_chains, self.x_dim))
        self.current_solutions = np.random.uniform(0, 1, (self.num_replicas, self.num_chains, self.x_dim))
        
        #Initialise in feasible region
        for i in range(self.num_replicas):
            for j in range(self.num_chains):
                while not satisfy_constraints(self.scale_up(self.current_solutions[i, j])):
                    self.current_solutions[i, j] = np.random.uniform(self.lb, self.ub, self.x_dim)

        # Diagonal matrix of max. allowable step sizes for individual
        # Step for each solution
        D = np.eye(self.x_dim) * 0.75

        # Need a matrix of these for each solution in each replica
        self.max_change = np.tile(D, (self.num_replicas, self.num_chains, 1, 1))

    def get_best_solution(self):
        """
        Return best solution out of all replicas and solutions.
        """
        # Flatten array of solutions
        all_solutions = self.get_all_solutions()

        # Evaluate function at each solution
        all_solutions_eval = np.array([self.func(x) for x in all_solutions])

        # Find index of best solution
        best_idx = np.argmax(all_solutions_eval)

        # Return best solution
        return all_solutions[best_idx]

    def metropolis_criterion(self, x, x_new, T, T_new=None):
        """
        Metropolis criterion for parallel tempering algorithm.
        Acceptance probability as advised by Simulated Annealing lecture notes (Parks et al.)

        Parameters:
        - x (np.array): Current solution
        - x_new (np.array): New solution
        - T (float): Temperature

        Returns:
        - bool: Whether to accept new solution or not
        """
        # Scale up solutions to original range

        # If constraints are to be satisfied, check if new solution satisfies constraints
        if self.constraints:
            if not satisfy_constraints(self.scale_up(x_new)):
                return False

        # Calculate the L2 norm of the step size
        d = np.linalg.norm(x_new - x)

        if T * d < 1e-6:
            return True
        elif T_new is not None:
            delta_T = ((1 / T) - (1 / T_new))**(-1)
            return np.random.uniform() < min(1, np.exp(self.deltaE(x, x_new, delta_T)))
        else:
            return np.random.uniform() < min(1, np.exp(self.deltaE(x, x_new, T)))
    
    def update_max_change(self, x, x_new, i, j):
        """
        Function to update max. allowable step size whenever a solution is accepted.
        """
        # Absolute difference between new and current solution
        R = np.diag(np.abs(x_new - x)) 

        # Update max. allowable step size
        self.max_change[i][j] = (1-self.alpha) * self.max_change[i][j] + self.alpha * self.omega * R

    def update_chains(self):
        """
        One step of the algorithm. Update solutions for each replica.
        """
        # Loop through each replica, i.e. each temperature
        for i in range(self.num_replicas):

            # Loop through each solution in replica
            for j in range(self.num_chains):

                # Generate new solution
                x_new = self.current_solutions[i, j] + self.max_change[i][j] @ np.random.uniform(-1, 1, self.x_dim)

                # Metropolis criterion
                if self.metropolis_criterion(self.current_solutions[i, j], x_new, self.temperature_schedule[i]):
                    
                    # Update current solution if new solution is accepted
                    self.current_solutions[i, j] = x_new

                    # Update max. allowable step size
                    self.update_max_change(self.current_solutions[i, j], x_new, i, j)
        

    def replica_exchange(self, iter):
        """
        Function to exchange solutions between replicas.
        """
        # Call exchange procedure
        self.exchange_procedure(self, iter)
    
    def get_fitness(self):
        """
        Function to calculate average and min fitness of population.
        """
        all_solutions = self.get_all_solutions()

        avg = np.mean([self.func(x) for x in all_solutions])
        best_solution = self.get_best_solution()
        min = - self.func(best_solution)

        return avg, min

    def get_all_solutions(self):
        """
        Return all solutions reshaped into a n-dim array.
        """
        return self.scale_up(self.current_solutions.flatten().reshape(-1, self.x_dim))
        