import sys; sys.path.append('..')
from src.utils.helper_functions import generate_initial
from src.algorithms.CGA.CGA import ContinousGeneticAlgorithm
from src.algorithms.PT.PT import ParallelTempering
from src.functions import KBF_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Generate initial solutions for both functions. 
# No solution has a function value > 0.3, (away from global optimum)
initial_pop_list, seeds = generate_initial(x_dim=8, pop_size=250)
print(seeds) # These are the random seeds used to generate the initial populations

# Convergence criteria
epsilon = 0.05

### CGA Gather Results ###
CGA_AvgFit_ALL = np.zeros((50, 1000))
CGA_MinFit_ALL = np.zeros((50, 1000))
CGA_times_ALL = np.zeros(50)
CGA_i_ALL = np.zeros(50)

for run_iter, initialisation in enumerate(initial_pop_list):
    
    # Instantiate optimally-tuned CGA
    CGA = ContinousGeneticAlgorithm(population_size = 250, 
                                    chromosome_length = 8, 
                                    num_parents = 62, 
                                    objective_function = KBF_function, 
                                    tournament_size = 62, 
                                    range=(0,10), 
                                    mutation_rate=0.1, 
                                    crossover_prob=0.65, 
                                    selection_method='Tournament', 
                                    mating_procedure='Heuristic Crossover', 
                                    constraints=True)

    # Now forcibly reset population to the initialisation
    CGA.population = initialisation

    # Initialise arrays for storing fitness
    avg_fitness = np.zeros(1000)
    min_fitness = np.zeros(1000)

    # Max number of iterations = 1000
    CGA_tic = time()
    for i in range(1000):
        
        # Evolve population
        CGA.evolve()

        # Update fitness arrays
        avg_fitness[i] = np.mean(CGA.fitness)
        min_fitness[i] = CGA.min_fitness

        # Check for convergence based on difference in avg fitness
        if abs(avg_fitness[i] - avg_fitness[i-1]) < epsilon:
            CGA_toc = time()
            CGA_times_ALL[run_iter] = CGA_toc - CGA_tic
            CGA_i_ALL[run_iter] = i
    
    # Update arrays
    CGA_AvgFit_ALL[run_iter, :] = avg_fitness
    CGA_MinFit_ALL[run_iter, :] = min_fitness

### PT Gather Results ###
PT_AvgFit_ALL = np.zeros((50, 1000))
PT_MinFit_ALL = np.zeros((50, 1000))
PT_times_ALL = np.zeros(50)
PT_i_ALL = np.zeros(50)

for run_iter, initialisation in enumerate(initial_pop_list):

    # Instantiate optimally-tuned PT
    PT = ParallelTempering(objective_function=KBF_function, 
                        x_dim=8, 
                        range=(0,10), 
                        alpha=0.1, 
                        omega=2.1, 
                        num_replicas=10, 
                        num_chains=25, 
                        exchange_procedure='Always', 
                        power_term=1, 
                        constraints=True)
    
    # Forcibly reset the PT solutions
    initialisation = initialisation / 10 # Scale down to the range of the PT algorithm (0-1) 
    initialisation = initialisation.reshape(10, 25, 8) # Share solutions between replicas, i.e. from shape: (250, 8) to (10, 25, 8)
    PT.current_solutions = initialisation

    # Initialise arrays for storing fitness
    avg_fitness = np.zeros(1000)
    min_fitness = np.zeros(1000)

    # Max number of iterations = 1000
    PT_tic = time()
    for i in range(1000):
        
        # Algorithm update
        PT.update_chains()
        PT.replica_exchange(iter)

        # Update fitness arrays
        avg_fitness[i], min_fitness[i] = PT.get_fitness()

        # Check for convergence based on difference in avg fitness
        if abs(avg_fitness[i] - avg_fitness[i-1]) < epsilon:
            PT_toc = time()
            PT_times_ALL[run_iter] = PT_toc - PT_tic
            PT_i_ALL[run_iter] = i

    # Update arrays
    PT_AvgFit_ALL[run_iter, :] = avg_fitness
    PT_MinFit_ALL[run_iter, :] = min_fitness

# Plot expected average fitness values for CGA and PT with error bars
CGA_AvgFit = np.mean(CGA_AvgFit_ALL, axis=0)
CGA_convergerce_i = np.mean(CGA_i_ALL)
PT_AvgFit = np.mean(PT_AvgFit_ALL, axis=0)
PT_convergerce_i = np.mean(PT_i_ALL)

plt.figure()
sns.set_style('darkgrid')
plt.plot(CGA_AvgFit, label=f'CGA, Final = {CGA_AvgFit[-1]:.3f}')
plt.axvline(CGA_convergerce_i, color='black', linestyle='--', label=f'CGA Convergence at {CGA_convergerce_i} iterations')
plt.plot(PT_AvgFit, label='PT, Final = {PT_AvgFit[-1]:.3f}')
plt.fill_between(np.arange(1000), PT_AvgFit-PT_AvgFit_CI, PT_AvgFit+PT_AvgFit_CI, alpha=0.3)
plt.axvline(PT_convergerce_i, color='red', linestyle='--', label=f'PT Convergence at {PT_convergerce_i} iterations')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title('Expected Average Fitness across 50 Different Initialisations')
plt.savefig('figures/Final Comparison/CGA vs PT Average Fitness.png', dpi=300)

# Plot expected minimum fitness values for CGA and PT with 95% confidence intervals
CGA_MinFit = np.mean(CGA_MinFit_ALL, axis=0)
CGA_MinFit_CI = 1.96 * np.std(CGA_MinFit_ALL, axis=0) / np.sqrt(50)
PT_MinFit = np.mean(PT_MinFit_ALL, axis=0)
PT_MinFit_CI = 1.96 * np.std(PT_MinFit_ALL, axis=0) / np.sqrt(50)

plt.figure()
sns.set_style('darkgrid')
plt.plot(CGA_MinFit, label=f'CGA, Final = {CGA_MinFit[-1]:.3f}')
plt.fill_between(np.arange(1000), CGA_MinFit-CGA_MinFit_CI, CGA_MinFit+CGA_MinFit_CI, alpha=0.3)
plt.plot(PT_MinFit, label='PT, Final = {PT_MinFit[-1]:.3f}')
plt.fill_between(np.arange(1000), PT_MinFit-PT_MinFit_CI, PT_MinFit+PT_MinFit_CI, alpha=0.3)
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title('Expected Minimum Fitness across 50 Different Initialisations')
plt.savefig('figures/Final Comparison/CGA vs PT Minimum Fitness.png', dpi=300)


