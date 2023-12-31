"""
Author: Prithvi Raj

Description :
    This file serves to compare the optimally-tuned CGA and PT algorithms in section 5 of the report.                 
"""

import sys; sys.path.append('..')
from src.utils.helper_functions import generate_initial
from src.algorithms.CGA.CGA import ContinousGeneticAlgorithm
from src.algorithms.PT.PT import ParallelTempering
from src.functions import KBF_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import pandas as pd

# Generate initial solutions for both functions. 
# No solution has a function value > 0.3, (away from global optimum)
initial_pop_list, seeds = generate_initial(x_dim=8, pop_size=250)

# Max number of iterations, as required by assignment handout
MAX_NUM_ITERS = 10000

# Convergence criteria
eps = 0.00025
conv_iters = 1300 

### CGA Gather Results ###
CGA_AvgFit_ALL = np.zeros((50, MAX_NUM_ITERS))
CGA_MinFit_ALL = np.zeros((50, MAX_NUM_ITERS))
CGA_times_ALL = np.zeros(50)
CGA_best_Xs = np.zeros((50, 8))
CGA_converg_iters = np.zeros(50)

# Run CGA 50 times with different initialisations
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
    avg_fitness = np.zeros(MAX_NUM_ITERS)
    min_fitness = np.zeros(MAX_NUM_ITERS)

    # Max number of iterations = MAX_NUM_ITERS
    conv_count = 0
    CGA_tic = time()
    for i in range(MAX_NUM_ITERS):
        
        # Evolve population
        CGA.evolve()

        # Update fitness arrays
        avg_fitness[i] = np.mean(CGA.fitness)
        min_fitness[i] = CGA.min_fitness

        # Check if the algorithm has converged, |f(x) - f(x_prev)| < eps for 'conv_iters' iterations
        if i != 0 and np.linalg.norm(min_fitness[i] - min_fitness[i-1]) < eps:
            conv_count += 1
        else:
            conv_count = 0

        # If the algorithm has converged, store the iteration at which it converged
        if conv_count == conv_iters:
            CGA_converg_iters[run_iter] = i - conv_iters

    # Time taken to run algorithm
    CGA_toc = time()
    CGA_times_ALL[run_iter] = CGA_toc - CGA_tic
    
    # Update arrays
    CGA_AvgFit_ALL[run_iter, :] = avg_fitness
    CGA_MinFit_ALL[run_iter, :] = min_fitness
    CGA_best_Xs[run_iter, :] = CGA.best_individual

### PT Gather Results ###
PT_AvgFit_ALL = np.zeros((50, MAX_NUM_ITERS))
PT_MinFit_ALL = np.zeros((50, MAX_NUM_ITERS))
PT_times_ALL = np.zeros(50)
PT_best_Xs = np.zeros((50, 8))
PT_converg_iters = np.zeros(50)

# Run PT 50 times with different initialisations
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
    avg_fitness = np.zeros(MAX_NUM_ITERS)
    min_fitness = np.zeros(MAX_NUM_ITERS)

    # Max number of iterations = MAX_NUM_ITERS
    conv_count = 0
    PT_tic = time()
    for i in range(MAX_NUM_ITERS):
        
        # Algorithm update
        PT.update_chains()
        PT.replica_exchange(iter)

        # Update fitness arrays
        avg_fitness[i], min_fitness[i] = PT.get_fitness()

        # Check if the algorithm has converged, |f(x) - f(x_prev)| < eps for 'conv_iters' iterations
        if i != 0 and np.linalg.norm(min_fitness[i] - min_fitness[i-1]) < eps:
            conv_count += 1
        else:
            conv_count = 0

        # If the algorithm has converged, store the iteration at which it converged
        if conv_count == conv_iters:
            PT_converg_iters[run_iter] = i - conv_iters

    # Time taken to run algorithm
    PT_toc = time()
    PT_times_ALL[run_iter] = PT_toc - PT_tic

    # Update arrays
    PT_AvgFit_ALL[run_iter, :] = avg_fitness
    PT_MinFit_ALL[run_iter, :] = min_fitness
    PT_best_Xs[run_iter, :] = PT.get_best_solution()

# Find the expectation of the average and min fitness across all 50 initialisations
CGA_AvgFit_mean = np.mean(CGA_AvgFit_ALL, axis=0)
PT_AvgFit_mean = np.mean(PT_AvgFit_ALL, axis=0)
CGA_MinFit_mean = np.mean(CGA_MinFit_ALL, axis=0)
PT_MinFit_mean = np.mean(PT_MinFit_ALL, axis=0)

# Find the expected iteration at which CGA and PT converge
CGA_Avg_i = np.mean(CGA_converg_iters)
PT_Avg_i = np.mean(PT_converg_iters)

# Plot expected average fitness values for CGA and PT
plt.figure()
sns.set_style('darkgrid')
plt.plot(CGA_AvgFit_mean, label='CGA', color='green')
plt.plot(PT_AvgFit_mean, label='PT', color='red')
plt.axvline(CGA_Avg_i, color='green', linestyle='--', label=f'CGA Best Converges at {int(CGA_Avg_i)} Iterations')
plt.axvline(PT_Avg_i, color='red', linestyle='--', label=f'PT Best Convergence at {int(PT_Avg_i)} Iterations')
plt.xlabel('Iterations')
plt.ylabel('Average Fitness')
plt.title('Expected Average Fitness across 50 Different Initialisations')
plt.legend()
plt.savefig('figures/Final Comparison/CGA vs PT Average Fitness.png', dpi=300)

# Plot expercted min fitness values for CGA and PT 
plt.figure()
sns.set_style('darkgrid')
plt.plot(CGA_MinFit_mean, label='CGA', color='green')
plt.plot(PT_MinFit_mean, label='PT', color='red')
plt.axvline(CGA_Avg_i, color='green', linestyle='--', label=f'CGA Best Converges at {int(CGA_Avg_i)} Iterations on Avg.')
plt.axvline(PT_Avg_i, color='red', linestyle='--', label=f'PT Best Convergence at {int(PT_Avg_i)} Iterations on Avg.')
plt.xlabel('Iterations')
plt.ylabel('Minimum Fitness')
plt.title('Expected Minimum Fitness across 50 Different Initialisations')
plt.legend()
plt.savefig('figures/Final Comparison/CGA vs PT Minimum Fitness.png', dpi=300)

# Save final results to csv
CGA_final_avg = CGA_AvgFit_mean[-1]
CGA_final_std = np.std(CGA_AvgFit_ALL[:, -1])
CGA_final_min = CGA_MinFit_mean[-1]
CGA_final_min_std = np.std(CGA_MinFit_ALL[:, -1])
CGA_final_time = np.mean(CGA_times_ALL)
CGA_final_time_std = np.std(CGA_times_ALL)
CGA_AVG_best_X = np.mean(CGA_best_Xs, axis=0)

PT_final_avg = PT_AvgFit_mean[-1]
PT_final_std = np.std(PT_AvgFit_ALL[:, -1])
PT_final_min = PT_MinFit_mean[-1]
PT_final_min_std = np.std(PT_MinFit_ALL[:, -1])
PT_final_time = np.mean(PT_times_ALL)
PT_final_time_std = np.std(PT_times_ALL)
PT_AVG_best_X = np.mean(PT_best_Xs, axis=0)

data = {'CGA': [CGA_final_avg, CGA_final_std, CGA_final_min, CGA_final_min_std, CGA_final_time, CGA_final_time_std, CGA_Avg_i, CGA_AVG_best_X],
        'PT': [PT_final_avg, PT_final_std, PT_final_min, PT_final_min_std, PT_final_time, PT_final_time_std, PT_Avg_i, PT_AVG_best_X]}
df = pd.DataFrame(data, index=['Final Avg. Fitness', 'Final Avg. Fitness Std', 'Final Min. Fitness', 'Final Min. Fitness Std', 'Total Time Taken', 'Time Taken Std', 'Mean Iters to Convergence', 'Mean Best Solution'])
df.to_csv('figures/Final Comparison/CGA vs PT Final Results.csv')

print(seeds) # These are the random seeds used to generate the initial populations
