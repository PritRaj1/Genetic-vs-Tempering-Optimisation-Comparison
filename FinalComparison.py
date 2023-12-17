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

MAX_NUM_ITERS = 25000

### CGA Gather Results ###
CGA_AvgFit_ALL = np.zeros((50, MAX_NUM_ITERS))
CGA_MinFit_ALL = np.zeros((50, MAX_NUM_ITERS))
CGA_times_ALL = np.zeros(50)

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
    CGA_tic = time()
    for i in range(MAX_NUM_ITERS):
        
        # Evolve population
        CGA.evolve()

        # Update fitness arrays
        avg_fitness[i] = np.mean(CGA.fitness)
        min_fitness[i] = CGA.min_fitness

    # Time taken to run algorithm
    CGA_toc = time()
    CGA_times_ALL[run_iter] = CGA_toc - CGA_tic
    
    # Update arrays
    CGA_AvgFit_ALL[run_iter, :] = avg_fitness
    CGA_MinFit_ALL[run_iter, :] = min_fitness

### PT Gather Results ###
PT_AvgFit_ALL = np.zeros((50, MAX_NUM_ITERS))
PT_MinFit_ALL = np.zeros((50, MAX_NUM_ITERS))
PT_times_ALL = np.zeros(50)

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
    PT_tic = time()
    for i in range(MAX_NUM_ITERS):
        
        # Algorithm update
        PT.update_chains()
        PT.replica_exchange(iter)

        # Update fitness arrays
        avg_fitness[i], min_fitness[i] = PT.get_fitness()

    # Time taken to run algorithm
    PT_toc = time()
    PT_times_ALL[run_iter] = PT_toc - PT_tic

    # Update arrays
    PT_AvgFit_ALL[run_iter, :] = avg_fitness
    PT_MinFit_ALL[run_iter, :] = min_fitness

# Find the expectation of the average and min fitness across all 50 initialisations
CGA_AvgFit_mean = np.mean(CGA_AvgFit_ALL, axis=0)
PT_AvgFit_mean = np.mean(PT_AvgFit_ALL, axis=0)
CGA_MinFit_mean = np.mean(CGA_MinFit_ALL, axis=0)
PT_MinFit_mean = np.mean(PT_MinFit_ALL, axis=0)

# Find the iteration at which CGA and PT converge (when the final min fitness is first reached by the min fitness)
CGA_Avg_i = np.where(CGA_AvgFit_mean == np.min(CGA_AvgFit_mean))[0][0]
PT_Avg_i = np.where(PT_AvgFit_mean == np.min(PT_AvgFit_mean))[0][0]

# Plot expected average fitness values for CGA and PT
plt.figure()
sns.set_style('darkgrid')
plt.plot(CGA_AvgFit_mean, label='CGA', color='green')
plt.plot(PT_AvgFit_mean, label='PT', color='red')
plt.axvline(CGA_Avg_i, color='green', linestyle='--', label=f'CGA Converges at {int(CGA_Avg_i)} Iterations')
plt.axvline(PT_Avg_i, color='red', linestyle='--', label=f'PT Convergence at {int(PT_Avg_i)} Iterations')
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

PT_final_avg = PT_AvgFit_mean[-1]
PT_final_std = np.std(PT_AvgFit_ALL[:, -1])
PT_final_min = PT_MinFit_mean[-1]
PT_final_min_std = np.std(PT_MinFit_ALL[:, -1])
PT_final_time = np.mean(PT_times_ALL)
PT_final_time_std = np.std(PT_times_ALL)

data = {'CGA': [CGA_final_avg, CGA_final_std, CGA_final_min, CGA_final_min_std, CGA_final_time, CGA_final_time_std, CGA_Avg_i],
        'PT': [PT_final_avg, PT_final_std, PT_final_min, PT_final_min_std, PT_final_time, PT_final_time_std, PT_Avg_i]}
df = pd.DataFrame(data, index=['Average Fitness', 'Average Fitness Std', 'Minimum Fitness', 'Minimum Fitness Std', 'Total Time Taken', 'Time Taken Std', 'Iterations to Convergence'])
df.to_csv('figures/Final Comparison/CGA vs PT Final Results.csv')

print(seeds) # These are the random seeds used to generate the initial populations
