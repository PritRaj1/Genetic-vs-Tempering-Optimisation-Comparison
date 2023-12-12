"""
Candidate No : 5730E, Module: 4M17 

Description :
    This parallelised script serves as a platform to run multiple simulations of the CGA algorithm
    to assess the impact of different hyperparameters on the algorithm's performance. The results
    are saved to a csv file and figures are generated to visualise the results.
"""

import sys; sys.path.append('..')
from src.utils.helper_functions import evaluate_2D, create_figure_directories
from src.utils.plotting_functions import plot_2D, plot_grey_contour, plot_population, plot_fitness
from src.functions import KBF_function, Rosenbrock_function
from src.algorithms.CGA.CGA import ContinousGeneticAlgorithm

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
import seaborn as sns

# Hyperparameters
X_RANGE=(0,10)
FUNCTION = KBF_function
POPULATION_SIZE = 250
CHROMOSOME_LENGTH = 2
NUM_PARENTS =  POPULATION_SIZE // 4 # 25% of population size
MUTATION_RATE_LIST = [0.05]
CROSSOVER_PROB_LIST = [0.7]
SELECTION_METHOD_LIST = ['Proportional', 'Tournament', 'SRS']
MATING_PROCEDURE_LIST = ['Crossover', 'Heuristic Crossover']
ITERS_LIST = [10, 100]
TOUNAMENT_SIZE = POPULATION_SIZE // 4 # 25% of population size

NAME = 'Rosenbrock' if FUNCTION == Rosenbrock_function else 'KBF'

# Make sure NUM_PARENTS is a multiple of 2
if NUM_PARENTS % 2 != 0:
    NUM_PARENTS += 1

# Create directories for figures
create_figure_directories(NAME, SELECTION_METHOD_LIST, MATING_PROCEDURE_LIST, ITERS_LIST)

# 2D Visualisation
X1, X2, f = evaluate_2D(FUNCTION, x_range=X_RANGE)
plot_2D(X1, X2, f, name=NAME)

# Visualise with carved out feasible region
X1, X2, f_feasible = evaluate_2D(FUNCTION, x_range=X_RANGE, constraints=True)
plot_2D(X1, X2, f_feasible, name=NAME, constraints=True)

def run_simulation(params):
    SELECTION_METHOD, MATING_PROCEDURE, MUTATION_RATE, CROSSOVER_PROB, NUM_ITERS = params

    # Instantiate CGA
    CGA = ContinousGeneticAlgorithm(
        population_size=POPULATION_SIZE,
        chromosome_length=CHROMOSOME_LENGTH,
        num_parents=NUM_PARENTS,
        objective_function=FUNCTION,
        tournament_size=TOUNAMENT_SIZE,
        range=X_RANGE,
        mutation_rate=MUTATION_RATE,
        crossover_prob=CROSSOVER_PROB,
        selection_method=SELECTION_METHOD,
        mating_procedure=MATING_PROCEDURE,
    )

    # Evaluate initial population
    CGA.evaluate_fitness()

    # Make a grid of plots to show evolution of population
    PLOT_EVERY = NUM_ITERS // 5
    num_plots = (NUM_ITERS // PLOT_EVERY)
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 5))
    fig.suptitle("Evolution of Population, \n" 
                + r"[Selection: \textbf{" + SELECTION_METHOD 
                + r"}, Mating: \textbf{" + MATING_PROCEDURE 
                +  r"}, Mutation Rate: \textbf{" + str(MUTATION_RATE) 
                + r"}, Crossover Prob: \textbf{" + str(CROSSOVER_PROB) 
                + r"}]", fontsize=18)

    # Plot grey contours of function on each subplot in grid
    # This will be overlayed with populations at different iterations
    for idx in range(num_plots):
        plot_grey_contour(X1, X2, f_feasible, plot=axs[idx], x_range=X_RANGE)

    # Initialise arrays to store fitness values
    avg_fitness = np.zeros(NUM_ITERS)
    min_fitness = np.zeros(NUM_ITERS)

    for iter in range(NUM_ITERS):
        # Overlay population on grey contour every "PLOT_EVERY" iterations
        if iter % PLOT_EVERY == 0:
            plot_num = (iter // PLOT_EVERY)
            idx = plot_num % num_plots
            axs[idx].set_title(f'Iteration: {iter}')
            plot_population(CGA.population, plot=axs[idx], best=CGA.best_individual)

        # Evolve population
        CGA.evolve()

        # Update fitness arrays
        avg_fitness[iter] = np.mean(CGA.fitness)
        min_fitness[iter] = CGA.min_fitness

    plt.tight_layout()
    plt.savefig(f'figures/{NAME}/{str(NUM_ITERS)}_iters/{SELECTION_METHOD}/{MATING_PROCEDURE}/{MUTATION_RATE}_{CROSSOVER_PROB}_Population.png')

    # Plot fitness evolution with iteration
    plot_fitness(avg_fitness, min_fitness, type=(NAME, NUM_ITERS, SELECTION_METHOD, MATING_PROCEDURE, MUTATION_RATE, CROSSOVER_PROB))

    # Save results to global dataframe
    return {
        'Selection Method': SELECTION_METHOD,
        'Mating Procedure': MATING_PROCEDURE,
        'Mutation Rate': MUTATION_RATE,
        'Crossover Rate': CROSSOVER_PROB,
        'Iterations': NUM_ITERS,
        'Final Avg Fitness': avg_fitness[-1],
        'Final Min Fitness': min_fitness[-1]
    } 

# Create a list of parameter combinations
params_list = [(SELECTION_METHOD, MATING_PROCEDURE, MUTATION_RATE, CROSSOVER_PROB, NUM_ITERS)
               for SELECTION_METHOD in SELECTION_METHOD_LIST
               for MATING_PROCEDURE in MATING_PROCEDURE_LIST
               for MUTATION_RATE in MUTATION_RATE_LIST
               for CROSSOVER_PROB in CROSSOVER_PROB_LIST
               for NUM_ITERS in ITERS_LIST]

print('Starting simulations for table and figure generation...')

# Create a pool of worker processes
pool = Pool()

global results
results = pd.DataFrame(columns=['Selection Method', 
                                'Mating Procedure', 
                                'Mutation Rate', 
                                'Crossover Rate', 
                                'Iterations', 
                                'Final Avg Fitness', 
                                'Final Min Fitness'], index=None)

# Run the simulations in parallel
for result in pool.map(run_simulation, params_list):
    results.loc[len(results)] = result

# Close the pool
pool.close()
pool.join()

# Save results to csv file
results
results.to_csv(f'figures/{NAME}/CGAresults.csv')

print('Done!')

print('Starting simulations for fitness plots generation')

MATING = 'Heuristic Crossover' # Mating procedure held constant

# Make one plot to show fitness evolution for each selection method
plt.figure(figsize=(14, 10))
sns.set_style('darkgrid')
plt.title(f'Average Fitness Evolution for each Selection Method on {NAME} Function \n' 
          + r"Mating Procdure held as \textbf{" + MATING + r"}", fontsize=24)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Average Fitness', fontsize=20)

for SELECTION_METHOD in SELECTION_METHOD_LIST:
    # Instantiate CGA
    CGA = ContinousGeneticAlgorithm(
        population_size=POPULATION_SIZE,
        chromosome_length=CHROMOSOME_LENGTH,
        num_parents=NUM_PARENTS,
        objective_function=FUNCTION,
        tournament_size=TOUNAMENT_SIZE,
        range=X_RANGE,
        mutation_rate=0.05,
        crossover_prob=0.7,
        selection_method=SELECTION_METHOD,
        mating_procedure=MATING,
        )
    
    # Evaluate initial population
    CGA.evaluate_fitness()

    # Initialise arrays to store fitness values
    avg_fitness = np.zeros(100)
    min_fitness = np.zeros(100)

    for iter in range(100):
        # Evolve population
        CGA.evolve()

        # Update fitness arrays
        avg_fitness[iter] = np.mean(CGA.fitness)
        min_fitness[iter] = CGA.min_fitness

    # Plot fitness evolution with iteration
    sns.lineplot(x=range(100), y=avg_fitness, label=SELECTION_METHOD)

plt.legend(fontsize=16)
plt.savefig(f'figures/{NAME}/Fitness_Evolution_{MATING}.png')



    