"""
Candidate No : 5730E, Module: 4M17 

Description :
    This file serves as a platform to run multiple simulations of the CGA algorithm.
    Used to generate the results for the table and figures in Section 3.2 of the report.


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
MUTATION_RATE_LIST = [0.05, 0.1]
CROSSOVER_PROB_LIST = [0.7, 0.65]
SELECTION_METHOD_LIST = ['Proportional', 'Tournament', 'SRS']
MATING_PROCEDURE_LIST = ['Crossover', 'Heuristic Crossover']
ITERS_LIST = [5, 100]
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

def selection_mating_tuning(params):
    """
    Parallelisable function to run multiple simulations of the CGA algorithm and 
    assess the impact of different hyperparameters on the algorithm's performance.
    The results are saved to a csv file and figures are generated to visualise the results.
    """

    # Parse parameters
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

def plot_fitnesses():
    """
    Function to plot fitness evolution for each selection method.
    """
    for MATING in MATING_PROCEDURE_LIST:

        # Make a plot for each mating procedure to show fitness evolution for the three selection method
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

# Define a function for parallel execution
def rate_prob_contour(params):
    """
    Parallelisable function to create contour plots to assess
    the impact of mutation rate and crossover probability on the
    algorithm's performance.
    """

    i, j, MUTATION_RATE, CROSSOVER_PROB = params

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

    # Evolve population
    for iter in range(100):
        CGA.evolve()

    # Return the result
    return i, j, np.mean(CGA.fitness), CGA.min_fitness

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

results = pd.DataFrame(columns=['Selection Method', 
                                'Mating Procedure', 
                                'Mutation Rate', 
                                'Crossover Rate', 
                                'Iterations', 
                                'Final Avg Fitness', 
                                'Final Min Fitness'], index=None)

# Run the simulations in parallel
for result in pool.map(selection_mating_tuning, params_list):
    results.loc[len(results)] = result

# Close the pool
pool.close()
pool.join()

# Save results to csv file
results
results.to_csv(f'figures/{NAME}/CGAresults.csv')

print('Done!')

print('Starting simulations for fitness plots generation')

# Plot fitness evolution for each selection method
plot_fitnesses()

print('Done!')

print('Starting simulations for contour plot generation')

# Selection and mating chosen as Tournament and Heuristic Crossover respectively, given results from report
SELECTION_METHOD = 'Tournament'
MATING_PROCEDURE = 'Heuristic Crossover'
MUTATION_RATE_LIST = [0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
CROSSOVER_PROB_LIST = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# Create plot with X = mutation rate, Y = crossover prob, Z = final avg fitness

# Initialise meshgrids
X, Y = np.meshgrid(MUTATION_RATE_LIST, CROSSOVER_PROB_LIST)

# Initialise array to store final avg and min fitnesses
AVG = np.zeros((len(MUTATION_RATE_LIST), len(CROSSOVER_PROB_LIST)))
MIN = np.zeros((len(MUTATION_RATE_LIST), len(CROSSOVER_PROB_LIST)))

# Create a pool of workers
pool = Pool()

# Create a list of parameters for parallel execution
params_list = [(i, j, MUTATION_RATE, CROSSOVER_PROB) for i, MUTATION_RATE in enumerate(MUTATION_RATE_LIST) for j, CROSSOVER_PROB in enumerate(CROSSOVER_PROB_LIST)]

# Run the simulations in parallel
results = pool.map(rate_prob_contour, params_list)

# Update fitness arrays
for i, j, fitness, min_fitness in results:
    AVG[i, j] = fitness
    MIN[i, j] = min_fitness

# Close the pool
pool.close()
pool.join()

# Plot average fitness contour
plt.figure(figsize=(14, 10))
plt.title(f'Average Final Fitness with varying Mutation Rate and Crossover Probability on {NAME} Function \n' 
        + r"[Selection Method: \textbf{" + SELECTION_METHOD + r"}, "
        + r"Mating Procedure: \textbf{" + MATING_PROCEDURE + r"}]", fontsize=18)
sns.heatmap(AVG, annot=True, xticklabels=MUTATION_RATE_LIST, yticklabels=CROSSOVER_PROB_LIST)
plt.xlabel('Mutation Rate', fontsize=14)
plt.ylabel('Crossover Probability', fontsize=14)
plt.savefig(f'figures/{NAME}/AVGContour_{SELECTION_METHOD}_{MATING_PROCEDURE}.png')

# Plot minimum fitness contour
plt.figure(figsize=(14, 10))
plt.title(f'Minimum Final Fitness with varying Mutation Rate and Crossover Probability on {NAME} Function \n' 
        + r"[Selection Method: \textbf{" + SELECTION_METHOD + r"}, "
        + r"Mating Procedure: \textbf{" + MATING_PROCEDURE + r"}]", fontsize=18)
sns.heatmap(MIN, annot=True, xticklabels=MUTATION_RATE_LIST, yticklabels=CROSSOVER_PROB_LIST)
plt.xlabel('Mutation Rate', fontsize=14)
plt.ylabel('Crossover Probability', fontsize=14)
plt.savefig(f'figures/{NAME}/MINContour_{SELECTION_METHOD}_{MATING_PROCEDURE}.png')


print('Done!')