import sys; sys.path.append('..')
from src.utils.helper_functions import evaluate_2D
from src.utils.plotting_functions import plot_2D, plot_grey_contour, plot_population, plot_fitness
from src.functions import KBF_function, Rosenbrock_function
from src.algorithms.CGA.CGA import ContinousGeneticAlgorithm

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Hyperparameters
X_RANGE=(0,10)
FUNCTION = KBF_function
POPULATION_SIZE = 250
CHROMOSOME_LENGTH = 2
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.8
SELECTION_METHOD = 'Tournament' # 'Proportional', 'Tournament', 'SRS'
MATING_PROCEDURE = 'Blending' # 'Crossover', 'Blending'
NUM_ITERS = 10

NAME = 'Rosenbrock' if FUNCTION == Rosenbrock_function else 'KBF'

# 2D Visualisation
X1, X2, f = evaluate_2D(FUNCTION, x_range=X_RANGE)
plot_2D(X1, X2, f, name=NAME, x_range=X_RANGE)

# Instantiate CGA
CGA = ContinousGeneticAlgorithm(
    population_size=POPULATION_SIZE,
    chromosome_length=CHROMOSOME_LENGTH,
    objective_function=FUNCTION,
    range=X_RANGE,
    mutation_rate=MUTATION_RATE,
    crossover_rate=CROSSOVER_RATE,
    selection_method=SELECTION_METHOD,
    mating_procedure=MATING_PROCEDURE,
)

# Evaluate initial population
CGA.evaluate_fitness()

# Make a grid of plots to show evolution of population
PLOT_EVERY = NUM_ITERS // 5
num_plots = (NUM_ITERS // PLOT_EVERY)
fig, axs = plt.subplots(1, num_plots, figsize=(20, 5))
fig.suptitle("Evolution of Population, \n" + r"[Selection: \textbf{" + SELECTION_METHOD + r"}, Mating: \textbf{" + MATING_PROCEDURE +  "}]", fontsize=22)

# Plot grey contours of function on each subplot in grid
# This will be overlayed with populations at different iterations
for idx in range(num_plots):
    plot_grey_contour(X1, X2, f, plot=axs[idx], x_range=X_RANGE)

# Initialise arrays to store fitness values
avg_fitness = np.zeros(NUM_ITERS)
min_fitness = np.zeros(NUM_ITERS)

# tqdm bar
tqdm_iter = tqdm(range(NUM_ITERS))

for iter in tqdm_iter:

    # Overlay population on grey contour every "PLOT_EVERY" iterations
    if iter % PLOT_EVERY == 0:
        plot_num = (iter // PLOT_EVERY)
        idx = plot_num % num_plots
        axs[idx].set_title(f'Iteration: {iter}')

        plot_population(X1, X2, f, CGA.population, plot=axs[idx], best=CGA.best_individual, range=X_RANGE)

    # Evolve population
    CGA.evolve()

    # Update fitness arrays
    avg_fitness[iter] = np.mean(CGA.fitness)
    min_fitness[iter] = np.min(CGA.fitness)

    # Update tqdm description
    tqdm_iter.set_description(f"Average Fitness: {avg_fitness[iter]:.2f}, Minimum Fitness: {min_fitness[iter]:.2f}")

plt.tight_layout()
plt.savefig(f'figures/{NAME}_{SELECTION_METHOD}_{MATING_PROCEDURE}_Evolution.png')

# Plot fitness evolution with iteration
plot_fitness(avg_fitness, min_fitness, type=(NAME, SELECTION_METHOD, MATING_PROCEDURE))


