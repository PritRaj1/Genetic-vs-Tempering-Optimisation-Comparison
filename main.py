import sys; sys.path.append('..')
from src.utils.helper_functions import evaluate_2D
from src.utils.plotting_functions import plot_2D, plot_grey_contour, plot_population, plot_fitness
from src.functions import KBF_function, Rosenbrock_function
from src.algorithms.CGA.CGA import ContinousGeneticAlgorithm

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

X_RANGE=(0,10)
FUNCTION = KBF_function
POPULATION_SIZE = 250
CHROMOSOME_LENGTH = 2
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
SELECTION_METHOD = 'Tournament' # 'Proportional', 'Tournament', 'SRS'
MATING_PROCEDURE = 'Crossover' # 'Crossover', 'Blending'
NUM_ITERS = 5

NAME = 'Rosenbrock' if FUNCTION == Rosenbrock_function else 'KBF'

X1, X2, f = evaluate_2D(FUNCTION, x_range=X_RANGE)
plot_2D(X1, X2, f, name=NAME, x_range=X_RANGE)

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

CGA.evaluate_fitness()

PLOT_EVERY = NUM_ITERS // 5
num_plots = (NUM_ITERS // PLOT_EVERY)

# Create a figure with subplots
fig, axs = plt.subplots(1, num_plots, figsize=(20, 5))
fig.suptitle("Evolution of Population, \n" + r"[Selection: \textbf{" + SELECTION_METHOD + r"}, Mating: \textbf{" + MATING_PROCEDURE +  "}]", fontsize=22)

# Plot grid of grey contours which will be overlayed with population
for idx in range(num_plots):
    plot_grey_contour(X1, X2, f, plot=axs[idx], x_range=X_RANGE)

avg_fitness = np.zeros(NUM_ITERS)
min_fitness = np.zeros(NUM_ITERS)

tqdm_iter = tqdm(range(NUM_ITERS))

for iter in tqdm_iter:

    # Overlay population on grey contour
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
plt.savefig(f'figures/{NAME}.png')

# Plot fitness evolution
plot_fitness(avg_fitness, min_fitness, name='NAME', type=(SELECTION_METHOD, MATING_PROCEDURE))


