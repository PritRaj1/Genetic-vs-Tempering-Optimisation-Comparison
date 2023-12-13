import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.PT.PT import ParallelTempering
from src.functions import KBF_function, Rosenbrock_function
from src.utils.helper_functions import evaluate_2D, create_figure_directories_PT
from src.utils.plotting_functions import visualise_schedule, plot_sub_contour, plot_population, plot_fitness

from tqdm import tqdm

# Hyperparameters
X_RANGE=(0,10)
FUNCTION = KBF_function
X_DIM = 2
NUM_REPLICAS = 10
NUM_SOL_PER_REPLICA = 10 # 250 // NUM_REPLICAS # 250 solutions overall, same as CGA population
EXCHANGE_PROCEDURE = 'Periodic'
EXCHANGE_PARAM = 0.2
TEMP_TYPE = 'Power'
PROGRESSION_POWER = 1
NUM_ITERS = 200

FUNC_NAME = 'Rosenbrock' if FUNCTION == Rosenbrock_function else 'KBF'
SCHEDULE_NAME = TEMP_TYPE + " " + str(PROGRESSION_POWER) if TEMP_TYPE == 'Power' else TEMP_TYPE

EXCHANGE_PROCEDURE_LIST = ['Periodic', 'Stochastic']
SCHEDULE_TYPE_LIST = ['Power']

create_figure_directories_PT(FUNC_NAME, EXCHANGE_PROCEDURE_LIST, SCHEDULE_TYPE_LIST, [NUM_ITERS])

# Instantiate PT object
PT = ParallelTempering(
    objective_function=FUNCTION,
    x_dim=X_DIM,
    range=X_RANGE,
    num_replicas=NUM_REPLICAS,
    num_x_per_replica=NUM_SOL_PER_REPLICA,
    exchange_procedure=EXCHANGE_PROCEDURE,
    exchange_param=EXCHANGE_PARAM,
    schedule_type=TEMP_TYPE,
    power_term=PROGRESSION_POWER,
    total_iterations=NUM_ITERS
)

visualise_schedule(PT.temperature_schedule, FUNCTION, X_RANGE, SCHEDULE_NAME, FUNC_NAME)

# Make a grid of 5 2D plots to show evolution of population
PLOT_EVERY = NUM_ITERS // 5
num_plots = (NUM_ITERS // PLOT_EVERY)
fig, axs = plt.subplots(1, num_plots, figsize=(20, 5))
plt.suptitle(f"Evolution of Solutions, \n"
             + r"[Exchange Procedure: \textbf{" + EXCHANGE_PROCEDURE
                + r"}, Exchange Parameter: \textbf{" + str(EXCHANGE_PARAM)
                + r"}, Schedule: \textbf{" + SCHEDULE_NAME
                + r"}, Power Term: \textbf{" + str(PROGRESSION_POWER)
                + r"}]", fontsize=16)

# Plot grey contour plots of function on each subplot in grid
# This will be overlayed with populations at different iterations
X1, X2, f_feasible = evaluate_2D(FUNCTION, x_range=X_RANGE, constraints=True)
for idx in range(num_plots):
    plot_sub_contour(X1, X2, f_feasible, plot=axs[idx], x_range=X_RANGE)

# Initialise arrays to store fitness values
avg_fitness = np.zeros(NUM_ITERS)
min_fitness = np.zeros(NUM_ITERS)

# Initial avg. and min. fitness
avg_fitness[0], min_fitness[0] = PT.get_fitness()
print(f"Initial average fitness: {avg_fitness[0]}")
print(f"Initial minimum fitness: {min_fitness[0]}")

for iter in tqdm(range(NUM_ITERS)):
    # Overlay population on grey contour every "PLOT_EVERY" iterations. (Periodic break to allow for visualisation).
    if iter % PLOT_EVERY == 0:
        plot_num = (iter // PLOT_EVERY)
        idx = plot_num % num_plots
        axs[idx].set_title(f'Iteration: {iter}')
        final_replica = PT.current_solutions[-1]
        plot_population(final_replica, axs[idx], best=PT.get_best_solution(), last=final_replica[-1])
    
    PT.update_solutions()
    PT.replica_exchange(iter)

    # Update fitness arrays
    avg_fitness[iter], min_fitness[iter] = PT.get_fitness()

plt.tight_layout()
plt.savefig(f'figures/{FUNC_NAME}/{str(NUM_ITERS)}_iters/{EXCHANGE_PROCEDURE}/{TEMP_TYPE}/{EXCHANGE_PARAM}_{PROGRESSION_POWER}_{PROGRESSION_POWER}_Solutions.png')

print(f"Average fitness: {avg_fitness[-1]}")
print(f"Minimum fitness: {min_fitness[-1]}")

plot_fitness(avg_fitness, min_fitness, [FUNC_NAME, 
                                        NUM_ITERS,
                                        EXCHANGE_PROCEDURE, 
                                        TEMP_TYPE, 
                                        EXCHANGE_PARAM, 
                                        PROGRESSION_POWER])


