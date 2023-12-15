import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import pandas as pd

from src.algorithms.PT.PT import ParallelTempering
from src.functions import KBF_function, Rosenbrock_function
from src.utils.helper_functions import evaluate_2D, create_figure_directories_PT
from src.utils.plotting_functions import visualise_schedule, plot_sub_contour, plot_population, plot_fitness, plot_temp_progressions

# Hyperparameters
X_RANGE=(0,10)
FUNCTION = KBF_function
X_DIM = 2
NUM_REPLICAS = 10
NUM_SOL_PER_REPLICA = 250 // NUM_REPLICAS # 250 solutions overall, same as CGA population
EXCHANGE_PROCEDURE_LIST = ['Periodic', 'Stochastic']
EXCHANGE_PARAM_LIST = [0.1, 0.3, 0.5]
TEMP_TYPE = 'Power'
PROGRESSION_POWER_LIST = [1, 3, 5] # 1 is uniform
NUM_ITERS_LIST = [100]

FUNC_NAME = 'Rosenbrock' if FUNCTION == Rosenbrock_function else 'KBF'

create_figure_directories_PT(FUNC_NAME, EXCHANGE_PROCEDURE_LIST, [TEMP_TYPE], NUM_ITERS_LIST)

def PT_initial_tuning(params):

    # Parse parameters
    EXCHANGE_PROCEDURE, EXCHANGE_PARAM, PROGRESSION_POWER, NUM_ITERS = params 

    # Set schedule name
    SCHEDULE_NAME = TEMP_TYPE + " " + str(PROGRESSION_POWER) if TEMP_TYPE == 'Power' else TEMP_TYPE
    # Instantiate PT object
    PT = ParallelTempering(
        objective_function=FUNCTION,
        x_dim=X_DIM,
        range=X_RANGE,
        num_replicas=NUM_REPLICAS,
        num_chains=NUM_SOL_PER_REPLICA,
        exchange_procedure=EXCHANGE_PROCEDURE,
        exchange_param=EXCHANGE_PARAM,
        schedule_type=TEMP_TYPE,
        power_term=PROGRESSION_POWER
    )
    
    visualise_schedule(PT.temperature_schedule, FUNCTION, X_RANGE, SCHEDULE_NAME, FUNC_NAME)

    # Make a grid of 5 2D plots to show evolution of population
    PLOT_EVERY = NUM_ITERS // 5
    num_plots = (NUM_ITERS // PLOT_EVERY)
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 5))
    plt.suptitle(f"Evolution of Solutions, \n"
                + r"[Exchange Procedure: \textbf{" + EXCHANGE_PROCEDURE
                + r"}, Exchange Parameter: \textbf{" + str(EXCHANGE_PARAM)
                + r"}, Schedule: \textbf{" + TEMP_TYPE
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

    for iter in range(NUM_ITERS):
        # Overlay population on grey contour every "PLOT_EVERY" iterations. (Periodic break to allow for visualisation).
        if iter % PLOT_EVERY == 0:
            plot_num = (iter // PLOT_EVERY)
            idx = plot_num % num_plots
            axs[idx].set_title(f'Iteration: {iter}')
            final_replica = PT.get_all_solutions()
            plot_population(final_replica, axs[idx], best=PT.get_best_solution(), last=final_replica[-1])

        PT.update_chains()
        PT.replica_exchange(iter)

        # Update fitness arrays
        avg_fitness[iter], min_fitness[iter] = PT.get_fitness()

    plt.tight_layout()
    plt.savefig(f'figures/{FUNC_NAME}/{str(NUM_ITERS)}_iters/{EXCHANGE_PROCEDURE}/{TEMP_TYPE}/{EXCHANGE_PARAM}_{PROGRESSION_POWER}_{PROGRESSION_POWER}_Solutions.png')

    plot_fitness(avg_fitness, min_fitness, [FUNC_NAME, 
                                            NUM_ITERS,
                                            EXCHANGE_PROCEDURE, 
                                            TEMP_TYPE, 
                                            EXCHANGE_PARAM, 
                                            PROGRESSION_POWER], PT=True)

    # Return results to dataframe
    return {
        'Exchange Procedure': EXCHANGE_PROCEDURE,
        'Exchange Parameter': EXCHANGE_PARAM,
        'Schedule': TEMP_TYPE,
        'Power Term': PROGRESSION_POWER,
        'Iterations': NUM_ITERS,
        'Final Avg. Fitness': avg_fitness[-1],
        'Final Min. Fitness': min_fitness[-1]
    }

# Create a list of all combinations of parameters
params_list = [(EXCHANGE_PROCEDURE, EXCHANGE_PARAM, PROGRESSION_POWER, NUM_ITERS) 
                for EXCHANGE_PROCEDURE in EXCHANGE_PROCEDURE_LIST
                for EXCHANGE_PARAM in EXCHANGE_PARAM_LIST
                for PROGRESSION_POWER in PROGRESSION_POWER_LIST
                for NUM_ITERS in NUM_ITERS_LIST]

print('Starting simulations for table and figure generation...')

# Create a pool of worker processes
pool = Pool()

results = pd.DataFrame(columns=['Exchange Procedure', 
                                'Exchange Parameter', 
                                'Schedule', 
                                'Power Term',
                                'Iterations', 
                                'Final Avg. Fitness', 
                                'Final Min. Fitness'])

# Run the simulations in parallel
for result in pool.map(PT_initial_tuning, params_list):
    results.loc[len(results)] = result # Append result to dataframe

# Save results to csv
results.to_csv(f'figures/{FUNC_NAME}/PT_initial_tuning.csv', index=False)

# Close the pool of workers
pool.close()
pool.join()
print('Done!')


