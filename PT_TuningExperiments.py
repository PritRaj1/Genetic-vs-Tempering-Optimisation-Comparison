import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import pandas as pd
import seaborn as sns

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
    plt.savefig(f'figures/{FUNC_NAME}/{str(NUM_ITERS)}_iters/{EXCHANGE_PROCEDURE}/{TEMP_TYPE}/{EXCHANGE_PARAM}_{PROGRESSION_POWER}_Solutions.png')

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
        'Final Min. Fitness': min_fitness[-1],
        'Avg. Fitness Progression': avg_fitness
    }

def power_exchange_mesh(params):
    # Parse parameters
    i, j, EXCHANGE_PROCEDURE, EXCHANGE_PARAM, PROGRESSION_POWER = params

    PT = ParallelTempering(
        objective_function=FUNCTION,
        x_dim=X_DIM,
        range=X_RANGE,
        num_replicas=NUM_REPLICAS,
        num_chains=NUM_SOL_PER_REPLICA,
        exchange_procedure=EXCHANGE_PROCEDURE,
        exchange_param=EXCHANGE_PARAM,
        schedule_type='Power',
        power_term=PROGRESSION_POWER
    )

    for iter in range(100):
        PT.update_chains()
        PT.replica_exchange(i)

    final_avg_fitness, final_min_fitness = PT.get_fitness()

    return i, j, final_avg_fitness, final_min_fitness

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
                                'Final Min. Fitness',
                                'Avg. Fitness Progression'])

# Run the simulations in parallel
for result in pool.map(PT_initial_tuning, params_list):
    results.loc[len(results)] = result # Append result to dataframe

# Save results to csv
results.to_csv(f'figures/{FUNC_NAME}/PT_initial_tuning.csv', index=False)

# Close the pool of workers
pool.close()
pool.join()
print('Done!')

print('Generating fitness figures...')

for EXCHANGE_PROCEDURE in EXCHANGE_PROCEDURE_LIST:
    plt.figure(figsize=(10, 8))
    sns.set_style('darkgrid')

    # Get results pertaining to that exchange procedure
    periodic_results = results[results['Exchange Procedure'] == EXCHANGE_PROCEDURE]

    # Get results corresponding to uniform schedule
    periodic_results_1 = periodic_results[periodic_results['Power Term'] == 1]

    # Get results correctponding to 'Exchange Parameter' = 0.1, 0.3, 0.5
    periodic_results_1_01 = periodic_results_1[periodic_results_1['Exchange Parameter'] == 0.1]
    periodic_results_1_03 = periodic_results_1[periodic_results_1['Exchange Parameter'] == 0.3]
    periodic_results_1_05 = periodic_results_1[periodic_results_1['Exchange Parameter'] == 0.5]

    # Plot fitness progression for each 'Exchange Parameter'
    plt.plot(periodic_results_1_01['Avg. Fitness Progression'].values[0], label='Exchange Parameter = 0.1')
    plt.plot(periodic_results_1_03['Avg. Fitness Progression'].values[0], label='Exchange Parameter = 0.3')
    plt.plot(periodic_results_1_05['Avg. Fitness Progression'].values[0], label='Exchange Parameter = 0.5')

    plt.title(f'Average Fitness Progression for {EXCHANGE_PROCEDURE} Exchange, \n'
                + r"[Temp Schedule: \textbf{Uniform}]")
    plt.xlabel('Iteration')
    plt.ylabel('Average Fitness')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'figures/{FUNC_NAME}/100_iters/{EXCHANGE_PROCEDURE}/PT_Avg_Fitness_Evolution.png')

print('Done!')  

print('Starting heatmap generation...')

EXCHANGE_PROCEDURE = 'Stochastic'
EXCHANGE_PARAM_LIST = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
PROGRESSION_POWER_LIST = range(1, 8)

X, Y = np.meshgrid(EXCHANGE_PARAM_LIST, PROGRESSION_POWER_LIST)

AVG = np.zeros((len(EXCHANGE_PARAM_LIST), len(PROGRESSION_POWER_LIST)))
MIN = np.zeros((len(EXCHANGE_PARAM_LIST), len(PROGRESSION_POWER_LIST)))

params_list = [(i, j, EXCHANGE_PROCEDURE, EXCHANGE_PARAM, PROGRESSION_POWER)
                for i, EXCHANGE_PARAM in enumerate(EXCHANGE_PARAM_LIST)
                for j, PROGRESSION_POWER in enumerate(PROGRESSION_POWER_LIST)]

# Create a pool of worker processes
pool = Pool()

results = pool.map(power_exchange_mesh, params_list)

for i, j, final_avg_fitness, final_min_fitness in results:
    AVG[i, j] = final_avg_fitness
    MIN[i, j] = final_min_fitness

# Close the pool
pool.close()
pool.join()

# Plot average fitness heat map
plt.figure(figsize=(14, 10))
plt.title(f'Average Final Fitness with varying Power Param and Exchange Param on {FUNC_NAME} Function \n' 
        + r"[Exchange Procedure: \textbf{" + EXCHANGE_PROCEDURE + r"}]", fontsize=18)
sns.heatmap(AVG, annot=True, fmt='.2f', xticklabels=EXCHANGE_PARAM_LIST, yticklabels=PROGRESSION_POWER_LIST, cmap='Greens')
plt.xlabel('Exchange Parameter', fontsize=16)
plt.ylabel('Power Parameter, p', fontsize=16)
plt.tight_layout()
plt.savefig(f'figures/{FUNC_NAME}/PT_Avg_Fitness_Heatmap_{EXCHANGE_PROCEDURE}.png')

# Plot minimum fitness heat map
plt.figure(figsize=(14, 10))
plt.title(f'Minimum Final Fitness with varying Power Param and Exchange Param on {FUNC_NAME} Function \n' 
        + r"[Exchange Procedure: \textbf{" + EXCHANGE_PROCEDURE + r"}]", fontsize=18)
sns.heatmap(MIN, annot=True, fmt='.2f', xticklabels=EXCHANGE_PARAM_LIST, yticklabels=PROGRESSION_POWER_LIST, cmap='Greens')
plt.xlabel('Exchange Parameter', fontsize=16)
plt.ylabel('Power Parameter, p', fontsize=16)
plt.tight_layout()
plt.savefig(f'figures/{FUNC_NAME}/PT_Min_Fitness_Heatmap_{EXCHANGE_PROCEDURE}.png')

print('Done!')

# Optimally-tuned PT

PT = ParallelTempering(
        objective_function=FUNCTION,
        x_dim=X_DIM,
        range=X_RANGE,
        num_replicas=NUM_REPLICAS,
        num_chains=NUM_SOL_PER_REPLICA,
        exchange_procedure='Periodic',
        exchange_param=0.005,
        schedule_type='Power',
        power_term=5
    )

# Make a grid of 10 2D plots to show evolution of population
PLOT_EVERY = 100 // 5
num_plots = 100 // PLOT_EVERY
# Two rows, one for evolution with iterations, one for solutions for each replica
fig, axs = plt.subplots(2, num_plots, figsize=(22, 10))
plt.suptitle(f"Optimall-Tuned PT; Evolution with Iteration and Final Solutions per Replica  \n"
            + r"[Exchange Procedure: \textbf{" + EXCHANGE_PROCEDURE
            + r"}, Exchange Parameter: \textbf{" + str(0.005)
            + r"}, Schedule: \textbf{" + TEMP_TYPE
            + r"}, Power Term: \textbf{" + str(5)
            + r"}]", fontsize=16)

# On first row, plot grey contour plots of function on each subplot in grid
X1, X2, f_feasible = evaluate_2D(FUNCTION, x_range=X_RANGE, constraints=True)
for idx in range(num_plots):
    plot_sub_contour(X1, X2, f_feasible, plot=axs[0][idx], x_range=X_RANGE)

# Make a new array of every t_index'th temperature, (excluding idx 0)
t_index = len(PT.temperature_schedule) // num_plots
temps = PT.temperature_schedule[1::t_index]

for idx in range(num_plots):
    f_temp = f_feasible ** temps[idx]

    plot_sub_contour(X1, X2, f_temp, axs[1][idx], x_range=X_RANGE, colour='Greys')
    axs[1][idx].set_title(f'Temperature = {temps[idx]:.2f}')
    plt.colorbar(axs[1][idx].contourf(X1, X2, f_temp, 100, cmap='inferno'), ax=axs[1][idx])

for iter in range(100):
    if iter % PLOT_EVERY == 0:
        plot_num = (iter // PLOT_EVERY)
        idx = plot_num % num_plots
        axs[0][idx].set_title(f'Iteration: {iter}')
        final_replica = PT.get_all_solutions()
        plot_population(final_replica, axs[0][idx], best=PT.get_best_solution(), last=final_replica[-1])

    PT.update_chains()
    PT.replica_exchange(iter)

replica_solutions = PT.current_solutions[1::t_index]

# Overlay replica solutions on second row
for idx in range(num_plots):
    solutions = PT.scale_up(replica_solutions[idx])
    axs[1][idx].scatter(solutions[:, 0], solutions[:, 1], c='lime', s=15, label='Replica Solutions')

plt.tight_layout()
plt.savefig(f'figures/{FUNC_NAME}/PT_Optimal_Tuning.png')





