"""
Author: Prithvi Raj

Description : 
    This file contains the functions for exchanging solutions between replicas.
"""

import numpy as np

def swap(PT):
    """
    Swap solutions between adjacent replicas, (subject to Metropolis criterion).
    """ 
    # Loop through each replica
    for i in range(PT.num_replicas - 1):
        
        # Temperatures of adjacent replicas to swap
        T_1 = PT.temperature_schedule[i]
        T_2 = PT.temperature_schedule[i + 1]

        # Loop through each solution in replica
        for j in range(PT.num_chains):

            # If solutions are the same, no need to swap
            if np.array_equal(PT.current_solutions[i, j], PT.current_solutions[i + 1, j]):
                continue

            # Check Metropolis criterion for both directions. 
            # Acceptance is now dependent on temp difference, so now both temps are sent in as args
            check_criterion = [
                PT.metropolis_criterion(PT.current_solutions[i, j], PT.current_solutions[i + 1, j], T_1, T_2),
                PT.metropolis_criterion(PT.current_solutions[i+1, j], PT.current_solutions[i , j], T_2, T_1)
            ]

            # Only swap solutions if both directions satisfy Metropolis criterion
            if all(check_criterion):

                # Swap solutions
                PT.current_solutions[i, j], PT.current_solutions[i + 1, j] = PT.current_solutions[i + 1, j], PT.current_solutions[i, j]

                # Update max. allowable step size
                PT.update_max_change(PT.current_solutions[i, j], PT.current_solutions[i + 1, j], i, j)


def period_exchange(PT, iter):
    """
    Periodic exchange of solutions between replicas, swaps solutions every PT.exchange_param*NUM_ITERS.
    """
    # Check if it is time to swap solutions between replicas
    if iter % (PT.exchange_param * PT.total_iterations) == 0:
        swap(PT)

def stochastic_exchange(PT, iter):
    """
    Random exchange of solutions between replicas, with probability PT.exchange_param.
    """
    # Chance to swap solutions between replicas
    if np.random.uniform() < PT.exchange_param:
        swap(PT)

def always_exchange(PT, iter):
    """
    Always exchange solutions between replicas.
    """
    swap(PT)






