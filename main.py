import sys; sys.path.append('..')
from src.utils.plotting_functions import plot_2D
from src.functions import KBF_function, Rosenbrock_function

# Plot the function
plot_2D(KBF_function)
plot_2D(Rosenbrock_function)

