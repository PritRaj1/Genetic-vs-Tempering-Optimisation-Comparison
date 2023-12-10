import numpy as np

def evaluate_2D(func, x_range=(0,10)): 
    """
    Function for generating a meshgrid and evaluating a function in R^2.
    """

    # Create a meshgrid
    x1 = np.linspace(x_range[0], x_range[1], 100)
    x2 = np.linspace(x_range[0], x_range[1], 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute the function values
    f = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            f[i, j] = func(np.array([X1[i, j], X2[i, j]]))
        
    return X1, X2, f