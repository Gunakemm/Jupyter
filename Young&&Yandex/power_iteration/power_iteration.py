import numpy as np
from numpy.linalg import norm

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    eigenvector = np.full(shape=data.shape[0], fill_value=1)
    for i in range(num_steps):
        eigenvector = data.dot(eigenvector) / norm(data.dot(eigenvector))
    eigenvalue = (eigenvector.T.dot(data.dot(eigenvector))) / (eigenvector.T.dot(eigenvector))
    return float(eigenvalue), eigenvector