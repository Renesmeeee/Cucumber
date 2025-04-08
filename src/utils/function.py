"""
Common utility functions for cucumber growth model
"""

import numpy as np

def gompertz_growth(t, a, b, c):
    """
    Define the Gompertz growth function
    
    Args:
        t (float): Time or thermal time
        a (float): Asymptote parameter
        b (float): Displacement along x-axis
        c (float): Growth rate parameter
        
    Returns:
        float: Calculated growth value
    """
    return a * np.exp(-np.exp(-(t - b) / c)) 