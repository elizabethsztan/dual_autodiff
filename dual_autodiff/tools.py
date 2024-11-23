import numpy as np
from typing import Callable, Dict, Tuple

# Type definition for clarity
tool_format = Tuple[Callable[[float], float], Callable[[float], float]]
tool_store = Dict[str, tool_format]

# Base implementation dictionary
base_implementations: tool_store = {
    # Basic Trigonometric Functions
    'sin': (np.sin, lambda x: np.cos(x)),
    'cos': (np.cos, lambda x: -np.sin(x)),
    'tan': (np.tan, lambda x: 1 / (np.cos(x)**2)),

    # Hyperbolic Functions
    'sinh': (np.sinh, lambda x: np.cosh(x)),
    'cosh': (np.cosh, lambda x: np.sinh(x)),
    'tanh': (np.tanh, lambda x: 1 / (np.cosh(x)**2)),

    # Exponential and Logarithmic Functions
    'exp': (np.exp, lambda x: np.exp(x)),
    'log': (np.log, lambda x: 1 / x),
    'sqrt': (np.sqrt, lambda x: 1 / (2 * np.sqrt(x))),

    # Inverse Trigonometric Functions
    'arcsin': (np.arcsin, lambda x: 1 / np.sqrt(1 - x**2)),
    'arccos': (np.arccos, lambda x: -1 / np.sqrt(1 - x**2)),
    'arctan': (np.arctan, lambda x: 1 / (1 + x**2))
}

def add_function(name: str, func: Callable[[float], float], 
                derivative: Callable[[float], float]) -> None:
    """
    Add a new function to the implementation dictionary.
    
    Args:
        name: Name of the function
        func: The function implementation
        derivative: The derivative implementation
    """
    base_implementations[name] = (func, derivative)

def get_functions() -> tool_store:
    """Get the current implementation dictionary."""
    return base_implementations.copy()

def remove_function(name: str) -> None:
    """Remove a function from the implementation dictionary."""
    if name in base_implementations:
        del base_implementations[name]