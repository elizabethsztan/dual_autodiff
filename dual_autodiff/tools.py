import numpy as np
from typing import Callable, Dict, Tuple

# Type definition for clarity
tool_format = Tuple[Callable[[float], float], Callable[[float], float]]
tool_store = Dict[str, tool_format]

# Base implementation dictionary
base_implementations: tool_store = {
    # Trigonometric functions
    'sin': (np.sin, lambda x: np.cos(x)),
    'cos': (np.cos, lambda x: -np.sin(x)),
    'tan': (np.tan, lambda x: 1 / (np.cos(x)**2)),

    # Hyperbolic functions
    'sinh': (np.sinh, lambda x: np.cosh(x)),
    'cosh': (np.cosh, lambda x: np.sinh(x)),
    'tanh': (np.tanh, lambda x: 1 / (np.cosh(x)**2)),

    # Exponential and logarithmic functions
    'exp': (np.exp, lambda x: np.exp(x)),
    'log': (np.log, lambda x: 1 / x),
    'sqrt': (np.sqrt, lambda x: 1 / (2 * np.sqrt(x))),

    # Inverse trigonometric functions
    'arcsin': (np.arcsin, lambda x: 1 / np.sqrt(1 - x**2)),
    'arccos': (np.arccos, lambda x: -1 / np.sqrt(1 - x**2)),
    'arctan': (np.arctan, lambda x: 1 / (1 + x**2))
}

def add_function(name: str, func: Callable[[float], float], 
                derivative: Callable[[float], float]) -> None:
    """
    Add a new function to the implementation dictionary.
    
    Args:
        name (str): name of the function
        func (Callable[[float], float]): the function implementation
        derivative (Callable[[float], float]): the function derivative implementation

    Example:
        >>> from dual_autodiff.tools import add_function
        >>> def func (x):
        >>>     return x**3
        >>> def func_derivative (x):
        >>>     return 3*x**2
        >>> add_function ('x_cubed', func, func_derivative) #adds function to tool_store
        >>> x_dual = Dual(2, {'x': 1})
        >>> print('The derivative of x cubed at x = 2 is ', x_cubed(x_dual.dual['x']))
        >>> The derivative of x cubed at x = 2 is 12.
    """
    base_implementations[name] = (func, derivative)

def get_functions() -> tool_store:
    """
    Get the current implementations dictionary.
    """
    return base_implementations.copy()

def remove_function(name: str) -> None:
    """
    Remove a function from the implementation dictionary.
    """
    if name in base_implementations:
        del base_implementations[name]