# cython: language_level=3
# distutils: language=c++

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, log, cos, sin, tan, sinh, cosh, tanh, asin, acos, atan

# Type definitions
tool_format = tuple
tool_store = dict

# Base implementation dictionary 
base_implementations = {
    # Trigonometric functions
    'sin': (np.sin, lambda x: cos(x)),
    'cos': (np.cos, lambda x: -sin(x)),
    'tan': (np.tan, lambda x: 1.0 / (cos(x) * cos(x))),

    # Hyperbolic functions
    'sinh': (np.sinh, lambda x: cosh(x)),
    'cosh': (np.cosh, lambda x: sinh(x)),
    'tanh': (np.tanh, lambda x: 1.0 / (cosh(x) * cosh(x))),

    # Exponential and logarithmic functions
    'exp': (np.exp, lambda x: exp(x)),
    'log': (np.log, lambda x: 1.0 / x),
    'sqrt': (np.sqrt, lambda x: 1.0 / (2.0 * sqrt(x))),

    # Inverse trigonometric functions
    'arcsin': (np.arcsin, lambda x: 1.0 / sqrt(1.0 - x * x)),
    'arccos': (np.arccos, lambda x: -1.0 / sqrt(1.0 - x * x)),
    'arctan': (np.arctan, lambda x: 1.0 / (1.0 + x * x))
}

cpdef add_function(str name, func, derivative):
    """Add a new function to the implementation dictionary."""
    base_implementations[name] = (func, derivative)

cpdef get_functions():
    """Get the current implementation dictionary."""
    return base_implementations.copy()

cpdef remove_function(str name):
    """Remove a function from the implementation dictionary."""
    if name in base_implementations:
        del base_implementations[name]