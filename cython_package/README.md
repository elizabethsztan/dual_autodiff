# Cython Dual Numbers Package

A high-performance Cython implementation for automatic differentiation using dual numbers. This package provides optimized computation of derivatives and partial derivatives through operator overloading and NumPy integration.

## Features

- Cython implementation for high performance
- Optimized dual number arithmetic operations (+, -, *, /)
- Efficient power operations (x^n, n^x)
- Fast computation of partial derivatives
- Accelerated NumPy universal function (ufunc) integration
- Optimized mathematical functions (trigonometric, exponential, logarithmic)
- Extensible function support

## Installation

### Prerequisites
- C compiler (GCC, Clang, or MSVC)
- Cython
- NumPy

Run
```bash
pip install -e .
```
from the root directory (cython_package folder).

## Usage

### Basic Operations

```python
from dual_autodiff_x.dual import Dual

# Create a dual number (x = 2 + ε)
x = Dual(2, {'x': 1})

# Basic arithmetic
y = x * x  # Computes derivative of x^2
print(y)  # Output: Dual(real=4, dual={'x': 4})

# Multiple variables for partial derivatives
z = Dual(3, {'y': 1})
result = x * z  # Computes partial derivatives
print(result)  # Output: Dual(real=6, dual={'x': 3, 'y': 2})
```

### Mathematical Functions

```python
import numpy as np

# Using NumPy functions (optimized implementation)
x = Dual(np.pi/4, {'x': 1})
y = np.sin(x)
print(y)  # Shows sin(π/4) and its derivative

# Custom function composition
z = np.exp(np.sin(x))  # Computes derivative using chain rule
```

### Adding Custom Functions

```python
from dual_autotodiff_x.tools import add_function

def custom_func(x):
    return x * x * x

def custom_func_derivative(x):
    return 3 * x * x

add_function('cube', custom_func, custom_func_derivative)

x = Dual(2, {'x': 1})
y = np.cube(x)
```

## Technical Details

### Performance Optimizations

- Cython-optimized core operations
- Efficient memory management
- Reduced Python object overhead
- Optimized dictionary operations for dual components
- Fast NumPy ufunc integration
