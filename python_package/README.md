# Python Dual Numbers Package

A pure Python implementation for automatic differentiation using dual numbers. This package provides an intuitive interface for computing derivatives and partial derivatives through operator overloading and NumPy integration.

## Features

- Pure Python implementation for maximum compatibility
- Dual number arithmetic operations (+, -, *, /)
- Power operations (x^n, n^x)
- Support for partial derivatives through dictionary-based dual components
- NumPy universal function (ufunc) integration
- Common mathematical functions (trigonometric, exponential, logarithmic)
- Extensible function support

## Installation
Run 
```bash
pip install -e .
```
from the root directory (python_package folder).

## Usage

### Basic Operations

```python
from dual_autodiff.dual import Dual

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

# Using NumPy functions
x = Dual(np.pi/4, {'x': 1})
y = np.sin(x)
print(y)  # Shows sin(π/4) and its derivative

# Custom function composition
z = np.exp(np.sin(x))  # Computes derivative using chain rule
```

### Adding Custom Functions

```python
from dual_autodiff.tools import add_function

def custom_func(x):
    return x * x * x

def custom_func_derivative(x):
    return 3 * x * x

add_function('cube', custom_func, custom_func_derivative)

x = Dual(2, {'x': 1})
y = np.cube(x)
```

## Technical Details

### Dual Number Theory

A dual number has the form a + bε where:
- a is the real component
- b is the dual component
- ε is the dual unit where ε² = 0

This implementation extends the concept to support multiple dual components for partial derivatives, representing them as a dictionary.

### Implementation Details

- Pure Python implementation for readability and maintainability
- Dictionary-based dual components for efficient partial derivative computation
- NumPy ufunc integration for mathematical functions
- Comprehensive operator overloading for natural syntax

## Use Cases

- Educational purposes and learning automatic differentiation
- Prototyping and testing derivative computations
- Small to medium-scale numerical computations
- Research and development scenarios where code readability is priority

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Running Tests

```bash
pytest tests/
```

## License

MIT License
