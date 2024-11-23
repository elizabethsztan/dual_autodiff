# test_tools.py
import pytest
import numpy as np
from dual_autodiff.tools import add_function, get_functions, remove_function

def test_get_functions():
    functions = get_functions()
    
    # All base functions that should be present
    expected_functions = [
        'sin', 'cos', 'tan',
        'sinh', 'cosh', 'tanh',
        'exp', 'log', 'sqrt',
        'arcsin', 'arccos', 'arctan'
    ]
    
    # Test if all expected functions are present
    for func_name in expected_functions:
        assert func_name in functions, f"Missing base function: {func_name}"
    
    # Test that we're not getting extra unexpected functions
    assert set(functions.keys()) == set(expected_functions), \
        "Got unexpected functions in base implementations"
    
    # Test function and derivative pairs at characteristic points
    test_points = {
        'sin': {'point': np.pi/2, 'expected_f': 1.0, 'expected_fprime': 0.0},
        'cos': {'point': 0, 'expected_f': 1.0, 'expected_fprime': 0.0},
        'exp': {'point': 0, 'expected_f': 1.0, 'expected_fprime': 1.0},
        'log': {'point': 1, 'expected_f': 0.0, 'expected_fprime': 1.0},
        'sqrt': {'point': 1, 'expected_f': 1.0, 'expected_fprime': 0.5},
    }
    
    for func_name, test_data in test_points.items():
        f, fprime = functions[func_name]
        point = test_data['point']
        assert pytest.approx(f(point)) == test_data['expected_f'], \
            f"Function {func_name} failed at point {point}"
        assert pytest.approx(fprime(point)) == test_data['expected_fprime'], \
            f"Derivative of {func_name} failed at point {point}"

def test_add_function():
    # Add a custom function
    def cubic(x):
        return x**3
    
    def cubic_derivative(x):
        return 3*x**2
    
    add_function('cubic', cubic, cubic_derivative)
    
    functions = get_functions()
    assert 'cubic' in functions
    
    f, fprime = functions['cubic']
    assert f(2) == 8
    assert fprime(2) == 12

def test_remove_function():
    # Add and then remove a function
    add_function('test_func', lambda x: x, lambda x: 1)
    assert 'test_func' in get_functions()
    
    remove_function('test_func')
    assert 'test_func' not in get_functions()
    
    # Test removing non-existent function (should not raise error)
    remove_function('non_existent_function')

def test_function_derivatives():
    functions = get_functions()
    
    # Test derivatives at specific points
    # Format: (function_name, test_point, expected_derivative)
    test_cases = [
        # Trigonometric functions
        ('sin', 0, 1.0),           # d/dx(sin(x)) = cos(x)
        ('cos', 0, 0.0),           # d/dx(cos(x)) = -sin(x)
        ('tan', 0, 1.0),           # d/dx(tan(x)) = sec²(x) = 1/cos²(x)
        
        # Hyperbolic functions
        ('sinh', 0, 1.0),          # d/dx(sinh(x)) = cosh(x)
        ('cosh', 0, 0.0),          # d/dx(cosh(x)) = sinh(x)
        ('tanh', 0, 1.0),          # d/dx(tanh(x)) = sech²(x) = 1/cosh²(x)
        
        # Exponential and logarithmic functions
        ('exp', 0, 1.0),           # d/dx(e^x) = e^x
        ('log', 1, 1.0),           # d/dx(ln(x)) = 1/x
        ('sqrt', 1, 0.5),          # d/dx(√x) = 1/(2√x)
        
        # Inverse trigonometric functions
        ('arcsin', 0, 1.0),        # d/dx(arcsin(x)) = 1/√(1-x²)
        ('arccos', 0, -1.0),       # d/dx(arccos(x)) = -1/√(1-x²)
        ('arctan', 0, 1.0)         # d/dx(arctan(x)) = 1/(1+x²)
    ]
    
    for func_name, x, expected_derivative in test_cases:
        _, fprime = functions[func_name]
        assert pytest.approx(fprime(x)) == expected_derivative, \
            f"Derivative of {func_name} at x={x} should be {expected_derivative}"

def test_function_implementations():
    functions = get_functions()
    x = 0.5
    
    # Only test numpy-available functions
    numpy_functions = [func_name for func_name in functions.keys() 
                      if hasattr(np, func_name)]
    
    # Test function values
    for func_name in numpy_functions:
        f, _ = functions[func_name]
        np_func = getattr(np, func_name)
        assert pytest.approx(f(x)) == np_func(x)

def test_copy_independence():
    original = get_functions()
    copied = get_functions()
    
    # Add a function to the copied dictionary
    copied['test'] = (lambda x: x, lambda x: 1)
    
    # Original should not be modified
    assert 'test' not in original

def test_custom_functions():
    # Use the same cubic function from test_add_function
    def cubic(x):
        return x**3
    
    def cubic_derivative(x):
        return 3*x**2
    
    # Add the function
    add_function('cubic', cubic, cubic_derivative)
    
    # Get functions and test
    functions = get_functions()
    f, fprime = functions['cubic']
    
    # Test at a few points
    x_values = [-1, 0, 1, 2]
    for x in x_values:
        assert f(x) == x**3
        assert fprime(x) == 3*x**2
    
    # Clean up
    remove_function('cubic')