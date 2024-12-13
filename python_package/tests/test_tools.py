# test_tools.py
import pytest
import numpy as np
from dual_autodiff.tools import add_function, get_functions, remove_function

def test_get_functions():
    """
    Test get_functions() to get base functions for tools_store.
    """
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
        # Trigonometric functions with both function and derivative tests
        'sin': [
            {'point': np.pi/2, 'expected_f': 1.0, 'expected_fprime': 0.0},
            {'point': 0, 'expected_f': 0.0, 'expected_fprime': 1.0}
        ],
        'cos': [
            {'point': 0, 'expected_f': 1.0, 'expected_fprime': 0.0},
            {'point': np.pi/2, 'expected_f': 0.0, 'expected_fprime': -1.0}
        ],
        'tan': [
            {'point': 0, 'expected_f': 0.0, 'expected_fprime': 1.0}
        ],
        
        # Hyperbolic functions
        'sinh': [
            {'point': 0, 'expected_f': 0.0, 'expected_fprime': 1.0}
        ],
        'cosh': [
            {'point': 0, 'expected_f': 1.0, 'expected_fprime': 0.0}
        ],
        'tanh': [
            {'point': 0, 'expected_f': 0.0, 'expected_fprime': 1.0}
        ],
        
        # Exponential and logarithmic functions
        'exp': [
            {'point': 0, 'expected_f': 1.0, 'expected_fprime': 1.0}
        ],
        'log': [
            {'point': 1, 'expected_f': 0.0, 'expected_fprime': 1.0}
        ],
        'sqrt': [
            {'point': 1, 'expected_f': 1.0, 'expected_fprime': 0.5}
        ],
        
        # Inverse trigonometric functions
        'arcsin': [
            {'point': 0, 'expected_f': 0.0, 'expected_fprime': 1.0}
        ],
        'arccos': [
            {'point': 0, 'expected_f': np.pi/2, 'expected_fprime': -1.0}
        ],
        'arctan': [
            {'point': 0, 'expected_f': 0.0, 'expected_fprime': 1.0}
        ]
    }
    
    for func_name, test_cases in test_points.items():
        f, fprime = functions[func_name]
        for test_data in test_cases:
            point = test_data['point']
            assert pytest.approx(f(point)) == test_data['expected_f'], \
                f"Function {func_name} failed at point {point}"
            assert pytest.approx(fprime(point)) == test_data['expected_fprime'], \
                f"Derivative of {func_name} at x={point} should be {test_data['expected_fprime']}"

def test_add_function():
    """
    Test add_function() to add a custom function.
    """
    #Add cubic function and derivative
    def cubic(x):
        return x**3
    
    def cubic_derivative(x):
        return 3*x**2
    
    add_function('cubic', cubic, cubic_derivative)
    
    #Call functions in tools_store
    functions = get_functions()
    assert 'cubic' in functions
    
    #Check that function and derivative return expected result
    f, fprime = functions['cubic']

    # Test the function at multiple points
    x = [-1, 0, 0.5, 2]
    for i in x:
        assert f(i) == i**3
        assert fprime(i) == 3 * i**2

def test_remove_function():
    """
    Test remove_function() to remove a function from tools_store.
    """
    # Add and then remove a function
    add_function('test_func', lambda x: x, lambda x: 1)
    assert 'test_func' in get_functions()
    
    remove_function('test_func')
    assert 'test_func' not in get_functions()
    
    # Test removing non-existent function (should not raise error)
    remove_function('non_existent_function')

def test_function_implementations():
    """
    Test that the numpy functions in the tools_store give the same results as 
    their numpy counterparts.
    """
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
    """
    Test that a copy of the tools_store is independent to the original tools_store.
    """
    original = get_functions()
    copied = get_functions()
    
    # Add a function to the copied dictionary
    copied['test'] = (lambda x: x, lambda x: 1)
    
    # Original should not be modified
    assert 'test' not in original
