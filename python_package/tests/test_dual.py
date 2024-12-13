# test_dual.py
import pytest
import numpy as np
from dual_autodiff import Dual

def test_dual_initialization():
    """
    Test basic initialisation of the Dual class.
    """
    d = Dual(1.0, {'x': 1.0})
    assert d.real == 1.0
    assert d.dual == {'x': 1.0}

    # Test initialization with empty dual component
    d = Dual(1.0, {})
    assert d.real == 1.0
    assert d.dual == {}

    # Check that TypeError raised when dual component is not a Dict
    with pytest.raises(TypeError):
        Dual(1.0, [1.0])  

def test_dual_arithmetic():
    """
    Test addition, subtraction and multiplication with dual numbers and scalars.
    """
    d1 = Dual(2.0, {'x': 1.0, 'y': 2.0})
    d2 = Dual(3.0, {'x': 2.0, 'z': 3.0})
    scalar = 2.0

    # Test addition
    result = d1 + d2
    assert result.real == 5.0
    assert result.dual == {'x': 3.0, 'y': 2.0, 'z': 3.0}

    # Test scalar addition
    result = d1 + scalar
    assert result.real == 4.0
    assert result.dual == {'x': 1.0, 'y': 2.0}

    # Test subtraction
    result = d1 - d2
    assert result.real == -1.0
    assert result.dual == {'x': -1.0, 'y': 2.0, 'z': -3.0}

    # Test scalar subtraction
    result = d1 - scalar
    assert result.real == 0.0
    assert result.dual == {'x': 1.0, 'y': 2.0}

    # Test multiplication
    result = d1 * d2
    assert result.real == 6.0
    assert result.dual == {'x': 7.0, 'y': 6.0, 'z': 6.0}

    # Test scalar multiplication
    result = d1 * scalar
    assert result.real == 4.0
    assert result.dual == {'x': 2.0, 'y': 4.0}

def test_reverse_arithmatic():
    """
    Test reverse addition, subtraction and multiplication with dual numbers and scalars.
    Reverse arithmatic because the scalar is on the left hand side.
    """
    d1 = Dual(6.0, {'x': 1.0, 'y': 2.0})
    scalar = 2.0

    # Test reverse scalar addition
    result = scalar + d1
    assert result.real == 8.0
    assert result.dual == {'x': 1.0, 'y': 2.0}

    # Test reverse scalar subtraction
    result = scalar - d1
    assert result.real == - 4.0
    assert result.dual == {'x': - 1.0, 'y': - 2.0}

    # Test reverse scalar multiplication
    result = scalar * d1
    assert result.real == 12.0
    assert result.dual == {'x': 2.0, 'y': 4.0}

def test_dual_division():
    """
    Test division of dual numbers and by a scalar.
    """
    d1 = Dual(6.0, {'x': 2.0})
    d2 = Dual(2.0, {'x': 1.0})
    scalar = 2.0

    # Test division
    result = d1 / d2
    assert result.real == 3.0
    assert pytest.approx(result.dual['x']) == -0.5

    # Test scalar division
    result = d1 / scalar
    assert result.real == 3.0
    assert result.dual == {'x': 1.0}

    # Test division by zero
    with pytest.raises(ZeroDivisionError):
        d1 / Dual(0.0, {'x': 1.0})

def test_reverse_division():
    """
    Test division of a scalar by a dual number.
    """
    d1 = Dual(5.0, {'x': 2.0})
    scalar = 1.0

    # Test division of scalar by dual
    result = scalar / d1
    assert result.real == 0.2
    assert result.dual == {'x': - 0.08}

    # Test division by zero
    with pytest.raises(ZeroDivisionError):
        d1 / Dual(0, {'x': 1})

def test_dual_power():
    """
    Test raising a dual number to the power of a dual number or scalar 
    """
    d = Dual(2.0, {'x': 1.0})
    scalar = 3.0

    # Test power with scalar exponent
    result = d ** scalar
    assert result.real == 8.0
    assert pytest.approx(result.dual['x']) == 12.0

    # Test power with dual exponent
    d2 = Dual(2.0, {'x': 0.5})
    result = d ** d2
    assert pytest.approx(result.real) == 4.0

    # Test power with negative base - cannot be calculated due to log 
    with pytest.raises(ValueError):
        result = Dual(-2.0, {'x': 1.0}) ** d2

def test_reverse_dual_power():
    """
    Test raising a scalar to the power of a dual number.
    """
    d = Dual(2.0, {'x': 2.0})  # Dual exponent
    scalar = 3.0               # Positive scalar base
    negative_scalar = -3.0     # Negative scalar base
    scalar_zero = 0.0          # Scalar base zero

    # Test power with dual exponent and scalar base
    result = scalar ** d
    assert result.real == pytest.approx(9.0)  
    assert result.dual['x'] == pytest.approx(3**2 * np.log(3) * 2) 

    # Test power with zero scalar base - cannot be calculated due to log
    with pytest.raises(ValueError):
        scalar_zero ** d

    # Test power with negative scalar base - cannot be calculated due to log
    with pytest.raises(ValueError):
        negative_scalar ** d

def test_numpy_functions():
    """
    Test all base implementations of numpy functions with dual numbers.
    """
    # Use pi/4 for trig functions (nice value) and 0.5 for others
    d_trig = Dual(np.pi/4, {'x': 1.0})
    d_other = Dual(0.5, {'x': 1.0})  # Better point for non-trig functions

    # Trigonometric functions
    # sin(x)
    result = np.sin(d_trig)
    assert pytest.approx(result.real) == np.sin(np.pi/4)
    assert pytest.approx(result.dual['x']) == np.cos(np.pi/4)

    # cos(x)
    result = np.cos(d_trig)
    assert pytest.approx(result.real) == np.cos(np.pi/4)
    assert pytest.approx(result.dual['x']) == -np.sin(np.pi/4)

    # tan(x)
    result = np.tan(d_trig)
    assert pytest.approx(result.real) == np.tan(np.pi/4)
    assert pytest.approx(result.dual['x']) == 1/np.cos(np.pi/4)**2

    # Hyperbolic functions
    # sinh(x)
    result = np.sinh(d_other)
    assert pytest.approx(result.real) == np.sinh(0.5)
    assert pytest.approx(result.dual['x']) == np.cosh(0.5)

    # cosh(x)
    result = np.cosh(d_other)
    assert pytest.approx(result.real) == np.cosh(0.5)
    assert pytest.approx(result.dual['x']) == np.sinh(0.5)

    # tanh(x)
    result = np.tanh(d_other)
    assert pytest.approx(result.real) == np.tanh(0.5)
    assert pytest.approx(result.dual['x']) == 1/np.cosh(0.5)**2

    # Exponential and logarithmic functions
    # exp(x)
    result = np.exp(d_other)
    assert pytest.approx(result.real) == np.exp(0.5)
    assert pytest.approx(result.dual['x']) == np.exp(0.5)

    # log(x)
    result = np.log(d_other)
    assert pytest.approx(result.real) == np.log(0.5)
    assert pytest.approx(result.dual['x']) == 1/0.5

    # sqrt(x)
    result = np.sqrt(d_other)
    assert pytest.approx(result.real) == np.sqrt(0.5)
    assert pytest.approx(result.dual['x']) == 1/(2*np.sqrt(0.5))

    # Inverse trigonometric functions
    # Need to use smaller values for inverse trig functions to stay in domain
    d_small = Dual(0.5, {'x': 1.0})  # 0.5 is safely within [-1,1]

    # arcsin(x)
    result = np.arcsin(d_small)
    assert pytest.approx(result.real) == np.arcsin(0.5)
    assert pytest.approx(result.dual['x']) == 1/np.sqrt(1 - 0.5**2)

    # arccos(x)
    result = np.arccos(d_small)
    assert pytest.approx(result.real) == np.arccos(0.5)
    assert pytest.approx(result.dual['x']) == -1/np.sqrt(1 - 0.5**2)

    # arctan(x)
    result = np.arctan(d_small)
    assert pytest.approx(result.real) == np.arctan(0.5)
    assert pytest.approx(result.dual['x']) == 1/(1 + 0.5**2)

def test_dual_negation():
    """
    Test that negating a dual number inverts the sign on the real and 
    dual components.
    """
    d = Dual(2.0, {'x': 1.0, 'y': 2.0})
    result = -d
    assert result.real == -2.0
    assert result.dual == {'x': -1.0, 'y': -2.0}

def test_dual_repr():
    """
    Test string representatation of the Dual class.
    """
    d = Dual(2.0, {'x': 1.0})
    assert repr(d) == "Dual(real=2.0, dual={'x': 1.0})"