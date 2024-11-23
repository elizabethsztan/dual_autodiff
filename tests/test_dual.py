# test_dual.py
import pytest
import numpy as np
from dual_autodiff import Dual

def test_dual_initialization():
    # Test basic initialization
    d = Dual(1.0, {'x': 1.0})
    assert d.real == 1.0
    assert d.dual == {'x': 1.0}

    # Test initialization with empty dual component
    d = Dual(1.0, {})
    assert d.real == 1.0
    assert d.dual == {}

    # Test initialization with invalid dual component
    with pytest.raises(TypeError):
        Dual(1.0, [1.0])  # Should raise TypeError

def test_dual_arithmetic():
    # Setup test numbers
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

def test_dual_division():
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

def test_dual_power():
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

    # Test power with negative base
    with pytest.raises(ValueError):
        result = Dual(-2.0, {'x': 1.0}) ** d2

def test_numpy_functions():
    d = Dual(np.pi/4, {'x': 1.0})

    # Test sine
    result = np.sin(d)
    assert pytest.approx(result.real) == np.sin(np.pi/4)
    assert pytest.approx(result.dual['x']) == np.cos(np.pi/4)

    # Test cosine
    result = np.cos(d)
    assert pytest.approx(result.real) == np.cos(np.pi/4)
    assert pytest.approx(result.dual['x']) == -np.sin(np.pi/4)

    # Test exponential
    result = np.exp(d)
    assert pytest.approx(result.real) == np.exp(np.pi/4)
    assert pytest.approx(result.dual['x']) == np.exp(np.pi/4)

def test_dual_negation():
    d = Dual(2.0, {'x': 1.0, 'y': 2.0})
    result = -d
    assert result.real == -2.0
    assert result.dual == {'x': -1.0, 'y': -2.0}

def test_dual_repr():
    d = Dual(2.0, {'x': 1.0})
    assert repr(d) == "Dual(real=2.0, dual={'x': 1.0})"