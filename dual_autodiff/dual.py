import numpy as np
from typing import Dict, Union

class Dual:
    """
    Dual numbers class to compute derivatives or partial derivatives of functions.
    The dual component must be inputted as a dictionary to allow multiple components of the dual number.
    This allows easy computation of partial derivatives.
    """
    def __init__(self, real_component, dual_component: Dict[str, float]):
        """
        Args:
            real_component (_type_): real component of dual number
            dual_component (Dict[str, float]): dual components
        """
        if not isinstance(dual_component, dict):
            raise TypeError(
                f"Expected 'dual_component' to be a dictionary, but got {type(dual_component).__name__}."
        )
        self.real = real_component
        self.dual = dual_component if isinstance(dual_component, dict) else {}
    
    def __repr__(self):
        """
        Allows you to print the dual number

        Returns:
            string: description of dual number
        """
        return f"Dual(real={self.real}, dual={self.dual})"

    def __add__(self, other):
        if isinstance(other, Dual):
            new_dual = self.dual.copy()
            for var, val in other.dual.items():
                new_dual[var] = new_dual.get(var, 0) + val
            return Dual(self.real + other.real, new_dual)
        else:
            return Dual(self.real + other, self.dual.copy())
    
    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, Dual):
            new_dual = {}
            all_vars = set(self.dual.keys()) | set(other.dual.keys())
            for var in all_vars:
                new_dual[var] = (self.real * other.dual.get(var, 0) + 
                                self.dual.get(var, 0) * other.real)
            return Dual(self.real * other.real, new_dual)
        else:
            return Dual(self.real * other, 
                       {k: v * other for k, v in self.dual.items()})
    
    __rmul__ = __mul__

    def __sub__(self, other):
        if isinstance(other, Dual):
            new_dual = self.dual.copy()
            for var, val in other.dual.items():
                new_dual[var] = new_dual.get(var, 0) - val
            return Dual(self.real - other.real, new_dual)
        else:
            return Dual(self.real - other, self.dual.copy())
    
    def __rsub__(self, other):
        return Dual(other - self.real, {k: -v for k, v in self.dual.items()})

    def __truediv__(self, other):
        if isinstance(other, Dual):
            if other.real == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            new_dual = {}
            all_vars = set(self.dual.keys()) | set(other.dual.keys())
            for var in all_vars:
                new_dual[var] = (
                    (self.dual.get(var, 0) * other.real - 
                     self.real * other.dual.get(var, 0)) / 
                    (other.real * other.real)
                )
            return Dual(self.real / other.real, new_dual)
        else:
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            return Dual(self.real / other, 
                       {k: v / other for k, v in self.dual.items()})
        
    def __rtruediv__(self, other):
        """Handle division when a regular number is divided by a Dual number."""
        if self.real == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")
        real_part = other / self.real
        dual_part = {k: -other * v / (self.real * self.real) 
                    for k, v in self.dual.items()}
        return Dual(real_part, dual_part)

    def __pow__(self, other):
        """Handle power operations when a Dual number is the base."""
        if isinstance(other, Dual):
            # If exponent is a Dual number: x^y = exp(y*ln(x))
            if self.real <= 0:
                raise ValueError("For Dual number exponents, base must be positive")
            real_part = self.real ** other.real
            # Using the formula d/dx(x^y) = y*x^(y-1)
            # and d/dy(x^y) = x^y * ln(x)
            dual_part = {}
            all_vars = set(self.dual.keys()) | set(other.dual.keys())
            for var in all_vars:
                # Term from differentiating base
                base_term = other.real * (self.real ** (other.real - 1)) * self.dual.get(var, 0)
                # Term from differentiating exponent
                exp_term = real_part * np.log(self.real) * other.dual.get(var, 0)
                dual_part[var] = base_term + exp_term
            return Dual(real_part, dual_part)
        else:
            # If exponent is a regular number: x^n
            real_part = self.real ** other
            # Using the power rule d/dx(x^n) = n*x^(n-1)
            dual_part = {k: other * (self.real ** (other - 1)) * v 
                        for k, v in self.dual.items()}
            return Dual(real_part, dual_part)

    def __rpow__(self, other):
        """Handle power operations when a regular number is the base."""
        if not isinstance(other, (int, float)):
            return NotImplemented
        # For a^x, derivative is a^x * ln(a)
        real_part = other ** self.real
        dual_part = {k: real_part * np.log(other) * v 
                    for k, v in self.dual.items()}
        return Dual(real_part, dual_part)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        args = []
        dual_args = []
        for arg in inputs:
            if isinstance(arg, Dual):
                args.append(arg.real)
                dual_args.append(arg.dual)
            else:
                args.append(arg)
                dual_args.append({})

        implementations = {#COULD STORE THIS IN SEP FILE SO USER CAN IMPLEMENT THEIR OWN 
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


        try:
            if ufunc is np.add:
                return self._handle_add(args, dual_args)
            elif ufunc is np.multiply:
                return self._handle_multiply(args, dual_args)
            elif ufunc is np.divide:
                return self._handle_divide(args, dual_args)
            elif ufunc is np.power:
                return self._handle_power(args, dual_args)
            elif ufunc.__name__ in implementations:
                f, fprime = implementations[ufunc.__name__]
                real_part = f(args[0])
                dual_part = {k: v * fprime(args[0]) 
                           for k, v in dual_args[0].items()}
                return Dual(real_part, dual_part)
        except Exception as e:
            print(f"Error in {ufunc.__name__}: {str(e)}")
            return NotImplemented

        return NotImplemented

    def _handle_add(self, args, dual_args):
        real_part = args[0] + args[1]
        dual_part = {}
        for d in dual_args:
            for k, v in d.items():
                dual_part[k] = dual_part.get(k, 0) + v
        return Dual(real_part, dual_part)

    def _handle_multiply(self, args, dual_args):
        real_part = args[0] * args[1]
        dual_part = {}
        
        if dual_args[0]:
            for k, v in dual_args[0].items():
                dual_part[k] = v * args[1]
        
        if dual_args[1]:
            for k, v in dual_args[1].items():
                dual_part[k] = dual_part.get(k, 0) + v * args[0]
                
        return Dual(real_part, dual_part)

    def _handle_divide(self, args, dual_args):
        if args[1] == 0:
            raise ZeroDivisionError("Division by zero")
            
        real_part = args[0] / args[1]
        dual_part = {}
        
        if dual_args[0]:
            for k, v in dual_args[0].items():
                dual_part[k] = v / args[1]
                
        if dual_args[1]:
            for k, v in dual_args[1].items():
                dual_part[k] = dual_part.get(k, 0) - (args[0] * v) / (args[1] * args[1])
                
        return Dual(real_part, dual_part)

    def _handle_power(self, args, dual_args):
        real_part = args[0] ** args[1]
        dual_part = {}
        
        if dual_args[0]:  # Base is Dual
            for k, v in dual_args[0].items():
                dual_part[k] = args[1] * (args[0] ** (args[1] - 1)) * v
                
        if dual_args[1]:  # Exponent is Dual
            for k, v in dual_args[1].items():
                dual_part[k] = dual_part.get(k, 0) + real_part * np.log(args[0]) * v
                
        return Dual(real_part, dual_part)
