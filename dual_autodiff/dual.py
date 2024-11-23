import numpy as np
from typing import Dict, Union

class Dual:
    def __init__(self, real_component, dual_component: Dict[str, float]):
        self.real = real_component
        self.dual = dual_component if isinstance(dual_component, dict) else {}
    
    def __repr__(self):
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

    __rmul__ = __mul__  # Added this line to fix right multiplication

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

    def __neg__(self):
        return Dual(-self.real, {k: -v for k, v in self.dual.items()})

    def __eq__(self, other):
        if isinstance(other, Dual):
            return self.real == other.real and self.dual == other.dual
        return False

    def __pow__(self, exponent):
        new_dual = {k: exponent * (self.real**(exponent - 1)) * v 
                   for k, v in self.dual.items()}
        return Dual(self.real**exponent, new_dual)

    def sin(self):
        return Dual(np.sin(self.real),
                   {k: v * np.cos(self.real) for k, v in self.dual.items()})

    def cos(self):
        return Dual(np.cos(self.real),
                   {k: -v * np.sin(self.real) for k, v in self.dual.items()})

    def tan(self):
        return Dual(np.tan(self.real),
                   {k: v / (np.cos(self.real) * np.cos(self.real)) 
                    for k, v in self.dual.items()})
    
    def exp(self):
        return Dual(np.exp(self.real),
                   {k: v * np.exp(self.real) for k, v in self.dual.items()})

    def log(self):
        if self.real <= 0:
            raise ValueError("Logarithm is undefined for non-positive values.")
        return Dual(np.log(self.real),
                   {k: v / self.real for k, v in self.dual.items()})

    def sinh(self):
        return Dual(np.sinh(self.real),
                   {k: v * np.cosh(self.real) for k, v in self.dual.items()})

    def cosh(self):
        return Dual(np.cosh(self.real),
                   {k: v * np.sinh(self.real) for k, v in self.dual.items()})

    def tanh(self):
        return Dual(np.tanh(self.real),
                   {k: v / (np.cosh(self.real) ** 2) for k, v in self.dual.items()})  # Fixed this line
    
    def arcsin(self):
        if self.real < -1 or self.real > 1:
            raise ValueError("arcsin is undefined for values outside [-1, 1].")
        return Dual(np.arcsin(self.real),
                   {k: v / np.sqrt(1 - self.real * self.real) 
                    for k, v in self.dual.items()})

    def arccos(self):
        if self.real < -1 or self.real > 1:
            raise ValueError("arccos is undefined for values outside [-1, 1].")
        return Dual(np.arccos(self.real),
                   {k: -v / np.sqrt(1 - self.real * self.real) 
                    for k, v in self.dual.items()})

    def arctan(self):
        return Dual(np.arctan(self.real),
                   {k: v / (1 + self.real * self.real) 
                    for k, v in self.dual.items()})

    def sqrt(self):
        if self.real < 0:
            raise ValueError("Square root is undefined for negative values.")
        return Dual(np.sqrt(self.real),
                   {k: v / (2 * np.sqrt(self.real)) for k, v in self.dual.items()})
