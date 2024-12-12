# cython: language_level=3
# distutils: language=c++

import numpy as np
cimport numpy as np
from libc.math cimport log, exp, sqrt
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem, PyDict_Keys
from cpython.set cimport PySet_Add, PySet_New
from .tools import get_functions, remove_function

cdef class Dual:
    """
    Dual numbers class to compute derivatives or partial derivatives of functions.
    """
    cdef public double real
    cdef public dict dual

    def __init__(self, real_component, dual_component):
        if not isinstance(real_component, (float, int)):
            raise TypeError(
                f"Expected 'real_component' to be a float or int, but got {type(real_component).__name__}."
            )
        if not isinstance(dual_component, dict):
            raise TypeError(
                f"Expected 'dual_component' to be a dictionary, but got {type(dual_component).__name__}."
            )
        self.real = float(real_component)
        self.dual = dual_component.copy() if isinstance(dual_component, dict) else {}
    
    def __repr__(self):
        return f"Dual(real={self.real}, dual={self.dual})"

    def __add__(self, other):
        cdef dict new_dual
        cdef str var
        cdef double val
        
        if isinstance(other, Dual):
            new_dual = self.dual.copy()
            for var, val in other.dual.items():
                new_dual[var] = new_dual.get(var, 0.0) + val
            return Dual(self.real + other.real, new_dual)
        else:
            return Dual(self.real + float(other), self.dual.copy())
    
    __radd__ = __add__

    def __mul__(self, other):
        cdef dict new_dual
        cdef str var
        cdef double val
        cdef set all_vars
        
        if isinstance(other, Dual):
            new_dual = {}
            all_vars = set(self.dual.keys()) | set(other.dual.keys())
            for var in all_vars:
                new_dual[var] = (self.real * other.dual.get(var, 0.0) + 
                                self.dual.get(var, 0.0) * other.real)
            return Dual(self.real * other.real, new_dual)
        else:
            other_float = float(other)
            return Dual(self.real * other_float, 
                       {k: v * other_float for k, v in self.dual.items()})
    
    __rmul__ = __mul__

    def __sub__(self, other):
        cdef dict new_dual
        cdef str var
        cdef double val
        
        if isinstance(other, Dual):
            new_dual = self.dual.copy()
            for var, val in other.dual.items():
                new_dual[var] = new_dual.get(var, 0.0) - val
            return Dual(self.real - other.real, new_dual)
        else:
            return Dual(self.real - float(other), self.dual.copy())
    
    def __rsub__(self, other):
        return Dual(float(other) - self.real, {k: -v for k, v in self.dual.items()})

    def __truediv__(self, other):
        cdef dict new_dual
        cdef str var
        cdef double val, other_real
        cdef set all_vars
        
        if isinstance(other, Dual):
            if other.real == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            new_dual = {}
            all_vars = set(self.dual.keys()) | set(other.dual.keys())
            other_real = other.real
            for var in all_vars:
                new_dual[var] = (
                    (self.dual.get(var, 0.0) * other_real - 
                     self.real * other.dual.get(var, 0.0)) / 
                    (other_real * other_real)
                )
            return Dual(self.real / other_real, new_dual)
        else:
            other_float = float(other)
            if other_float == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            return Dual(self.real / other_float, 
                       {k: v / other_float for k, v in self.dual.items()})
        
    def __rtruediv__(self, other):
        cdef double other_float
        if self.real == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")
        other_float = float(other)
        return Dual(other_float / self.real,
                   {k: -other_float * v / (self.real * self.real) 
                    for k, v in self.dual.items()})

    def __pow__(self, other):
        cdef dict new_dual
        cdef str var
        cdef double val, base_term, exp_term, real_part
        cdef set all_vars
        
        if isinstance(other, Dual):
            if self.real <= 0:
                raise ValueError("For Dual number exponents, base must be positive")
            real_part = self.real ** other.real
            new_dual = {}
            all_vars = set(self.dual.keys()) | set(other.dual.keys())
            for var in all_vars:
                base_term = other.real * (self.real ** (other.real - 1)) * self.dual.get(var, 0.0)
                exp_term = real_part * log(self.real) * other.dual.get(var, 0.0)
                new_dual[var] = base_term + exp_term
            return Dual(real_part, new_dual)
        else:
            other_float = float(other)
            real_part = self.real ** other_float
            return Dual(real_part,
                       {k: other_float * (self.real ** (other_float - 1)) * v 
                        for k, v in self.dual.items()})

    def __rpow__(self, other):
        cdef double other_float, real_part
        
        if not isinstance(other, (int, float)):
            return NotImplemented
            
        if other <= 0:
            raise ValueError("Base must be positive to raise it to a Dual exponent.")
            
        other_float = float(other)
        real_part = other_float ** self.real
        return Dual(real_part,
                   {k: real_part * log(other_float) * v 
                    for k, v in self.dual.items()})
    
    def __neg__(self):
        return Dual(-self.real, {k: -v for k, v in self.dual.items()})

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        cdef dict implementations
        cdef list args, dual_args
        cdef dict dual_part
        cdef object f, fprime
        cdef double real_part
        
        if method != "__call__":
            return NotImplemented

        implementations = get_functions()
        args = []
        dual_args = []
        for arg in inputs:
            if isinstance(arg, Dual):
                args.append(arg.real)
                dual_args.append(arg.dual)
            else:
                args.append(arg)
                dual_args.append({})

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

    cdef _handle_add(self, list args, list dual_args):
        cdef dict dual_part, d
        cdef str k
        cdef double v
        
        real_part = args[0] + args[1]
        dual_part = {}
        for d in dual_args:
            for k, v in d.items():
                dual_part[k] = dual_part.get(k, 0.0) + v
        return Dual(real_part, dual_part)

    cdef _handle_multiply(self, list args, list dual_args):
        cdef dict dual_part
        cdef str k
        cdef double v
        
        real_part = args[0] * args[1]
        dual_part = {}
        
        if dual_args[0]:
            for k, v in dual_args[0].items():
                dual_part[k] = v * args[1]
        
        if dual_args[1]:
            for k, v in dual_args[1].items():
                dual_part[k] = dual_part.get(k, 0.0) + v * args[0]
                
        return Dual(real_part, dual_part)

    cdef _handle_divide(self, list args, list dual_args):
        cdef dict dual_part
        cdef str k
        cdef double v
        
        if args[1] == 0:
            raise ZeroDivisionError("Division by zero")
            
        real_part = args[0] / args[1]
        dual_part = {}
        
        if dual_args[0]:
            for k, v in dual_args[0].items():
                dual_part[k] = v / args[1]
                
        if dual_args[1]:
            for k, v in dual_args[1].items():
                dual_part[k] = dual_part.get(k, 0.0) - (args[0] * v) / (args[1] * args[1])
                
        return Dual(real_part, dual_part)

    cdef _handle_power(self, list args, list dual_args):
        cdef dict dual_part
        cdef str k
        cdef double v
        
        real_part = args[0] ** args[1]
        dual_part = {}
        
        if dual_args[0]:
            for k, v in dual_args[0].items():
                dual_part[k] = args[1] * (args[0] ** (args[1] - 1)) * v
                
        if dual_args[1]:
            for k, v in dual_args[1].items():
                dual_part[k] = dual_part.get(k, 0.0) + real_part * log(args[0]) * v
                
        return Dual(real_part, dual_part)