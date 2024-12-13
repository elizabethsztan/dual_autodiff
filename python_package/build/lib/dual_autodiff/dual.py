import numpy as np
from typing import Dict, Union
from dual_autodiff.tools import get_functions, remove_function

class Dual:
    """
    Dual numbers class to compute derivatives or partial derivatives of functions.
    The dual component must be inputted as a dictionary to allow multiple components of the dual number.
    This allows easy computation of partial derivatives.

    Args:
        real_component (float): real component of dual number.
        dual_component (Dict[str, float]): dual components.
    
    Raises:
        TypeError: If `real_component` is not a number or `dual_component` is not a dictionary.

    Example:
        >>> from dual_autodiff import Dual
        >>> x_dual = Dual (3, {'x': 1})
        >>> def f(x):
        >>>     return x ** 3
        >>> print('The derivative of f at x = 3 is: '(x_dual).dual['x'])
        The derivative of f at x = 3 is: 12
    
    For more examples please see the demo.
    """
    def __init__(self, real_component: Union[float, int], dual_component: Dict[str, float]):
        """
        Args:
            real_component (float): real component of dual number.
            dual_component (Dict[str, float]): dual components.
        
        Raises:
            TypeError: If `real_component` is not a number or `dual_component` is not a dictionary.
        """
        if not isinstance(real_component, (float, int)):
            raise TypeError(
                f"Expected 'real_component' to be a float or int, but got {type(real_component).__name__}."
            )
        if not isinstance(dual_component, dict):
            raise TypeError(
                f"Expected 'dual_component' to be a dictionary, but got {type(dual_component).__name__}."
        )
        self.real = real_component
        self.dual = dual_component if isinstance(dual_component, dict) else {}
    
    def __repr__(self):
        """
        Returns a string representation of the Dual number.

        Returns:
            string: a string in the format "Dual(real=<real>, dual=<dual>)".
        """
        return f"Dual(real={self.real}, dual={self.dual})"

    def __add__(self, other):
        """
        Adds a Dual number or a scalar to the current Dual number.

        If the `other` is a Dual number, the real and dual components of both
        numbers are added. If `other` is a scalar, only the real part is updated,
        and the dual part remains unchanged.

        Args:
            other (Union[Dual, float, int]): the value to be added. Can be either
            another Dual number or a scalar (float or int).

        Returns:
            Dual: the Dual number that is the result of the addition.
        """
        if isinstance(other, Dual):
            new_dual = self.dual.copy()
            for var, val in other.dual.items():
                new_dual[var] = new_dual.get(var, 0) + val #add the different dual parts component-wise
            return Dual(self.real + other.real, new_dual)
        else:
            return Dual(self.real + other, self.dual.copy())
    
    __radd__ = __add__ #Reverse addition to allow for scalar + dual

    def __mul__(self, other):
        """
        Multiplies a Dual number or a scalar to the current Dual number.
        For a base x = a + bε and multiplier y = c + dε:
        - Real part: a * c
        - Dual part: a * d + b * c

        Args:
            other (Union[Dual, float, int]): the value to be multiplied with. Can be either
            another Dual number or a scalar (float or int).

        Returns:
            Dual: the Dual number that is the result of the multiplication.
        """
        if isinstance(other, Dual):
            new_dual = {}
            all_vars = set(self.dual.keys()) | set(other.dual.keys()) #collect all Dual variables
            for var in all_vars:
                new_dual[var] = (self.real * other.dual.get(var, 0) + 
                                self.dual.get(var, 0) * other.real)
            return Dual(self.real * other.real, new_dual)
        else:
            return Dual(self.real * other, 
                       {k: v * other for k, v in self.dual.items()})
    
    __rmul__ = __mul__ #Reverse addition to allow for scalar * Dual

    def __sub__(self, other):
        """
        Subtracts a dual number or a scalar to the current dual number.
        If the `other` is a dual number, the real and dual components of the current
        dual number are reduced by the corresponding components of `other`. If `other`
        is a scalar, only the real part is updated.

        Args:
            other (Union[Dual, float, int]): The value to be subtracted with. Can be either
            another Dual number or a scalar (float or int).

        Returns:
            Dual: the dual number that is the result of the subtraction.
        """

        if isinstance(other, Dual):
            new_dual = self.dual.copy()
            for var, val in other.dual.items():
                new_dual[var] = new_dual.get(var, 0) - val
            return Dual(self.real - other.real, new_dual)
        else:
            return Dual(self.real - other, self.dual.copy())
    
    def __rsub__(self, other):
        """
        Subtracts a dual number from a scalar.
        For a scalar `other` and a Dual number x = a + bε:
        - Real part: other - a
        - Dual part: -b (negates the dual part since the scalar has no dual component).

        Args:
            other (Union[float, int]): the scalar that the Dual number will be subtracted from.

        Returns:
             Dual: the Dual number that is the result of the subtraction.
        """
        return Dual(other - self.real, {k: -v for k, v in self.dual.items()})

    def __truediv__(self, other):
        """
        Divides a Dual number by a scalar or another Dual number.
        For a base x = a + bε and divisor y = c + dε:
        - Real part: a / c
        - Dual part: (b * c - a * d) / c^2

        If `other` is a scalar, both the real and dual parts are divided by `other`.

        Args:
            other (Union[Dual, float, int]): the value that the dual number will
            be divided by. Can be either another dual number or a scalar (float or int).

        Raises:
            ZeroDivisionError: if the dual number is being divided by another dual
            number whose real component is zero. 
            ZeroDivisionError: if the dual number is being divided by a scalar equal
            to zero.

        Returns:
            Dual: the dual number that is the result of the division.
        """
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
        """
        Divides a scalar by a dual number.
        For a scalar `other` and a dual number x = a + bε:
        - Real part: other / a
        - Dual part: -other * b / a^2

        Args:
            other (Union[float, int]): the scalar that will be divided by the dual number.

        Raises:
            ZeroDivisionError: if the scalar is being divided by a dual number whose
            real component is equal to zero.

        Returns:
            Dual: the dual number that is the result of the division.
        """
        if self.real == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")
        real_part = other / self.real
        dual_part = {k: -other * v / (self.real * self.real) 
                    for k, v in self.dual.items()}
        return Dual(real_part, dual_part)

    def __pow__(self, other):
        """
        Powers where the base is a dual number.

        For a base x = a + bε:
        - If the exponent n is a scalar:
        Real part: a^n
        Dual part: n * a^(n-1) * b
        - If the exponent y = c + dε is also a dual number:
        Real part: a^c
        Dual part: a^c * [ (b * c) / a + d * ln(a) ]

        Args:
            other (Union[Dual, float, int]): the exponent. Can be a scalar or a dual number.

        Raises:
            ValueError: if raising to the power of a dual number, the base must be 
            a scalar or a dual number with a positive real component.

        Returns:
            Dual: the dual number that is the result of the power calculation.
        """
        if isinstance(other, Dual):
            if self.real <= 0:
                raise ValueError("For Dual number exponents, base must be positive")
            real_part = self.real ** other.real
            dual_part = {}
            all_vars = set(self.dual.keys()) | set(other.dual.keys())
            for var in all_vars:
                base_term = other.real * (self.real ** (other.real - 1)) * self.dual.get(var, 0)
                exp_term = real_part * np.log(self.real) * other.dual.get(var, 0)
                dual_part[var] = base_term + exp_term
            return Dual(real_part, dual_part)
        else:
            real_part = self.real ** other
            dual_part = {k: other * (self.real ** (other - 1)) * v 
                        for k, v in self.dual.items()}
            return Dual(real_part, dual_part)

    def __rpow__(self, other):  
        """
        Raises a scalar to the power of the current dual number.

        For a scalar `other` and a dual number x = a + bε:
        - Real part: other^a
        - Dual part: ln(other) * other^a * b

        This operation uses the formula:
        - y^x = exp(x * ln(y))

        Args:
            other (Union[float, int]): The base scalar to be raised to the power
            of the current dual number.

        Returns:
            Dual: the dual number that is the result of the power calculation.

        Raises:
            ValueError: If `other` is non-positive (logarithm is undefined for non-positive bases).
        """
        if other <= 0:
            raise ValueError("Base must be positive to raise it to a Dual exponent.")
        
        if not isinstance(other, (int, float)):
            return NotImplemented
        real_part = other ** self.real
        dual_part = {k: real_part * np.log(other) * v 
                    for k, v in self.dual.items()}
        return Dual(real_part, dual_part)
    
    def __neg__(self):
        """
        Negates a dual number. Inverts the sign on both the real and dual components.

        Returns:
            Dual: a dual number where the signs on the real and dual components of the 
            original dual number have been inverted.
        """
        return Dual(-self.real, {k: -v for k, v in self.dual.items()})

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handles NumPy universal functions (ufuncs) for the Dual number class.

        Allows the Dual number class to be operated on by numpy functions(e.g., np.add, np.multiply, np.sin)
        by overriding how these operations are applied to dual numbers.

        Args:
            ufunc (numpy.ufunc): the numpy universal function being applied.
            method (str): the method of the ufunc being called (only "__call__" is supported).
            *inputs: The input arguments for the ufunc, which may include dual numbers
                    or scalars.
            **kwargs: Additional keyword arguments passed to the ufunc (e.g., out, where).

        Returns:
            Dual or NotImplemented:
                - A new Dual number if the ufunc is supported and successfully applied.
                - `NotImplemented` if the ufunc or method is unsupported or if an error occurs.
        """
        if method != "__call__": #does not support other methods such as reduction or accumulation
            return NotImplemented
        
        implementations = get_functions() #functions stored in tools module
        args = [] #stores real components
        dual_args = [] #stores dual components 
        #iterate over inputs provided by the ufunc to split into real and dual classes
        #functions operate on real part, derivatives operate on dual part
        for arg in inputs:
            if isinstance(arg, Dual):
                args.append(arg.real)
                dual_args.append(arg.dual)
            else:
                args.append(arg)
                dual_args.append({})
        
        #functionality for np.add etc.
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
        """
        Implements `np.add` for dual numbers.

        Args:
            args (list): real components of the dual numbers.
            dual_args (list): dual components of the dual numbers.

        Returns:
            Dual: a dual number with the real components being the sum of real
            inputs and the dual components the sum of dual inputs represented as Dict.
        """
        real_part = args[0] + args[1]
        dual_part = {}
        for d in dual_args:
            for k, v in d.items():
                dual_part[k] = dual_part.get(k, 0) + v
        return Dual(real_part, dual_part)

    def _handle_multiply(self, args, dual_args):
        """
        Implements `np.multiply` for dual numbers.

        Multiplies the real parts and computes the dual part using contributions
        from both operands.

        Args:
            args (list): real components of the dual numbers.
            dual_args (list): dual components of the dual numbers represented as Dict.

        Returns:
            Dual: A new Dual number with:
                - Real part: Product of the real parts of the inputs.
                - Dual part: Computed using the product rule:
                    - For each variable in the dual components:
                    - Contribution from the first operand: v1 * real_part2
                    - Contribution from the second operand: v2 * real_part1
        """
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
        """
        Implements `np.divide` for dual numbers.

        Divides the real parts and computes the dual part based on the derivative
        of the quotient.

        Args:
            args (list): real components of the dual numbers.
            dual_args (list): dual components of the dual numbers represented as Dict.

        Returns:
            Dual: A new Dual number with:
                - Real part: Quotient of the real parts of the inputs.
                - Dual part: Computed using the quotient rule:
                    - For each variable in the dual components:
                    - Contribution from the numerator: v1 / real_part2
                    - Contribution from the denominator: -(real_part1 * v2) / (real_part2^2)

        Raises:
            ZeroDivisionError: If the denominator's real part is zero.
        """
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
        """
        Implements `np.power` for dual numbers.

        Args:
            args (list): real components of the dual numbers.
            dual_args (list): dual components of the dual numbers represented as Dict.

        Returns:
            Dual: A new Dual number with:
                - Real part: Base raised to the power of the exponent.
                - Dual part: Contributions from the base and the exponent:
                    - If the base is Dual: (exponent * base^(exponent - 1)) * dual_base
                    - If the exponent is Dual: (real_part * ln(base)) * dual_exponent

        Raises:
            ValueError: If the base is non-positive and the exponent is fractional or Dual.
        """
        real_part = args[0] ** args[1]
        dual_part = {}
        
        if dual_args[0]:  # Base is Dual
            for k, v in dual_args[0].items():
                dual_part[k] = args[1] * (args[0] ** (args[1] - 1)) * v
                
        if dual_args[1]:  # Exponent is Dual
            for k, v in dual_args[1].items():
                dual_part[k] = dual_part.get(k, 0) + real_part * np.log(args[0]) * v
                
        return Dual(real_part, dual_part)
