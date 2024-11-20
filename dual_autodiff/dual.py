import numpy as np

class Dual:

    def __init__(self, real_component, dual_component):

        self.real = real_component
        self.dual = dual_component
    
    def __repr__(self):
        return f"Dual(real={self.real}, dual={self.dual})"
    
    def __add__(self, other):
        if isinstance (other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        else:
            return Dual(self.real + other, self.dual)
        
    __radd__ = __add__ 

    def __sub__(self, other):
        if isinstance (other, Dual):
            return Dual(self.real - other.real, self.dual - other.dual)
        else:
            return Dual(self.real - other, self.dual)
    
    def __rsub__(self, other):
        return Dual(other - self.real, self.dual)
    
    def __mul__(self, other):
        if isinstance (other, Dual):
            real_component = self.real * other.real
            dual_component = self.real * other.dual + self.dual * other.real
            return Dual(real_component, dual_component)
        else:
            return Dual(self.real * other, self.dual * other)

    def __truediv__(self, other):
        if isinstance (other, Dual):
            if other.real == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            real_component = self.real / other.real
            dual_component = (self.dual * other.real - self.real * other.dual) / (other.real * other.real)
            return Dual(real_component, dual_component)
        else:
            return Dual(self.real / other, self.dual / other)

    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __eq__(self, other):
        if isinstance(other, Dual):
            return self.real == other.real and self.dual == other.dual
        return False

    def __pow__(self, exponent):
        real_component = self.real**exponent
        dual_component = exponent * (self.real**(exponent - 1)) * self.dual
        return Dual(real_component, dual_component)

    def sin(self):
        real_component = np.sin(self.real)
        dual_component = np.cos(self.real) * self.dual 
        return Dual(real_component, dual_component)

    def cos(self):
        real_component = np.cos(self.real)
        dual_component = - np.sin(self.real) * self.dual 
        return Dual(real_component, dual_component)

    def tan(self):
        real_component = np.tan(self.real)
        dual_component = self.dual / (np.cos(self.real) * np.cos(self.real))
        return Dual(real_component, dual_component)
    
    def exp(self):
        real_component = np.exp(self.real)
        dual_component = np.exp(self.real) * self.dual
        return Dual(real_component, dual_component)

    def log(self):
        if self.real <= 0:
            raise ValueError("Logarithm is undefined for non-positive values.")
        real_component = np.log(self.real)
        dual_component = self.dual / self.real 
        return Dual(real_component, dual_component)

    def sinh(self):
        real_component = np.sinh(self.real)
        dual_component = np.cosh(self.real) * self.dual
        return Dual(real_component, dual_component)

    def cosh(self):
        real_component = np.cosh(self.real)
        dual_component = np.sinh(self.real) * self.dual
        return Dual(real_component, dual_component)

    def tanh(self):
        real_component = np.tanh(self.real)
        dual_component = self.dual / (np.cosh(self.real) ** np.cosh(self.real))
        return Dual(real_component, dual_component)
    
    def arcsin(self):
        if self.real < -1 or self.real > 1:
            raise ValueError("arcsin is undefined for values outside [-1, 1].")
        real_component = np.arcsin(self.real)
        dual_component = self.dual / np.sqrt(1 - self.real * self.real)
        return Dual(real_component, dual_component)

    def arccos(self):
        if self.real < -1 or self.real > 1:
            raise ValueError("arccos is undefined for values outside [-1, 1].")
        real_component = np.arccos(self.real)
        dual_component = -self.dual / np.sqrt(1 - self.real * self.real)
        return Dual(real_component, dual_component)

    def arctan(self):
        real_component = np.arctan(self.real)
        dual_component = self.dual / (1 + self.real * self.real)
        return Dual(real_component, dual_component)

    def sqrt(self):
        if self.real < 0:
            raise ValueError("Square root is undefined for negative values.")
        real_component = np.sqrt(self.real)
        dual_component = self.dual / (2 * np.sqrt(self.real))
        return Dual(real_component, dual_component)

x = Dual (2, 3)
print(x**2)