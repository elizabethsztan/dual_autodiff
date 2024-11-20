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
