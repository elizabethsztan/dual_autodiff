import numpy as np

class Dual:

    def __init__(self, real_component, dual_component):

        self.real = real_component
        self.dual = dual_component
    
    def __repr__(self):
        return f"Dual(real={self.real}, dual={self.dual})"
    
    def __add__(self, other):
        return Dual(self.real + other.real, self.dual + other.dual)

    def __sub__(self, other):
        return Dual(self.real - other.real, self.dual - other.dual)
    
    def __mul__(self, other):
        real_component = self.real * other.real
        dual_component = self.real * other.dual + self.dual * other.real
        return Dual(real_component, dual_component)

    def __truediv__(self, other):
        if other.real == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")
        real_component = self.real / other.real
        dual_component = (self.dual * other.real - self.real * other.dual) / (other.real * other.real)
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
        dual_component = np.tan(self.real) * self.dual / (np.cos(self.real) * np.cos(self.real))
        return Dual(real_component, dual_component)
    
    def exp(self):
        real_component = np.exp(self.real)
        dual_component = np.exp(self.real) * self.dual
        return Dual(real_component, dual_component)

    def log(self):
        real_component = np.log(self.real)
        dual_component = self.dual / self.real 
        return Dual(real_component, dual_component)



x = Dual (9, 2)
y = Dual (4, 2)
print(x)
print(x.exp())
print(x/y)
#print(y+3)