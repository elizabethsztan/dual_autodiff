from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

# Define extensions to be cythonized
extensions = [
    Extension(
        "dual_autodiff_x.dual",
        ["dual_autodiff_x/dual.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "dual_autodiff_x.tools",
        ["dual_autodiff_x/tools.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    # Package discovery
    packages=find_packages(),
    
    # Cython compilation
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
        }
    ),
    
    # Ensure source files aren't included in wheels
    package_data={
        'dual_autodiff_x': ['*.pyd', '*.so'],
    },
    exclude_package_data={
        'dual_autodiff_x': ['*.py', '*.pyx'],
    },
)