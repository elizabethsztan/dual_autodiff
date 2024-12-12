from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "dual_autodiff.dual",
        ["dual_autodiff/dual.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "dual_autodiff.tools",
        ["dual_autodiff/tools.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="dual_autodiff",
    ext_modules=cythonize(extensions, 
                         compiler_directives={
                             'language_level': "3",
                             'boundscheck': False,
                             'wraparound': False,
                             'initializedcheck': False,
                             'nonecheck': False,
                         }),
    zip_safe=False,
)