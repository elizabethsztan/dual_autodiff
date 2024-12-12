from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

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
    name="dual_autodiff_x",
    packages=find_packages(),
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