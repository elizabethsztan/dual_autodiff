.. dual_autodiff documentation master file, created by
   sphinx-quickstart on Sun Nov 24 17:32:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Automatic Differentiation with dual_autodiff 
============================================

*Author*: `Elizabeth S.Z. Tan <https://github.com/elizabethsztan>`_

This package handles automatic (analytic) differentiation and partial differentiation using dual numbers. 
It was created to work seamlessly with numerous NumPy functions and also allows the user to add their own functions to the tool_store.
There are two packages available with the exact same functionality - a pure Python package (dual_autodiff) and a Cython package (dual_autodiff_x).
The Cython package handles operations faster. 

Installation
-------------
To install the pure Python package, go to the root folder, python_package, run the following command in terminal to download the package.

.. code-block:: bash

   pip install -e .

To install the Cython package, go to the Cython root folder, cython_package, and run the same command.

Quick Start
-----------
Please see the demo for a detailed demonstration of the package capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   implementation
   Demo <demos/dual_autodiff_demo>
   Cython Analysis <demos/dual_autodiff_analysis>

Examples
--------
Check out the :doc:`demos/dual_autodiff_demo` notebook for comprehensive examples and usage patterns.
To see a comparison between the Cython and Python version of the package, please go to :doc:`demos/dual_autodiff_analysis`.

Requirements
---------------------------
- NumPy
- Python 3.7+
