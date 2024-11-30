.. dual_autodiff documentation master file, created by
   sphinx-quickstart on Sun Nov 24 17:32:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Automatic Differentiation with dual_autodiff 
============================================

*Author*: `Elizabeth S.Z. Tan <https://github.com/elizabethsztan>`_

This package handles automatic (analytic) differentiation and partial differentiation using dual numbers. 
It was created to work seamlessly with numerous NumPy functions and also allows the user to add their own functions to the tool_store.

Installation
-------------
.. code-block:: bash

   pip install -e .

Quick Start
-----------
Please see the demo for a detailed demonstration of the package capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   implementation
   Demo <demos/dual_autodiff>

Examples
--------
Check out the :doc:`demos/dual_autodiff` notebook for comprehensive examples and usage patterns.

Requirements (update these)
---------------------------
- NumPy
- Python 3.7+
