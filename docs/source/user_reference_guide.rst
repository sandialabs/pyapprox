User Reference Guide
====================

:mod:`pyapprox` provides functions for surrogate modeling, sensitivity analysis, quadrature, inference, optimal experimental design and multi-fidelity modeling

Surrogate modeling
------------------
:func:`pyapprox.approximate.approximate` (fun,univariate_variables,method,[,kwargs]) produces
response surface approximations also known as supervised machine learning models
of a function ``fun``.

The minimize function supports the following methods

  - 'sparse-grid' See :func:`pyapprox.approximate.approximate_sparse_grid`
  - 'polynomial-chaos' See :func:`pyapprox.approximate.approximate_polynomial_chaos`

Sensitivity analysis
--------------------

Quadrature
----------

Inference
---------

Optimal experimental design
---------------------------

Multi-fidelity modeling
-----------------------

