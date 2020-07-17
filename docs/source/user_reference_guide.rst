User Reference Guide
====================

:mod:`pyapprox` provides functions for surrogate modeling, sensitivity analysis, quadrature, inference, optimal experimental design and multi-fidelity modeling

Surrogate modeling
-----------------
Supervised learning 
^^^^^^^^^^^^^^^^^^^
The :func:`pyapprox.approximate.approximate` function produces 
response surface approximations from training data.

The `approximate` function supports the following methods

  - 'polynomial_chaos' See :func:`pyapprox.approximate.approximate_polynomial_chaos`
  - 'gaussian_process' See :func:`pyapprox.approximate.approximate_gaussian_process`

Supervised active learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :func:`pyapprox.approximate.adaptive_approximate` function produces adaptive
response surface approximations of a function ``fun``, As the approximation is built the function being approximated is sampled at locations that greedily minimize an estimate of error.

The `adaptive_approximate` function supports the following methods

  - 'sparse_grid' See :func:`pyapprox.approximate.adaptive_approximate_sparse_grid`
  - 'polynomial_chaos' See :func:`pyapprox.approximate.adaptive_approximate_polynomial_chaos`

Sensitivity analysis
--------------------
Surrogate based global sensitivity analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following functions can be used to extract sensitivity metrics analytically from a surrogates. 

  - 'sparse_grid'  See :func:`pyapprox.sensitivity_analysis.analyze_sensitivity_sparse_grid`
    
  - 'polynomial_chaos'  See :func:`pyapprox.sensitivity_analysis.analyze_sensitivity_polynomial_chaos`

The following functions can be used to visualize variance based sensitivity measures

  - :func:`pyapprox.sensitivity_analysis.plot_main_effects`

  - :func:`pyapprox.sensitivity_analysis.plot_total_effects`

  - :func:`pyapprox.sensitivity_analysis.plot_interaction_values`
    
Local sensitivity analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^
The :func:`pyapprox.sensitivity_analysis.analyze_sensitivity_morris` computes
morris sensitivity indices.

Quadrature
----------

Inference
---------

Optimal experimental design
---------------------------

Multi-fidelity modeling
-----------------------

