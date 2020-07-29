[![Actions Status](https://github.com/sandialabs/pyapprox/workflows/Build%20and%20Test%20Using%20Pip/badge.svg)](https://github.com/sandialabs/pyapprox/actions)
[![Actions Status](https://github.com/sandialabs/pyapprox/workflows/Build%20and%20Test%20Using%20Conda/badge.svg)](https://github.com/sandialabs/pyapprox/actions)

# PyApprox

Documentation
-------------
Online documentation can be found at [PyApprox](https://sandialabs.github.io/pyapprox/index.html)

Description
-----------
PyApprox provides flexible and efficient tools for high-dimensional approximation and uncertainty quantification. PyApprox implements methods addressing various issues surrounding high-dimensional parameter spaces and limited evaluations of expensive simulation models with the goal of facilitating simulation-aided knowledge discovery, prediction and design. Tools are provided for: (1) building surrogates using polynomial chaos expansions using least squares, compressive sensing and interpolation; (2) sparse grid interpolation and quadrature; (3) low-rank tensor-decompositions; (4) multi-fidelity approximation and sampling; (5) large-scale Bayesian inference; (6) numerical solvers for canonical ordinary and partial differential equations useful for demonstration purposes; (7) compressive sensing solvers; and (8) visualization. The code is intended to as a python toolbox but provides c++ code with Python interfaces to computationally expensive algorithms to increase performance.

Practical Application
---------------------
The software provides foundational numerical algorithms for approximation of multivariate functions and quantifying uncertainty in numerical models. The software is primarily used to build surrogates of generic functions. Often such functions are quantities of interest of numerical simulation models of, for example, sea-level change due to ice-sheet evolution, or ground-water flow. Once surrogates are generated they are used to undertake sensitivity analysis to identity important model parameters and to compute statistics of the variable model output caused by sources of uncertainty.

Method of Solution
------------------
The tools provided are based on mathematical algorithms for: (1) building surrogates using polynomial chaos expansions using least squares, compressive sensing and interpolation; (2) sparse grid interpolation and quadrature; (3) low-rank tensor-decompositions; (4) multi-fidelity approximation and sampling; (5) large-scale Bayesian inference; (6) numerical solvers for canonical ordinary and partial differential equations useful for demonstration purposes; and (7) compressive sensing solvers. The modularity of the code structure and function API are intended to facilitate flexible use and extension of the available tools. Numerous functions are provided to facilitate testing and benchmarking of algorithms.

Acknowledgements
----------------
This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA). The views, opinions and/or findings expressed are those of the author and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.