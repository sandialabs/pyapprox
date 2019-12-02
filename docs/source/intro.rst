Introduction
============

PyApprox provides flexible and efficient tools for high-dimensional approximation and uncertainty quantification. PyApprox implements methods addressing various issues surrounding high-dimensional parameter spaces and limited evaluations of expensive simulation models with the goal of facilitating simulation-aided knowledge discovery, prediction and design. Tools are provided for: (1) building surrogates using polynomial chaos expansions using least squares, compressive sensing and interpolation; (2) sparse grid interpolation and quadrature; (3) low-rank tensor-decompositions; (4) multi-fidelity approximation and sampling; (5) large-scale Bayesian inference; (6) numerical solvers for canonical ordinary and partial differential equations useful for demonstration purposes; (7) compressive sensing solvers; and (8) visualization. The code is intended to as a python toolbox but provides c++ code with Python interfaces to computationally expensive algorithms to increase performance

The software provides foundational numerical algorithms for approximation of multivariate functions and quantifying uncertainty in numerical models. The software is primarily used to build surrogates of generic functions. Often such functions are quantities of interest of numerical simulation models of, for example, sea-level change due to ice-sheet evolution, or ground-water flow. Once surrogates are generated they are used to undertake sensitivity analysis to identity important model parameters and to compute statistics of the variable model output caused by sources of uncertainty.

Acknowledgments
===============
Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA-0003525. The views expressed in the article do not necessarily represent the views of the U.S. Department of Energy or the United States Government.
	     

