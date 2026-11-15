"""Convergence verification of any discretization.

Problems are symbolic (sympy expressions on domains with named boundary
parts) and evaluate nothing; a stack supplies a discretizer whose
approximations integrate their own errors; checks judge the resulting
studies. Nothing here depends on a particular stack or array library.

Import from the modules: ``domains``, ``conditions``, ``problem``,
``protocols``, ``quantities``, ``exact``, ``study``. Expressions are
written in the symbols of :mod:`pyapprox.pde.expressions.symbols`.
"""
