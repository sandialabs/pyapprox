"""Parameterized models for the Galerkin solver.

The modules here load the Galerkin discretization (and skfem through it),
so this package re-exports nothing: import from the leaf modules
(``essential_boundary_flux``, ``parameterized_residual``,
``physics_adapter``, ``steady_residual``, ``transient``).
"""
