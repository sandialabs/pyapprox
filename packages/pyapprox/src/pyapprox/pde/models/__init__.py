"""Parameterized PDE models: solver + parameterization wiring.

This layer sits above the solver packages (collocation, galerkin) and
the parameterizations package. It owns everything that requires BOTH a
physics/solver and a parameterization: parameterized ODE-residual
adapters and forward models. Non-parameterized solver entry points
(``CollocationModel``, ``GalerkinModel``) stay solver-level.
"""
