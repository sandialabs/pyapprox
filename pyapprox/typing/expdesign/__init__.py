"""
Experimental Design module for pyapprox.typing.

This module provides optimal experimental design (OED) functionality,
with a focus on Bayesian OED using expected information gain (KL divergence).

Submodules
----------
protocols
    Protocol definitions for OED components.
likelihood
    OED-specific likelihood wrappers.
evidence
    Evidence computation for Bayesian OED.
objective
    OED objective functions (KL-OED, etc.).
quadrature
    Quadrature samplers for expectation computation.
solver
    OED optimization solvers.
"""

__all__: list[str] = []
