"""
Variational inference module for pyapprox.

This module provides variational inference tools including:
- ELBO objective with joint quadrature design
- Covariance parameterization protocol (for Gaussian copula)

Variational distributions are conditional distributions from
``pyapprox.typing.probability.conditional`` (e.g., ConditionalGaussian,
ConditionalBeta, ConditionalIndependentJoint) with BasisExpansion
parameter functions.
"""

from pyapprox.typing.inverse.variational.elbo import (
    ELBOObjective,
    make_single_problem_elbo,
    make_discrete_group_elbo,
)

__all__ = [
    "ELBOObjective",
    "make_single_problem_elbo",
    "make_discrete_group_elbo",
]
