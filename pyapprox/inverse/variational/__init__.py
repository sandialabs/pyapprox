"""
Variational inference module for pyapprox.

This module provides variational inference tools including:
- ELBO objective with joint quadrature design
- Covariance parameterization protocol (for Gaussian copula)

Variational distributions are conditional distributions from
``pyapprox.probability.conditional`` (e.g., ConditionalGaussian,
ConditionalBeta, ConditionalIndependentJoint) with BasisExpansion
parameter functions.
"""

from pyapprox.inverse.variational.elbo import (
    ELBOObjective,
    make_discrete_group_elbo,
    make_single_problem_elbo,
)
from pyapprox.inverse.variational.fitter import (
    VariationalFitter,
    VIFitResult,
)

__all__ = [
    "ELBOObjective",
    "make_single_problem_elbo",
    "make_discrete_group_elbo",
    "VIFitResult",
    "VariationalFitter",
]
