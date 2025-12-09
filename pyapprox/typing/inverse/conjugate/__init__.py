"""
Conjugate prior solvers for exact posterior computation.

This package provides solvers for conjugate prior problems where
the posterior can be computed analytically:

- Gaussian-Gaussian: Linear model with Gaussian prior and noise
- Beta-Bernoulli: Binary outcomes with Beta prior
- Dirichlet-Multinomial: Categorical outcomes with Dirichlet prior

Classes
-------
DenseGaussianConjugatePosterior
    Gaussian conjugate posterior for linear observation models.
BetaConjugatePosterior
    Beta conjugate posterior for Bernoulli likelihood.
DirichletConjugatePosterior
    Dirichlet conjugate posterior for multinomial likelihood.
"""

from .gaussian import DenseGaussianConjugatePosterior
from .beta import BetaConjugatePosterior
from .dirichlet import DirichletConjugatePosterior

__all__ = [
    "DenseGaussianConjugatePosterior",
    "BetaConjugatePosterior",
    "DirichletConjugatePosterior",
]
