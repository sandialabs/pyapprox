"""Affine surrogates module.

This module provides protocol-based implementations of affine surrogates,
which are approximations that depend linearly on their hyperparameters.

Submodules
----------
protocols
    Protocol definitions for basis functions, expansions, indices, etc.
univariate
    Univariate (1D) basis functions including orthonormal polynomials.
indices
    Multi-index generation with composable admissibility criteria.

Phases:
- Phase 1: Univariate basis protocols and implementations
- Phase 2: Index generation with composable admissibility criteria
"""

from pyapprox.typing.surrogates.affine.protocols import (
    # Univariate protocols
    Basis1DProtocol,
    Basis1DHasJacobianProtocol,
    Basis1DHasHessianProtocol,
    Basis1DHasDerivativesProtocol,
    Basis1DWithJacobianProtocol,
    Basis1DWithJacobianAndHessianProtocol,
    OrthonormalPolynomial1DProtocol,
    Basis1DHasQuadratureProtocol,
    # Multivariate protocols
    BasisProtocol,
    BasisHasJacobianProtocol,
    BasisHasHessianProtocol,
    BasisWithJacobianProtocol,
    BasisWithJacobianAndHessianProtocol,
    MultiIndexBasisProtocol,
    TensorProductBasisProtocol,
    MultiIndexBasisWithJacobianProtocol,
    MultiIndexBasisWithJacobianAndHessianProtocol,
    # Index protocols
    IndexGeneratorProtocol,
    IterativeIndexGeneratorProtocol,
    AdmissibilityCriteriaProtocol,
    IndexGrowthRuleProtocol,
    CompositeAdmissibilityCriteriaProtocol,
)

from pyapprox.typing.surrogates.affine.univariate import (
    # Base class
    OrthonormalPolynomial1D,
    evaluate_orthonormal_polynomial_1d,
    evaluate_orthonormal_polynomial_derivatives_1d,
    # Jacobi family
    JacobiPolynomial1D,
    LegendrePolynomial1D,
    Chebyshev1stKindPolynomial1D,
    Chebyshev2ndKindPolynomial1D,
    jacobi_recurrence,
    # Hermite
    HermitePolynomial1D,
    hermite_recurrence,
    # Quadrature
    GaussQuadratureRule,
    GaussLobattoQuadratureRule,
)

from pyapprox.typing.surrogates.affine.indices import (
    # Utilities
    hash_index,
    compute_hyperbolic_indices,
    compute_hyperbolic_level_indices,
    sort_indices_lexiographically,
    argsort_indices_lexiographically,
    indices_pnorm,
    # Admissibility criteria
    AdmissibilityCriteria,
    MaxLevelCriteria,
    Max1DLevelsCriteria,
    MaxIndicesCriteria,
    CompositeCriteria,
    # Growth rules
    IndexGrowthRule,
    LinearGrowthRule,
    DoublePlusOneGrowthRule,
    ConstantGrowthRule,
    ExponentialGrowthRule,
    # Generators
    IndexGenerator,
    IterativeIndexGenerator,
    HyperbolicIndexGenerator,
)

__all__ = [
    # Univariate protocols
    "Basis1DProtocol",
    "Basis1DHasJacobianProtocol",
    "Basis1DHasHessianProtocol",
    "Basis1DHasDerivativesProtocol",
    "Basis1DWithJacobianProtocol",
    "Basis1DWithJacobianAndHessianProtocol",
    "OrthonormalPolynomial1DProtocol",
    "Basis1DHasQuadratureProtocol",
    # Multivariate protocols
    "BasisProtocol",
    "BasisHasJacobianProtocol",
    "BasisHasHessianProtocol",
    "BasisWithJacobianProtocol",
    "BasisWithJacobianAndHessianProtocol",
    "MultiIndexBasisProtocol",
    "TensorProductBasisProtocol",
    "MultiIndexBasisWithJacobianProtocol",
    "MultiIndexBasisWithJacobianAndHessianProtocol",
    # Index protocols
    "IndexGeneratorProtocol",
    "IterativeIndexGeneratorProtocol",
    "AdmissibilityCriteriaProtocol",
    "IndexGrowthRuleProtocol",
    "CompositeAdmissibilityCriteriaProtocol",
    # Univariate implementations
    "OrthonormalPolynomial1D",
    "evaluate_orthonormal_polynomial_1d",
    "evaluate_orthonormal_polynomial_derivatives_1d",
    "JacobiPolynomial1D",
    "LegendrePolynomial1D",
    "Chebyshev1stKindPolynomial1D",
    "Chebyshev2ndKindPolynomial1D",
    "jacobi_recurrence",
    "HermitePolynomial1D",
    "hermite_recurrence",
    "GaussQuadratureRule",
    "GaussLobattoQuadratureRule",
    # Index utilities
    "hash_index",
    "compute_hyperbolic_indices",
    "compute_hyperbolic_level_indices",
    "sort_indices_lexiographically",
    "argsort_indices_lexiographically",
    "indices_pnorm",
    # Admissibility criteria
    "AdmissibilityCriteria",
    "MaxLevelCriteria",
    "Max1DLevelsCriteria",
    "MaxIndicesCriteria",
    "CompositeCriteria",
    # Growth rules
    "IndexGrowthRule",
    "LinearGrowthRule",
    "DoublePlusOneGrowthRule",
    "ConstantGrowthRule",
    "ExponentialGrowthRule",
    # Index generators
    "IndexGenerator",
    "IterativeIndexGenerator",
    "HyperbolicIndexGenerator",
]
