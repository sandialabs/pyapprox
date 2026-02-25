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
basis
    Multivariate basis functions with Jacobian/Hessian support.
expansions
    Basis expansions including PCE with statistics support.

Phases:
- Phase 1: Univariate basis protocols and implementations
- Phase 2: Index generation with composable admissibility criteria
- Phase 3: Multivariate basis with Jacobian/Hessian support
- Phase 4: Basis expansions and PCE statistics
- Phase 5: Linear system solvers
- Phase 6: Extended univariate polynomials (Laguerre, discrete, numeric)
- Phase 7: Adaptive indexing (priority queue, refinement criteria, basis generator)
"""

from pyapprox.surrogates.affine.basis import (
    FixedTensorProductQuadratureRule,
    # Core basis classes
    MultiIndexBasis,
    OrthonormalPolynomialBasis,
    # Quadrature rules
    QuadratureRule,
    TensorProductQuadratureRule,
)
from pyapprox.surrogates.affine.expansions import (
    # Base class
    BasisExpansion,
    # PCE
    PolynomialChaosExpansion,
    create_pce,
    # Statistics module
    pce_statistics,
)
from pyapprox.surrogates.affine.indices import (
    # Adaptive refinement
    AdaptiveIndexRefinement,
    # Admissibility criteria
    AdmissibilityCriteria,
    # Basis generator
    BasisIndexGenerator,
    ClenshawCurtisGrowthRule,
    CompositeCriteria,
    ConstantGrowthRule,
    # Refinement criteria
    CostFunction,
    CostWeightedRefinementCriteria,
    ExponentialCostFunction,
    ExponentialGrowthRule,
    HyperbolicIndexGenerator,
    # Generators
    IndexGenerator,
    # Growth rules
    IndexGrowthRule,
    IterativeIndexGenerator,
    LevelCostFunction,
    LevelRefinementCriteria,
    LinearGrowthRule,
    Max1DLevelsCriteria,
    MaxIndicesCriteria,
    MaxLevelCriteria,
    # Priority queue
    PriorityQueue,
    RefinementCriteria,
    UnitCostFunction,
    argsort_indices_lexiographically,
    compute_hyperbolic_indices,
    compute_hyperbolic_level_indices,
    # Utilities
    hash_index,
    indices_pnorm,
    sort_indices_lexiographically,
)
from pyapprox.surrogates.affine.protocols import (
    # Adaptive protocols
    AdaptiveIteratorProtocol,
    AdmissibilityCriteriaProtocol,
    Basis1DHasDerivativesProtocol,
    Basis1DHasHessianProtocol,
    Basis1DHasJacobianProtocol,
    Basis1DHasQuadratureProtocol,
    # Univariate protocols
    Basis1DProtocol,
    Basis1DWithJacobianAndHessianProtocol,
    Basis1DWithJacobianProtocol,
    BasisExpansionHasHessianProtocol,
    BasisExpansionHasJacobianProtocol,
    # Expansion protocols
    BasisExpansionProtocol,
    BasisHasHessianProtocol,
    BasisHasJacobianProtocol,
    BasisIndexGeneratorProtocol,
    # Multivariate protocols
    BasisProtocol,
    BasisWithJacobianAndHessianProtocol,
    BasisWithJacobianProtocol,
    CompositeAdmissibilityCriteriaProtocol,
    ConstrainedSolverProtocol,
    # Refinement protocols
    CostFunctionProtocol,
    FittableBasisExpansionProtocol,
    # Index protocols
    IndexGeneratorProtocol,
    IndexGrowthRuleProtocol,
    IterativeIndexGeneratorProtocol,
    LinearSystemSolverProtocol,
    MultiIndexBasisProtocol,
    MultiIndexBasisWithJacobianAndHessianProtocol,
    MultiIndexBasisWithJacobianProtocol,
    OrthonormalPolynomial1DProtocol,
    PCEStatisticsProtocol,
    PrioritizedCandidateQueueProtocol,
    QuantileSolverProtocol,
    RefinementCriteriaProtocol,
    RegularizedSolverProtocol,
    SparseSolverProtocol,
    TensorProductBasisProtocol,
    # Solver protocols
    WeightedSolverProtocol,
)
from pyapprox.surrogates.affine.solvers import (
    BasisPursuitDenoisingSolver,
    BasisPursuitSolver,
    ExpectileRegressionSolver,
    # Least squares
    LeastSquaresSolver,
    LinearlyConstrainedLstSqSolver,
    # Base
    LinearSystemSolver,
    # Sparse
    OMPSolver,
    OMPTerminationFlag,
    # Quantile
    QuantileRegressionSolver,
    RidgeRegressionSolver,
    SingleQoiSolverMixin,
)
from pyapprox.surrogates.affine.univariate import (
    CharlierPolynomial1D,
    Chebyshev1stKindPolynomial1D,
    Chebyshev2ndKindPolynomial1D,
    DiscreteChebyshevPolynomial1D,
    # Numeric polynomials
    DiscreteNumericOrthonormalPolynomial1D,
    GaussLobattoQuadratureRule,
    # Quadrature
    GaussQuadratureRule,
    HahnPolynomial1D,
    # Hermite
    HermitePolynomial1D,
    # Jacobi family
    JacobiPolynomial1D,
    # Discrete polynomials
    KrawtchoukPolynomial1D,
    # Laguerre
    LaguerrePolynomial1D,
    LegendrePolynomial1D,
    # Base class
    OrthonormalPolynomial1D,
    WeightedSamplePolynomial1D,
    charlier_recurrence,
    discrete_chebyshev_recurrence,
    evaluate_orthonormal_polynomial_1d,
    evaluate_orthonormal_polynomial_derivatives_1d,
    hahn_recurrence,
    hermite_recurrence,
    jacobi_recurrence,
    krawtchouk_recurrence,
    laguerre_recurrence,
    lanczos_recursion,
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
    # Expansion protocols
    "BasisExpansionProtocol",
    "BasisExpansionHasJacobianProtocol",
    "BasisExpansionHasHessianProtocol",
    "FittableBasisExpansionProtocol",
    "PCEStatisticsProtocol",
    "LinearSystemSolverProtocol",
    # Solver protocols
    "WeightedSolverProtocol",
    "SparseSolverProtocol",
    "RegularizedSolverProtocol",
    "QuantileSolverProtocol",
    "ConstrainedSolverProtocol",
    # Refinement protocols
    "CostFunctionProtocol",
    "RefinementCriteriaProtocol",
    # Adaptive protocols
    "AdaptiveIteratorProtocol",
    "PrioritizedCandidateQueueProtocol",
    "BasisIndexGeneratorProtocol",
    # Univariate implementations - base
    "OrthonormalPolynomial1D",
    "evaluate_orthonormal_polynomial_1d",
    "evaluate_orthonormal_polynomial_derivatives_1d",
    # Univariate - Jacobi family
    "JacobiPolynomial1D",
    "LegendrePolynomial1D",
    "Chebyshev1stKindPolynomial1D",
    "Chebyshev2ndKindPolynomial1D",
    "jacobi_recurrence",
    # Univariate - Hermite
    "HermitePolynomial1D",
    "hermite_recurrence",
    # Univariate - Laguerre
    "LaguerrePolynomial1D",
    "laguerre_recurrence",
    # Univariate - Discrete
    "KrawtchoukPolynomial1D",
    "krawtchouk_recurrence",
    "HahnPolynomial1D",
    "hahn_recurrence",
    "CharlierPolynomial1D",
    "charlier_recurrence",
    "DiscreteChebyshevPolynomial1D",
    "discrete_chebyshev_recurrence",
    # Univariate - Numeric
    "DiscreteNumericOrthonormalPolynomial1D",
    "WeightedSamplePolynomial1D",
    "lanczos_recursion",
    # Quadrature
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
    "ClenshawCurtisGrowthRule",
    "ConstantGrowthRule",
    "ExponentialGrowthRule",
    # Index generators
    "IndexGenerator",
    "IterativeIndexGenerator",
    "HyperbolicIndexGenerator",
    # Priority queue
    "PriorityQueue",
    # Refinement criteria
    "CostFunction",
    "UnitCostFunction",
    "LevelCostFunction",
    "ExponentialCostFunction",
    "RefinementCriteria",
    "LevelRefinementCriteria",
    "CostWeightedRefinementCriteria",
    # Basis generator
    "BasisIndexGenerator",
    # Adaptive refinement
    "AdaptiveIndexRefinement",
    # Multivariate basis
    "MultiIndexBasis",
    "OrthonormalPolynomialBasis",
    # Quadrature rules
    "QuadratureRule",
    "TensorProductQuadratureRule",
    "FixedTensorProductQuadratureRule",
    # Expansions
    "BasisExpansion",
    "PolynomialChaosExpansion",
    "create_pce",
    # Solvers - Base
    "LinearSystemSolver",
    "SingleQoiSolverMixin",
    # Solvers - Least squares
    "LeastSquaresSolver",
    "RidgeRegressionSolver",
    "LinearlyConstrainedLstSqSolver",
    # Solvers - Sparse
    "OMPSolver",
    "OMPTerminationFlag",
    "BasisPursuitSolver",
    "BasisPursuitDenoisingSolver",
    # Solvers - Quantile
    "QuantileRegressionSolver",
    "ExpectileRegressionSolver",
    # Statistics module
    "pce_statistics",
]
