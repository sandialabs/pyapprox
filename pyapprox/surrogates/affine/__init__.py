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

from pyapprox.surrogates.affine.protocols import (
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
    # Expansion protocols
    BasisExpansionProtocol,
    BasisExpansionHasJacobianProtocol,
    BasisExpansionHasHessianProtocol,
    FittableBasisExpansionProtocol,
    PCEStatisticsProtocol,
    LinearSystemSolverProtocol,
    # Solver protocols
    WeightedSolverProtocol,
    SparseSolverProtocol,
    RegularizedSolverProtocol,
    QuantileSolverProtocol,
    ConstrainedSolverProtocol,
    # Refinement protocols
    CostFunctionProtocol,
    RefinementCriteriaProtocol,
    # Adaptive protocols
    AdaptiveIteratorProtocol,
    PrioritizedCandidateQueueProtocol,
    BasisIndexGeneratorProtocol,
)

from pyapprox.surrogates.affine.univariate import (
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
    # Laguerre
    LaguerrePolynomial1D,
    laguerre_recurrence,
    # Discrete polynomials
    KrawtchoukPolynomial1D,
    krawtchouk_recurrence,
    HahnPolynomial1D,
    hahn_recurrence,
    CharlierPolynomial1D,
    charlier_recurrence,
    DiscreteChebyshevPolynomial1D,
    discrete_chebyshev_recurrence,
    # Numeric polynomials
    DiscreteNumericOrthonormalPolynomial1D,
    WeightedSamplePolynomial1D,
    lanczos_recursion,
    # Quadrature
    GaussQuadratureRule,
    GaussLobattoQuadratureRule,
)

from pyapprox.surrogates.affine.indices import (
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
    ClenshawCurtisGrowthRule,
    ConstantGrowthRule,
    ExponentialGrowthRule,
    # Generators
    IndexGenerator,
    IterativeIndexGenerator,
    HyperbolicIndexGenerator,
    # Priority queue
    PriorityQueue,
    # Refinement criteria
    CostFunction,
    UnitCostFunction,
    LevelCostFunction,
    ExponentialCostFunction,
    RefinementCriteria,
    LevelRefinementCriteria,
    CostWeightedRefinementCriteria,
    # Basis generator
    BasisIndexGenerator,
    # Adaptive refinement
    AdaptiveIndexRefinement,
)

from pyapprox.surrogates.affine.basis import (
    # Core basis classes
    MultiIndexBasis,
    OrthonormalPolynomialBasis,
    # Quadrature rules
    QuadratureRule,
    TensorProductQuadratureRule,
    FixedTensorProductQuadratureRule,
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

from pyapprox.surrogates.affine.solvers import (
    # Base
    LinearSystemSolver,
    SingleQoiSolverMixin,
    # Least squares
    LeastSquaresSolver,
    RidgeRegressionSolver,
    LinearlyConstrainedLstSqSolver,
    # Sparse
    OMPSolver,
    OMPTerminationFlag,
    BasisPursuitSolver,
    BasisPursuitDenoisingSolver,
    # Quantile
    QuantileRegressionSolver,
    ExpectileRegressionSolver,
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
