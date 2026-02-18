"""Protocols for affine surrogates module."""

from pyapprox.typing.surrogates.affine.protocols.basis1d import (
    Basis1DProtocol,
    Basis1DHasJacobianProtocol,
    Basis1DHasHessianProtocol,
    Basis1DHasDerivativesProtocol,
    Basis1DWithJacobianProtocol,
    Basis1DWithJacobianAndHessianProtocol,
    InterpolationBasis1DProtocol,
    OrthonormalPolynomial1DProtocol,
    Basis1DHasQuadratureProtocol,
    PhysicalDomainBasis1DProtocol,
)

from pyapprox.typing.surrogates.affine.protocols.multivariate_basis import (
    BasisProtocol,
    BasisHasJacobianProtocol,
    BasisHasHessianProtocol,
    BasisWithJacobianProtocol,
    BasisWithJacobianAndHessianProtocol,
    MultiIndexBasisProtocol,
    TensorProductBasisProtocol,
    MultiIndexBasisWithJacobianProtocol,
    MultiIndexBasisWithJacobianAndHessianProtocol,
)

from pyapprox.typing.surrogates.affine.protocols.index import (
    IndexGeneratorProtocol,
    IterativeIndexGeneratorProtocol,
    AdmissibilityCriteriaProtocol,
    IndexGrowthRuleProtocol,
    IndexSequenceProtocol,
    CompositeAdmissibilityCriteriaProtocol,
)

from pyapprox.typing.surrogates.affine.protocols.expansion import (
    BasisExpansionProtocol,
    BasisExpansionHasJacobianProtocol,
    BasisExpansionHasHessianProtocol,
    FittableBasisExpansionProtocol,
    PCEStatisticsProtocol,
    LinearSystemSolverProtocol,
)

from pyapprox.typing.surrogates.affine.protocols.solver import (
    WeightedSolverProtocol,
    SparseSolverProtocol,
    RegularizedSolverProtocol,
    QuantileSolverProtocol,
    ConstrainedSolverProtocol,
)

from pyapprox.typing.surrogates.affine.protocols.refinement import (
    CostFunctionProtocol,
    RefinementCriteriaProtocol,
)

from pyapprox.typing.surrogates.affine.protocols.adaptive import (
    AdaptiveIteratorProtocol,
    PrioritizedCandidateQueueProtocol,
    BasisIndexGeneratorProtocol,
)

from pyapprox.typing.surrogates.affine.protocols.quadrature import (
    QuadratureRuleGeneratorProtocol,
    QuadratureRuleStatefulProtocol,
)

__all__ = [
    # Univariate basis protocols
    "Basis1DProtocol",
    "Basis1DHasJacobianProtocol",
    "Basis1DHasHessianProtocol",
    "Basis1DHasDerivativesProtocol",
    "Basis1DWithJacobianProtocol",
    "Basis1DWithJacobianAndHessianProtocol",
    "InterpolationBasis1DProtocol",
    "OrthonormalPolynomial1DProtocol",
    "Basis1DHasQuadratureProtocol",
    "PhysicalDomainBasis1DProtocol",
    # Multivariate basis protocols
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
    "IndexSequenceProtocol",
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
    # Quadrature protocols
    "QuadratureRuleGeneratorProtocol",
    "QuadratureRuleStatefulProtocol",
]
