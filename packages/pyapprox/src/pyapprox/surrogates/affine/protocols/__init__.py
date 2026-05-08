"""Protocols for affine surrogates module."""

from pyapprox.surrogates.affine.protocols.adaptive import (
    AdaptiveIteratorProtocol,
    BasisIndexGeneratorProtocol,
    PrioritizedCandidateQueueProtocol,
)
from pyapprox.surrogates.affine.protocols.basis1d import (
    Basis1DHasDerivativesProtocol,
    Basis1DHasHessianProtocol,
    Basis1DHasJacobianProtocol,
    Basis1DHasQuadratureProtocol,
    Basis1DProtocol,
    Basis1DWithJacobianAndHessianProtocol,
    Basis1DWithJacobianProtocol,
    InterpolationBasis1DProtocol,
    OrthonormalPolynomial1DProtocol,
    PhysicalDomainBasis1DProtocol,
)
from pyapprox.surrogates.affine.protocols.expansion import (
    BasisExpansionHasHessianProtocol,
    BasisExpansionHasJacobianProtocol,
    BasisExpansionProtocol,
    FittableBasisExpansionProtocol,
    LinearSystemSolverProtocol,
    PCEStatisticsProtocol,
)
from pyapprox.surrogates.affine.protocols.index import (
    AdmissibilityCriteriaProtocol,
    CompositeAdmissibilityCriteriaProtocol,
    IndexGeneratorProtocol,
    IndexGrowthRuleProtocol,
    IndexSequenceProtocol,
    IterativeIndexGeneratorProtocol,
)
from pyapprox.surrogates.affine.protocols.multivariate_basis import (
    BasisHasHessianProtocol,
    BasisHasJacobianProtocol,
    BasisProtocol,
    BasisWithJacobianAndHessianProtocol,
    BasisWithJacobianProtocol,
    MultiIndexBasisProtocol,
    MultiIndexBasisWithJacobianAndHessianProtocol,
    MultiIndexBasisWithJacobianProtocol,
    TensorProductBasisProtocol,
)
from pyapprox.surrogates.affine.protocols.quadrature import (
    QuadratureRuleGeneratorProtocol,
    QuadratureRuleStatefulProtocol,
)
from pyapprox.surrogates.affine.protocols.refinement import (
    CostFunctionProtocol,
    RefinementCriteriaProtocol,
)
from pyapprox.surrogates.affine.protocols.solver import (
    ConstrainedSolverProtocol,
    QuantileSolverProtocol,
    RegularizedSolverProtocol,
    SparseSolverProtocol,
    WeightedSolverProtocol,
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
