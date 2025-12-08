"""Protocols for affine surrogates module."""

from pyapprox.typing.surrogates.affine.protocols.basis1d import (
    Basis1DProtocol,
    Basis1DHasJacobianProtocol,
    Basis1DHasHessianProtocol,
    Basis1DHasDerivativesProtocol,
    Basis1DWithJacobianProtocol,
    Basis1DWithJacobianAndHessianProtocol,
    OrthonormalPolynomial1DProtocol,
    Basis1DHasQuadratureProtocol,
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

__all__ = [
    # Univariate basis protocols
    "Basis1DProtocol",
    "Basis1DHasJacobianProtocol",
    "Basis1DHasHessianProtocol",
    "Basis1DHasDerivativesProtocol",
    "Basis1DWithJacobianProtocol",
    "Basis1DWithJacobianAndHessianProtocol",
    "OrthonormalPolynomial1DProtocol",
    "Basis1DHasQuadratureProtocol",
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
]
