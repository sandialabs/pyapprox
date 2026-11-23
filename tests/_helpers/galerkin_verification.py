"""Checks shared by the Galerkin manufactured-solution catalogs.

Each physics' catalog (``tests/pde/models/galerkin/
test_verification_<physics>.py``) supplies its own system factory and
cases; these run a case through the verification harness and assert its
verdict, so a failure reports the observed rate and errors.

``converges``, ``recovers`` and ``stalls`` use scalar Lagrange elements;
the ``*_on`` forms take the basis factory, for vector and mixed bases,
and ``converges_on`` and ``stalls_on`` may measure some fields only.
"""

from typing import Any, Dict, Optional, Sequence, TypeVar, Union

from numpy.typing import NDArray

from pyapprox.pde.galerkin.basis import LagrangeBasis, MixedBasis
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.models.galerkin.verification import (
    BasisFactoryProtocol,
    GalerkinDiscretizer,
    Lagrange,
    MeshFamilyProtocol,
    StructuredBox,
    SystemFactoryProtocol,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Robin,
)
from pyapprox.pde.verification.domains import DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.study import (
    Converges,
    L2Error,
    Recovers,
    Stalls,
    verify,
)

_Np = NDArray[Any]
_B = TypeVar("_B", bound=Union[GalerkinBasisProtocol[_Np], MixedBasis[_Np]])
_Scalar = SystemFactoryProtocol[LagrangeBasis[_Np]]

_KINDS: Dict[str, ConditionProtocol] = {
    "D": Dirichlet(),
    "N": Neumann(),
    "R": Robin(1.0),
}


def coded(codes: str, domain: DomainProtocol) -> Dict[str, ConditionProtocol]:
    """Conditions by boundary part, one letter each in the domain's part
    order: D, N, or R (Robin with alpha 1)."""
    parts = domain.boundaries()
    if len(codes) != len(parts):
        raise ValueError(f"one code per part {list(parts)}, got {codes!r}")
    return {part: _KINDS[code] for code, part in zip(codes, parts)}


def converges_on(
    problem: ManufacturedProblem,
    meshes: MeshFamilyProtocol,
    basis: BasisFactoryProtocol[_B],
    factory: SystemFactoryProtocol[_B],
    rate: float,
    measured: Optional[Sequence[str]] = None,
) -> None:
    """The fitted L2 rate over the last three levels reaches ``rate``
    within 0.1, for the ``measured`` fields (default all)."""
    discretizer = GalerkinDiscretizer(meshes, basis, factory, measured=measured)
    result = verify(problem, discretizer, L2Error(), Converges(rate, last=3))
    assert result.passed, result.message


def recovers_on(
    problem: ManufacturedProblem,
    basis: BasisFactoryProtocol[_B],
    factory: SystemFactoryProtocol[_B],
    n: int,
    simplex: bool = False,
) -> None:
    """One structured mesh of ``n`` cells per direction reproduces the
    solution: L2 error below 1e-10."""
    discretizer = GalerkinDiscretizer(StructuredBox([n], simplex), basis, factory)
    result = verify(problem, discretizer, L2Error(), Recovers(1e-10))
    assert result.passed, result.message


def stalls_on(
    problem: ManufacturedProblem,
    meshes: MeshFamilyProtocol,
    basis: BasisFactoryProtocol[_B],
    factory: SystemFactoryProtocol[_B],
    measured: Optional[Sequence[str]] = None,
) -> None:
    """Wrong-model control: the fitted L2 rate of the ``measured`` fields
    (default all) stays below 0.5."""
    discretizer = GalerkinDiscretizer(meshes, basis, factory, measured=measured)
    result = verify(problem, discretizer, L2Error(), Stalls(below=0.5))
    assert result.passed, result.message


def converges(
    problem: ManufacturedProblem,
    meshes: MeshFamilyProtocol,
    degree: int,
    factory: _Scalar,
    rate: float,
) -> None:
    """:func:`converges_on` with Lagrange elements of ``degree``."""
    converges_on(problem, meshes, Lagrange(degree), factory, rate)


def recovers(
    problem: ManufacturedProblem,
    factory: _Scalar,
    n: int,
    simplex: bool = False,
) -> None:
    """:func:`recovers_on` with P2 (Q2) Lagrange elements."""
    recovers_on(problem, Lagrange(2), factory, n, simplex)


def stalls(
    problem: ManufacturedProblem,
    meshes: MeshFamilyProtocol,
    degree: int,
    factory: _Scalar,
) -> None:
    """:func:`stalls_on` with Lagrange elements of ``degree``."""
    stalls_on(problem, meshes, Lagrange(degree), factory)
