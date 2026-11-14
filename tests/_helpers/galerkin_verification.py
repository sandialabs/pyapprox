"""Checks shared by the Galerkin manufactured-solution catalogs.

Each physics' catalog (``tests/pde/models/galerkin/
test_verification_<physics>.py``) supplies its own system factory and
cases; these run a case through the verification harness and assert its
verdict, so a failure reports the observed rate and errors.
"""

from typing import Dict

from pyapprox.pde.models.galerkin.verification import (
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
    run_study,
    verify,
)

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


def converges(
    problem: ManufacturedProblem,
    meshes: MeshFamilyProtocol,
    degree: int,
    factory: SystemFactoryProtocol,
    rate: float,
) -> None:
    """The fitted L2 rate over the last three levels reaches ``rate``
    within 0.1."""
    discretizer = GalerkinDiscretizer(meshes, Lagrange(degree), factory)
    result = verify(problem, discretizer, L2Error(), Converges(rate, last=3))
    assert result.passed, result.message


def recovers(
    problem: ManufacturedProblem,
    factory: SystemFactoryProtocol,
    n: int,
    simplex: bool = False,
) -> None:
    """P2 on one structured mesh of ``n`` cells per direction reproduces
    the solution: L2 error below 1e-10."""
    discretizer = GalerkinDiscretizer(StructuredBox([n], simplex), Lagrange(2), factory)
    result = verify(problem, discretizer, L2Error(), Recovers(1e-10))
    assert result.passed, result.message


def stalls(
    problem: ManufacturedProblem,
    meshes: MeshFamilyProtocol,
    degree: int,
    factory: SystemFactoryProtocol,
) -> None:
    """Wrong-model control: the fitted L2 rate stays below 0.5."""
    discretizer = GalerkinDiscretizer(meshes, Lagrange(degree), factory)
    result = Stalls(below=0.5).verify(run_study(problem, discretizer, L2Error()))
    assert result.passed, result.message
