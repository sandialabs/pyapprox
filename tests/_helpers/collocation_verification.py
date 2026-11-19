"""Checks shared by the collocation manufactured-solution catalogs.

Each physics' catalog (``tests/pde/models/collocation/
test_verification_<physics>.py``) supplies its own system factory and
cases; these run a case through the verification harness on Chebyshev
boxes and assert its verdict, so a failure reports the observed decay or
rate and the errors.
"""

from typing import Dict, Optional, Sequence

from pyapprox.pde.models.collocation.verification import (
    ChebyshevBox,
    CollocationDiscretizer,
    DecaysExponentially,
    SystemFactoryProtocol,
    run_node_study,
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
    L2Error,
    Recovers,
    Stalls,
    run_study,
    verify,
)
from pyapprox.util.backends.protocols import Array, Backend

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


def recovers(
    problem: ManufacturedProblem,
    factory: SystemFactoryProtocol,
    n: int,
    bkd: Backend[Array],
    initial: float = 0.0,
) -> None:
    """``n`` nodes per direction reproduce the solution: L2 error below
    1e-10. Solves start from the constant ``initial``."""
    discretizer = CollocationDiscretizer(
        ChebyshevBox([n], bkd), factory, initial=initial
    )
    result = verify(problem, discretizer, L2Error(), Recovers(1e-10))
    assert result.passed, result.message


def decays(
    problem: ManufacturedProblem,
    factory: SystemFactoryProtocol,
    sizes: Sequence[int],
    rate: float,
    bkd: Backend[Array],
    last: Optional[int] = None,
    initial: float = 0.0,
) -> None:
    """The L2 error falls exponentially over ``sizes`` nodes per
    direction, at least ``rate`` per node. Solves start from the
    constant ``initial``."""
    discretizer = CollocationDiscretizer(
        ChebyshevBox(sizes, bkd), factory, initial=initial
    )
    study = run_node_study(problem, discretizer, L2Error())
    result = DecaysExponentially(rate, last).verify(study)
    assert result.passed, result.message


def stalls(
    problem: ManufacturedProblem,
    factory: SystemFactoryProtocol,
    sizes: Sequence[int],
    bkd: Backend[Array],
    initial: float = 0.0,
) -> None:
    """Wrong-model control: the fitted algebraic L2 rate stays below 0.5,
    so the error does not fall as a correct model's would. Solves start
    from the constant ``initial``."""
    discretizer = CollocationDiscretizer(
        ChebyshevBox(sizes, bkd), factory, initial=initial
    )
    result = Stalls(below=0.5).verify(run_study(problem, discretizer, L2Error()))
    assert result.passed, result.message
