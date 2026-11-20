"""Checks shared by the collocation manufactured-solution catalogs.

Each physics' catalog (``tests/pde/models/collocation/
test_verification_<physics>.py``) supplies its own system factory and
cases; these run a case through the verification harness on Chebyshev
boxes and assert its verdict, so a failure reports the observed decay or
rate and the errors.
"""

from typing import Callable, Dict, Optional, Sequence

from pyapprox.pde.models.collocation.verification import (
    ChebyshevBox,
    CollocationDiscretizer,
    DecaysExponentially,
    InitialGuessProtocol,
    LevelFamilyProtocol,
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
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

# A level family built from node counts per direction and a backend, as
# ChebyshevBox, ChebyshevAnnulus and ChebyshevEllipticalAnnulus are.
_Family = Callable[[Sequence[int], Backend[Array]], LevelFamilyProtocol[Array]]

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
    initial_guess: Optional[InitialGuessProtocol] = None,
    nonlinear: Optional[NonlinearSolverProtocol[Array, Array]] = None,
) -> None:
    """``n`` nodes per direction reproduce the solution: L2 error below
    1e-10. Solves start from ``initial_guess`` (default zero), with
    ``nonlinear`` (default the discretizer's Newton)."""
    discretizer = CollocationDiscretizer(
        ChebyshevBox([n], bkd),
        factory,
        nonlinear=nonlinear,
        initial_guess=initial_guess,
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
    initial_guess: Optional[InitialGuessProtocol] = None,
    nonlinear: Optional[NonlinearSolverProtocol[Array, Array]] = None,
    family: _Family[Array] = ChebyshevBox,
) -> None:
    """The L2 error falls exponentially over ``sizes`` nodes per
    direction, at least ``rate`` per node, on the levels ``family``
    builds from the sizes (default Chebyshev boxes). Solves start from
    ``initial_guess`` (default zero), with ``nonlinear`` (default the
    discretizer's Newton)."""
    discretizer = CollocationDiscretizer(
        family(sizes, bkd),
        factory,
        nonlinear=nonlinear,
        initial_guess=initial_guess,
    )
    study = run_node_study(problem, discretizer, L2Error())
    result = DecaysExponentially(rate, last).verify(study)
    assert result.passed, result.message


def stalls(
    problem: ManufacturedProblem,
    factory: SystemFactoryProtocol,
    sizes: Sequence[int],
    bkd: Backend[Array],
    initial_guess: Optional[InitialGuessProtocol] = None,
    nonlinear: Optional[NonlinearSolverProtocol[Array, Array]] = None,
    family: _Family[Array] = ChebyshevBox,
) -> None:
    """Wrong-model control: the fitted algebraic L2 rate stays below 0.5,
    so the error does not fall as a correct model's would, on the levels
    ``family`` builds from the sizes (default Chebyshev boxes). Solves
    start from ``initial_guess`` (default zero), with ``nonlinear``
    (default the discretizer's Newton)."""
    discretizer = CollocationDiscretizer(
        family(sizes, bkd),
        factory,
        nonlinear=nonlinear,
        initial_guess=initial_guess,
    )
    result = Stalls(below=0.5).verify(run_study(problem, discretizer, L2Error()))
    assert result.passed, result.message
