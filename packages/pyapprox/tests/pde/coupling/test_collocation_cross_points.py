r"""Collocation subdomains meeting at cross points; NumPy and Torch.

The unit square cut 2 x 2 and the unit cube 2 x 2 x 2, so interior
skeleton vertices (and in 3D edges) are shared by four (eight)
subdomains. A subdomain's node there lies on two (three) of its port
sides; it is one port node, tested on each side with that side's normal,
and returns the sum. Advection-diffusion-reaction.

- Matching nodes, lumped testing: a cubic, in every space, is reproduced
  on 2 x 2 (n = 5) and 2 x 2 x 2 (n = 5) to 1e-10 (measured 4e-14,
  2e-13), the skeleton values included.
- Matching nodes, e^x sin 3x cos 2y on 2 x 2: 3.6e-5, 4.2e-7, 1.1e-9,
  1.3e-12 for n = 6, 8, 10, 12 under either rule, asserted as each below
  a fiftieth of the last and the finest below 1e-10.
- Different nodes [n, n+2, n+1, n+3], skeleton n, Gauss testing: the
  cubic (n = 4) to 1e-10 (1.3e-13); the smooth solution 6.0e-5, 2.6e-7,
  3.9e-10, 2.2e-13 for n = 6..12, asserted the same way; lumped testing
  at n = 12 is more than a hundred times worse (7.8e-5).
- The fluxes balance at every free skeleton DOF, the vertex included.
"""

import pytest

from pyapprox.pde.collocation.quadrature.boundary_quadrature import (
    CollocationNodeRule,
    GaussLegendreSideRule,
    SideRuleProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.coupling_collocation import (
    ATOL,
    CUBIC,
    CUBIC_3D,
    SMOOTH,
    Adr,
    Coefficients,
    CoupledCollocation,
    max_state_error,
)

_ADR_2D = Coefficients(2.0, (0.7, -0.4), -1.5)
_ADR_3D = Coefficients(2.0, (0.7, -0.4, 0.3), -1.5)


def _solve(
    bkd: Backend[Array],
    ncells: tuple[int, ...],
    ns: list[int],
    nskel: int,
    exact: str,
    interpolate: bool,
    rule: SideRuleProtocol,
) -> tuple[CoupledCollocation[Array], Array]:
    coefs = _ADR_2D if len(ncells) == 2 else _ADR_3D
    problem = CoupledCollocation(
        bkd, ncells, ns, nskel, Adr(coefs, exact), interpolate, rule
    )
    return problem, problem.solve_skeleton()


def _assert_spectral(errors: list[float]) -> None:
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert fine < coarse / 50.0
    assert errors[-1] < 1e-10


@pytest.mark.parametrize("ncells", [(2, 2), (2, 2, 2)], ids=["2d", "3d"])
def test_matching_nodes_reproduce_a_cubic(
    bkd: Backend[Array], ncells: tuple[int, ...]
) -> None:
    nsub = 2 ** len(ncells)
    cubic = CUBIC if len(ncells) == 2 else CUBIC_3D
    problem, lam = _solve(
        bkd, ncells, [5] * nsub, 5, cubic, False, CollocationNodeRule()
    )
    assert max_state_error(problem, lam) < 1e-10
    expected = problem.skeleton_exact()
    bkd.assert_allclose(lam[:, 0], expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "rule", [CollocationNodeRule(), GaussLegendreSideRule()], ids=["lumped", "gauss"]
)
def test_matching_nodes_converge_spectrally(
    bkd: Backend[Array], rule: SideRuleProtocol
) -> None:
    errors = [
        max_state_error(*_solve(bkd, (2, 2), [n] * 4, n, SMOOTH, False, rule))
        for n in (6, 8, 10, 12)
    ]
    _assert_spectral(errors)


def test_different_nodes_reproduce_a_cubic(bkd: Backend[Array]) -> None:
    problem, lam = _solve(
        bkd, (2, 2), [5, 6, 7, 6], 4, CUBIC, True, GaussLegendreSideRule()
    )
    assert max_state_error(problem, lam) < 1e-10


def test_different_nodes_converge_spectrally(bkd: Backend[Array]) -> None:
    errors = [
        max_state_error(
            *_solve(
                bkd,
                (2, 2),
                [n, n + 2, n + 1, n + 3],
                n,
                SMOOTH,
                True,
                GaussLegendreSideRule(),
            )
        )
        for n in (6, 8, 10, 12)
    ]
    _assert_spectral(errors)
    lumped = _solve(
        bkd, (2, 2), [12, 14, 13, 15], 12, SMOOTH, True, CollocationNodeRule()
    )
    assert max_state_error(*lumped) > 100.0 * errors[-1]


@pytest.mark.parametrize("interpolate", [False, True], ids=["matching", "interp"])
def test_the_fluxes_balance_at_the_vertex(
    bkd: Backend[Array], interpolate: bool
) -> None:
    ns = [8, 10, 9, 11] if interpolate else [8] * 4
    problem, lam = _solve(
        bkd, (2, 2), ns, 8, SMOOTH, interpolate, GaussLegendreSideRule()
    )
    balance = problem.residual()(lam)[:, 0]
    free = sorted(set(range(int(lam.shape[0]))) - set(problem.pinned()))
    assert bkd.to_float(bkd.max(bkd.abs(balance[free]))) < ATOL
