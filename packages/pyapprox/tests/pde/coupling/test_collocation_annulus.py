r"""Collocation subdomains coupled across a curved interface; NumPy and
Torch.

The quarter annulus :math:`1 \le r \le 2`, :math:`0 \le \theta \le
\pi/2`, cut at r = 3/2 into two polar collocation subdomains, so the
interface is an arc. The partition and skeleton live in the (r, theta)
parameter box (:class:`~tests._helpers.coupling_collocation.
QuarterAnnulus`); normals, the arc-length weights and the physics are
physical. Advection-diffusion-reaction, u = e^x sin 3x cos 2y.

- Matching nodes, lumped testing: the coupled error equals the
  single-domain polar solve's, 5.0e-2, 1.2e-3, 1.5e-5 for n = 8, 12, 16
  (within 5%), and the interface fluxes balance.
- Different nodes [n, n + 3], skeleton n, Gauss testing: 2.8e-2, 3.9e-4,
  2.5e-6, asserted as each below a fiftieth of the last; lumped testing
  at n = 16 misses by more than ten times as much (8.6e-5).

On the arc the surface element is the constant r, so the Gauss rule
needs no extra points there.
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
    Coefficients,
    CoupledCollocation,
    QuarterAnnulus,
    max_state_error,
    smooth,
)

_ADR = Coefficients(2.0, (0.7, -0.4), -1.5)


def _problem(
    bkd: Backend[Array],
    ns: list[int],
    nskel: int,
    interpolate: bool,
    rule: SideRuleProtocol,
) -> CoupledCollocation[Array]:
    return CoupledCollocation(
        bkd, (2, 1), ns, nskel, _ADR, smooth, interpolate, rule, QuarterAnnulus()
    )


@pytest.mark.parametrize("n", [8, 12, 16])
def test_matching_nodes_give_the_single_domain_error(
    bkd: Backend[Array], n: int
) -> None:
    problem = _problem(bkd, [n, n], n, False, CollocationNodeRule())
    lam = problem.solve_skeleton()
    box, state = problem.single_domain(n)
    single = bkd.to_float(bkd.max(bkd.abs(state - smooth(box.points(), bkd)[0])))
    coupled = max_state_error(problem, lam)
    assert abs(coupled - single) < 0.05 * single
    balance = problem.residual()(lam)[:, 0]
    free = sorted(set(range(int(lam.shape[0]))) - set(problem.pinned()))
    assert bkd.to_float(bkd.max(bkd.abs(balance[free]))) < ATOL


def test_different_nodes_converge_spectrally(bkd: Backend[Array]) -> None:
    errors = []
    for n in (8, 12, 16):
        problem = _problem(bkd, [n, n + 3], n, True, GaussLegendreSideRule())
        errors.append(max_state_error(problem, problem.solve_skeleton()))
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert fine < coarse / 50.0
    lumped = _problem(bkd, [16, 19], 16, True, CollocationNodeRule())
    assert max_state_error(lumped, lumped.solve_skeleton()) > 10.0 * errors[-1]
