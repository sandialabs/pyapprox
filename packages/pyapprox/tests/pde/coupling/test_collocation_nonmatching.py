r"""Collocation subdomains of different node counts coupled by
interpolation through a coarser skeleton; NumPy and Torch.

A 3 x 1 strip of the unit square, subdomain ``s`` with ``ns[s]`` nodes
per direction, the skeleton's face basis on ``nskel`` Chebyshev-Gauss-
Lobatto nodes. Each subdomain's interface nodes off the outer boundary
are ports reading the skeleton's interpolant; the rest of its boundary is
Dirichlet. Each side tests its flux with the Gauss side rule: the flux's
nodal interpolant against the side's Lagrange polynomials, exactly.

- A cubic lies in every subdomain space and in the skeleton's (counts
  [5, 6, 7], skeleton 4): reproduced to 1e-10 for diffusion and for
  advection-diffusion-reaction.
- Control: lumped testing on the same setup misses the advective cubic
  by more than 1e-6 (6.9e-5): each side's Clenshaw-Curtis rule is exact
  only to about its node count, and unequal rules do not cancel.
- e^x sin 3x cos 2y with advection-diffusion-reaction, counts
  [n, n + 2, n + 1], skeleton n: errors 5.5e-5, 2.4e-7, 6.3e-10,
  1.2e-12 for n = 6, 8, 10, 12, asserted as each below a hundredth of the
  last and the finest below 1e-10.
- The interface fluxes balance at the coupled solution, recomputed there
  by a fresh residual, to the subdomain solves' tolerance (1e-10): each
  re-solve stops at its own tolerance, looser than the skeleton's, so the
  skeleton's 1e-11 is no bound on the recomputed balance (2.1e-11 on
  Linux CI).
"""

import pytest

from pyapprox.pde.collocation.quadrature.boundary_quadrature import (
    CollocationNodeRule,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.coupling_collocation import (
    CUBIC,
    SMOOTH,
    SUBDOMAIN_ATOL,
    Adr,
    Coefficients,
    CoupledCollocation,
    max_state_error,
)

_DIFFUSION = Coefficients(2.0, None, None)
_ADR = Coefficients(2.0, (0.7, -0.4), -1.5)
_COUNTS = [5, 6, 7]


@pytest.mark.parametrize("coefs", [_DIFFUSION, _ADR], ids=["diffusion", "adr"])
def test_a_cubic_is_reproduced(bkd: Backend[Array], coefs: Coefficients) -> None:
    problem = CoupledCollocation(
        bkd, (3, 1), _COUNTS, 4, Adr(coefs, CUBIC), interpolate=True
    )
    assert max_state_error(problem, problem.solve_skeleton()) < 1e-10


def test_lumped_testing_misses_the_advective_cubic(bkd: Backend[Array]) -> None:
    problem = CoupledCollocation(
        bkd, (3, 1), _COUNTS, 4, Adr(_ADR, CUBIC), True, CollocationNodeRule()
    )
    assert max_state_error(problem, problem.solve_skeleton()) > 1e-6


def test_a_smooth_solution_converges_spectrally(bkd: Backend[Array]) -> None:
    errors = []
    for n in (6, 8, 10, 12):
        problem = CoupledCollocation(
            bkd, (3, 1), [n, n + 2, n + 1], n, Adr(_ADR, SMOOTH), interpolate=True
        )
        errors.append(max_state_error(problem, problem.solve_skeleton()))
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert fine < coarse / 100.0
    assert errors[-1] < 1e-10


def test_the_interface_fluxes_balance(bkd: Backend[Array]) -> None:
    problem = CoupledCollocation(
        bkd, (3, 1), [8, 10, 9], 8, Adr(_ADR, SMOOTH), interpolate=True
    )
    lam = problem.solve_skeleton()
    balance = problem.residual()(lam)[:, 0]
    free = sorted(set(range(int(lam.shape[0]))) - set(problem.pinned()))
    assert bkd.to_float(bkd.max(bkd.abs(balance[free]))) < SUBDOMAIN_ATOL
