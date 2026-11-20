r"""Strips of collocation subdomains coupled through a matching skeleton,
against a single-domain collocation solve; NumPy and Torch.

The unit square is cut into 3 x 1 subdomains and the unit cube into
3 x 1 x 1 (:mod:`tests._helpers.coupling_collocation`). The PDE is
:math:`D \Delta u - \nabla\cdot(v u) + r u + f = 0` with Dirichlet data on
the whole boundary, f manufactured; diffusion alone (D = 2) and
advection-diffusion-reaction (D = 2, constant v, r = -1.5). Interface
nodes on the outer boundary are pinned skeleton DOFs.

- A cubic exact solution lies in every subdomain space (5 nodes per
  direction) and in the single-domain one (6): every subdomain's state
  equals the single-domain solution interpolated to its nodes, and the
  exact values, to 1e-10; so do the skeleton values.
- For :math:`u = e^x \sin 3x \cos 2y` on the 3 x 1 strip the largest
  nodal error decays spectrally in the nodes per direction.
- At the solution the summed tested fluxes at the free skeleton DOFs are
  below the solver tolerance.
"""

import pytest

from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.coupling_collocation import (
    ATOL,
    Adr,
    Coefficients,
    CoupledCollocation,
    cubic,
    interpolate,
    max_state_error,
    smooth,
)

_DIFFUSION_2D = Coefficients(2.0, None, None)
_ADR_2D = Coefficients(2.0, (0.7, -0.4), -1.5)
_DIFFUSION_3D = Coefficients(2.0, None, None)
_ADR_3D = Coefficients(2.0, (0.7, -0.4, 0.3), -1.5)


@pytest.mark.parametrize(
    "ncells, coefs",
    [
        ((3, 1), _DIFFUSION_2D),
        ((3, 1), _ADR_2D),
        ((3, 1, 1), _DIFFUSION_3D),
        ((3, 1, 1), _ADR_3D),
    ],
    ids=["2d-diffusion", "2d-adr", "3d-diffusion", "3d-adr"],
)
def test_a_cubic_equals_the_single_domain_solve(
    bkd: Backend[Array], ncells: tuple[int, ...], coefs: Coefficients
) -> None:
    problem = CoupledCollocation(bkd, ncells, [5] * 3, 5, Adr(coefs, cubic))
    lam = problem.solve_skeleton()
    coords = problem.skeleton_coords()
    bkd.assert_allclose(lam[:, 0], cubic(coords, bkd)[0], rtol=1e-10, atol=1e-10)
    whole, values = problem.single_domain(6)
    for piece in problem.pieces():
        state, points = piece.state(lam), piece.box.points()
        bkd.assert_allclose(
            state, interpolate(values, 6, points, bkd), rtol=1e-10, atol=1e-10
        )
        bkd.assert_allclose(state, cubic(points, bkd)[0], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("coefs", [_DIFFUSION_2D, _ADR_2D], ids=["diffusion", "adr"])
def test_a_smooth_solution_converges_spectrally(
    bkd: Backend[Array], coefs: Coefficients
) -> None:
    # Measured: 4.4e-3, 3.7e-5, 1.3e-7, 2.8e-10, 4.2e-13 (diffusion).
    errors = []
    for n in (4, 6, 8, 10, 12):
        problem = CoupledCollocation(bkd, (3, 1), [n] * 3, n, Adr(coefs, smooth))
        errors.append(max_state_error(problem, problem.solve_skeleton()))
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert fine < coarse / 100.0
    assert errors[-1] < 1e-11


@pytest.mark.parametrize(
    "ncells, coefs, n",
    [((3, 1), _ADR_2D, 8), ((3, 1, 1), _ADR_3D, 5)],
    ids=["2d-smooth", "3d-cubic"],
)
def test_the_interface_fluxes_balance(
    bkd: Backend[Array], ncells: tuple[int, ...], coefs: Coefficients, n: int
) -> None:
    exact = smooth if len(ncells) == 2 else cubic
    problem = CoupledCollocation(bkd, ncells, [n] * 3, n, Adr(coefs, exact))
    lam = problem.solve_skeleton()
    balance = problem.residual()(lam)[:, 0]
    free = sorted(set(range(int(lam.shape[0]))) - set(problem.pinned()))
    assert bkd.to_float(bkd.max(bkd.abs(balance[free]))) < ATOL
