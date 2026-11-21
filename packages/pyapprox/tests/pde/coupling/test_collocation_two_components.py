r"""Collocation subdomains coupled through a skeleton of two components
per node, against the exact and the single-domain collocation solutions;
NumPy and Torch.

The unit square is cut into 3 x 1 strips or 2 x 2 boxes
(:mod:`tests._helpers.coupling_collocation`), Dirichlet data on the whole
boundary, forcing manufactured. Two physics, each checked over both
components:

- Two species, :math:`D_c \Delta u_c + R_c(u) + f_c = 0`, D = (2, 3),
  linear reaction (-1, 0.5, 0.3, -2); ports return each species'
  diffusive flux.
- Linear elasticity, :math:`\nabla\cdot\sigma + f = 0`, lambda = 1,
  mu = 2; ports return the traction.

Cases:

- Cubic fields lie in every subdomain space (5 nodes per direction):
  matching skeletons reproduce them at the skeleton and in every
  subdomain to 1e-10, equal to the single-domain solve (6 nodes)
  interpolated there; non-matching counts (5, 6, 7) with a 4-node
  skeleton reproduce them too.
- Smooth fields on the 2 x 2 boxes (cross points carrying both
  components): the largest nodal error decays spectrally.
- At the solution the summed tested fluxes at the free skeleton DOFs are
  below the solver tolerance.
"""

from typing import Callable

import pytest

from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.coupling_collocation import (
    ATOL,
    CUBIC,
    SMOOTH,
    CoupledCollocation,
    Elasticity,
    Model,
    TwoSpecies,
    interpolate,
    max_state_error,
)

# Exact fields (u, v). Species: u the scalar cubic or smooth field, v a
# second cubic or a harmonic field. Displacement cubic: div u = 4x^2 + 2x
# and shear u_y + v_x = 1 + 2x, both nonzero.
_FIELDS = {
    "species": {
        "cubic": (CUBIC, "1 + 2*x*y - y**2 + x**3/2"),
        "smooth": (SMOOTH, "exp(y)*cos(x)"),
    },
    "elasticity": {
        "cubic": ("x**3 - x*y**2 + y", "x**2*y + y**3/3 + x**2"),
        "smooth": ("sin(2*x)*cos(y)", "exp(x)*sin(y)"),
    },
}


def _species(exact: str) -> TwoSpecies[Array]:
    fields = _FIELDS["species"][exact]
    return TwoSpecies((2.0, 3.0), (-1.0, 0.5, 0.3, -2.0), fields)


def _elasticity(exact: str) -> Elasticity[Array]:
    return Elasticity(1.0, 2.0, _FIELDS["elasticity"][exact])


_MODELS = pytest.mark.parametrize(
    "make", [_species, _elasticity], ids=["species", "elasticity"]
)


@_MODELS
@pytest.mark.parametrize("ncells", [(3, 1), (2, 2)], ids=["strips", "boxes"])
def test_cubic_fields_equal_the_single_domain_solve(
    bkd: Backend[Array], make: Callable[[str], Model[Array]], ncells: tuple[int, int]
) -> None:
    model = make("cubic")
    nsub = ncells[0] * ncells[1]
    problem = CoupledCollocation(bkd, ncells, [5] * nsub, 5, model)
    lam = problem.solve_skeleton()
    bkd.assert_allclose(lam[:, 0], problem.skeleton_exact(), rtol=1e-10, atol=1e-10)
    assert max_state_error(problem, lam) < 1e-10
    _, values = problem.single_domain(6)
    single = bkd.reshape(values, (2, -1))
    for piece in problem.pieces():
        points = piece.box.points()
        expected = bkd.concatenate(
            [interpolate(single[c], 6, points, bkd) for c in range(2)], axis=0
        )
        bkd.assert_allclose(piece.state(lam), expected, rtol=1e-10, atol=1e-10)


@_MODELS
def test_non_matching_counts_reproduce_cubic_fields(
    bkd: Backend[Array], make: Callable[[str], Model[Array]]
) -> None:
    model = make("cubic")
    problem = CoupledCollocation(bkd, (3, 1), [5, 6, 7], 4, model, interpolate=True)
    assert max_state_error(problem, problem.solve_skeleton()) < 1e-10


@_MODELS
def test_smooth_fields_converge_spectrally(
    bkd: Backend[Array], make: Callable[[str], Model[Array]]
) -> None:
    # Measured: species 7.9e-3, 3.9e-5, 4.2e-7, 1.1e-9, 1.3e-12;
    # elasticity 3.4e-3, 1.0e-5, 1.5e-8, 1.3e-11, 4.0e-15.
    errors = []
    for n in (4, 6, 8, 10, 12):
        model = make("smooth")
        problem = CoupledCollocation(bkd, (2, 2), [n] * 4, n, model)
        errors.append(max_state_error(problem, problem.solve_skeleton()))
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert fine < coarse / 30.0
    assert errors[-1] < 1e-10


@_MODELS
def test_the_interface_fluxes_balance(
    bkd: Backend[Array], make: Callable[[str], Model[Array]]
) -> None:
    model = make("smooth")
    problem = CoupledCollocation(bkd, (2, 2), [8] * 4, 8, model)
    lam = problem.solve_skeleton()
    balance = problem.residual()(lam)[:, 0]
    free = sorted(set(range(int(lam.shape[0]))) - set(problem.pinned()))
    assert bkd.to_float(bkd.max(bkd.abs(balance[free]))) < ATOL
