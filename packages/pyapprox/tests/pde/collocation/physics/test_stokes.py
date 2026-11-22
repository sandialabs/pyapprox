"""The staggered Stokes interior operator.

Velocity on n Gauss-Lobatto nodes per direction of the unit square or
cube, pressure on n - 2 interior Chebyshev-Gauss nodes:

- the Jacobian passes DerivativeChecker with variable viscosity, 2D and 3D;
- the residual vanishes at the nodes on an exact polynomial pair
  (velocity of degree at most n - 1, pressure at most n - 3);
- with the velocity prescribed on the whole boundary, the only free mode
  is the constant pressure (one zero singular value); replacing one
  continuity row with the pressure's mean makes the system nonsingular,
  and its solution is the pair;
- the refusals.
"""

from typing import Generic, List, Optional, Sequence, Set, Tuple

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis, ChebyshevGauss1D
from pyapprox.pde.collocation.basis.transfer import grid_transfer_matrix
from pyapprox.pde.collocation.boundary.mean_value_rows import MeanValueRows
from pyapprox.pde.collocation.mesh import TransformedMesh
from pyapprox.pde.collocation.mesh.transforms.affine import AffineTransform
from pyapprox.pde.collocation.physics.coefficients import CoefficientInput, NodalField
from pyapprox.pde.collocation.physics.stokes import StokesPhysics
from pyapprox.polynomials.univariate.globalpoly.jacobi import LegendrePolynomial1D
from pyapprox.polynomials.univariate.globalpoly.quadrature import GaussQuadratureRule
from pyapprox.util.backends.protocols import Array, Backend

_Pair = Tuple[TransformedMesh[Array], TransformedMesh[Array]]


def _meshes(bkd: Backend[Array], ndim: int, n: int) -> _Pair[Array]:
    box = AffineTransform([(0.0, 1.0)] * ndim, bkd)
    velocity = TransformedMesh((n,) * ndim, bkd, box)
    pressure = TransformedMesh(
        (n - 2,) * ndim, bkd, box, bases=[ChebyshevGauss1D[Array]()] * ndim
    )
    return velocity, pressure


def _physics(
    bkd: Backend[Array],
    meshes: _Pair[Array],
    viscosity: CoefficientInput[Array],
    forcing: Optional[Sequence[CoefficientInput[Array]]] = None,
) -> StokesPhysics[Array]:
    velocity, pressure = meshes
    return StokesPhysics(
        ChebyshevBasis(velocity, bkd),
        grid_transfer_matrix(pressure, velocity, bkd),
        grid_transfer_matrix(velocity, pressure, bkd),
        viscosity,
        forcing,
    )


class _Residual(Generic[Array]):
    """The interior residual as a function of the state."""

    def __init__(self, physics: StokesPhysics[Array]) -> None:
        self._physics = physics

    def bkd(self) -> Backend[Array]:
        return self._physics.bkd()

    def nvars(self) -> int:
        return self._physics.nstates()

    def nqoi(self) -> int:
        return self._physics.nstates()

    def __call__(self, samples: Array) -> Array:
        return self._physics.interior_residual(samples[:, 0], 0.0)[:, None]

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)

    def jacobian(self, sample: Array) -> Array:
        return self._physics.interior_jacobian(sample[:, 0], 0.0)


@pytest.mark.parametrize("ndim, n", [(2, 5), (3, 4)])
def test_jacobian(bkd: Backend[Array], ndim: int, n: int) -> None:
    """Viscosity 1 + x0 x1 / 2 at the velocity nodes, a random state."""
    meshes = _meshes(bkd, ndim, n)
    points = meshes[0].points()
    physics = _physics(
        bkd, meshes, viscosity=NodalField(1.0 + points[0] * points[1] / 2.0)
    )
    assert physics.nstates() == ndim * n**ndim + (n - 2) ** ndim
    checker: DerivativeChecker[Array] = DerivativeChecker(_Residual(physics))
    state = bkd.asarray(
        np.random.default_rng(5).normal(0.0, 1.0, (physics.nstates(), 1))
    )
    errors = checker.check_derivatives(state, central=True)
    assert checker.error_ratio(errors[0]) <= 1e-6


def _exact_pair(bkd: Backend[Array], meshes: _Pair[Array]) -> Tuple[Array, Array]:
    """u = (y^2, x^2), p = x y and its forcing f = (y - 2, x - 2), g = 0;
    the state and the forcings."""
    velocity, pressure = meshes
    x, y = velocity.points()[0], velocity.points()[1]
    px, py = pressure.points()[0], pressure.points()[1]
    state = bkd.concatenate([y**2, x**2, px * py])
    return state, bkd.stack([y - 2.0, x - 2.0])


def test_residual_vanishes_on_a_polynomial_pair(bkd: Backend[Array]) -> None:
    meshes = _meshes(bkd, 2, 5)
    state, forcing = _exact_pair(bkd, meshes)
    physics = _physics(
        bkd,
        meshes,
        viscosity=1.0,
        forcing=[NodalField(forcing[0]), NodalField(forcing[1])],
    )
    bkd.assert_allclose(
        physics.interior_residual(state, 0.0), bkd.zeros((59,)), atol=1e-11
    )


def _boundary_nodes(bkd: Backend[Array], mesh: TransformedMesh[Array]) -> List[int]:
    nodes: Set[int] = set()
    for boundary in range(2 * mesh.ndim()):
        indices = mesh.boundary_indices(boundary)
        nodes.update(bkd.to_int(indices[k]) for k in range(indices.shape[0]))
    return sorted(nodes)


def _pressure_weights(bkd: Backend[Array], n: int) -> Array:
    """The integrals over [0, 1]^2 of the interpolants on n x n Gauss
    nodes, from a 2n-point Gauss-Legendre rule."""
    legendre = LegendrePolynomial1D(bkd)
    legendre.set_nterms(2 * n)
    reference, probability = GaussQuadratureRule(legendre)(2 * n)
    interpolation = ChebyshevGauss1D[Array]().interpolation_matrix(
        n, reference[0, :], bkd
    )
    # Probability weights sum to one, the interval [0, 1] has length one.
    weights_1d = probability[:, 0] @ interpolation
    return bkd.kron(weights_1d, weights_1d)


def _closed(
    bkd: Backend[Array], physics: StokesPhysics[Array], velocity: TransformedMesh[Array]
) -> Tuple[Array, List[int]]:
    """The Jacobian with identity rows for the velocity on the boundary."""
    n = velocity.npts()
    rows = [k * n + node for k in range(2) for node in _boundary_nodes(bkd, velocity)]
    jacobian = physics.interior_jacobian(bkd.zeros((physics.nstates(),)), 0.0)
    index = bkd.asarray(rows, dtype=bkd.int64_dtype())
    identity = bkd.eye(physics.nstates())[index]
    return bkd.index_update(jacobian, index, identity), rows


def test_only_the_constant_pressure_is_free(numpy_bkd: Backend[Array]) -> None:
    bkd = numpy_bkd
    meshes = _meshes(bkd, 2, 7)
    physics = _physics(bkd, meshes, viscosity=1.0)
    jacobian, _ = _closed(bkd, physics, meshes[0])
    singular = bkd.svd(jacobian)[1]
    assert bkd.to_float(singular[-1]) < 1e-12
    assert bkd.to_float(singular[-2]) > 1e-6
    constant = bkd.concatenate(
        [bkd.zeros((2 * meshes[0].npts(),)), bkd.ones((meshes[1].npts(),))]
    )
    bkd.assert_allclose(
        jacobian @ constant, bkd.zeros((physics.nstates(),)), atol=1e-11
    )


def test_mean_row_determines_the_pressure(bkd: Backend[Array]) -> None:
    """One continuity row replaced by the pressure's integral, 1/4 for
    p = x y: the solve from zero recovers the pair."""
    meshes = _meshes(bkd, 2, 5)
    exact, forcing = _exact_pair(bkd, meshes)
    physics = _physics(
        bkd,
        meshes,
        viscosity=1.0,
        forcing=[NodalField(forcing[0]), NodalField(forcing[1])],
    )
    jacobian, rows = _closed(bkd, physics, meshes[0])
    mean = MeanValueRows(bkd, physics.layout(), 2, _pressure_weights(bkd, 3), 0.25)
    zero = bkd.zeros((physics.nstates(),))
    (row,) = mean.rows().indices()
    index = bkd.asarray(rows, dtype=bkd.int64_dtype())
    residual = physics.interior_residual(zero, 0.0)
    residual = bkd.index_update(residual, index, -exact[index])
    residual = bkd.index_update(
        residual,
        bkd.asarray([row], dtype=bkd.int64_dtype()),
        mean.residual(zero, 0.0)[row : row + 1],
    )
    jacobian = bkd.index_update(
        jacobian,
        bkd.asarray([row], dtype=bkd.int64_dtype()),
        mean.jacobian(zero, 0.0)[row : row + 1],
    )
    solution = -bkd.solve(jacobian, residual[:, None])[:, 0]
    bkd.assert_allclose(solution, exact, atol=1e-11)


def test_refusals(bkd: Backend[Array]) -> None:
    velocity, pressure = _meshes(bkd, 2, 5)
    basis = ChebyshevBasis(velocity, bkd)
    to_velocity = grid_transfer_matrix(pressure, velocity, bkd)
    to_pressure = grid_transfer_matrix(velocity, pressure, bkd)
    with pytest.raises(ValueError, match="to_pressure must have shape"):
        StokesPhysics(basis, to_velocity, to_velocity.T[:, :-1], 1.0)
    with pytest.raises(ValueError, match="to_velocity must have shape"):
        StokesPhysics(basis, to_pressure, to_pressure, 1.0)
    with pytest.raises(ValueError, match="one entry per velocity component"):
        StokesPhysics(basis, to_velocity, to_pressure, 1.0, forcing=[0.0])
    with pytest.raises(TypeError, match="StokesBasisProtocol"):
        StokesPhysics("basis", to_velocity, to_pressure, 1.0)  # type: ignore[arg-type]
    line = TransformedMesh((4,), bkd)
    with pytest.raises(ValueError, match="2D or 3D basis, got 1D"):
        StokesPhysics(
            ChebyshevBasis(line, bkd), bkd.ones((4, 2)), bkd.ones((2, 4)), 1.0
        )
    negative = StokesPhysics(basis, to_velocity, to_pressure, -1.0)
    with pytest.raises(ValueError, match="viscosity must be positive"):
        negative.interior_residual(bkd.zeros((negative.nstates(),)), 0.0)
