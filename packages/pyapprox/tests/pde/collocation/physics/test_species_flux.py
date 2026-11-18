"""The diffusive flux of one species of the two-species physics, and its
normal component on a stacked state.

- On 6 x 6 Chebyshev nodes of [-1, 1]^2 with D_0 = 1 + x/2, D_1 = 3,
  u = x**2 y and v = x y**2 (differentiated exactly), species 0's flux
  is -(1 + x/2)(2 x y, x**2) and species 1's -3 (y**2, 2 x y), each
  reading only its own block of the stacked state, to 1e-12.
- Each species' flux Jacobian, (npts, 2 npts) and zero outside the
  species' columns, passes DerivativeChecker at a random state.
- FluxNormalOperator on that flux has one Jacobian column per state
  entry (2 npts), equal to the normal-weighted rows of the flux
  Jacobian.
- A non-diffusing species has zero flux and Jacobian; a species other
  than 0 or 1 is refused.
"""

from typing import Generic, Tuple

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.normal_operators import FluxNormalOperator
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.physics.reaction_diffusion import (
    LinearReaction,
    SpeciesDiffusiveFlux,
    TwoSpeciesReactionDiffusionPhysics,
)
from pyapprox.pde.collocation.protocols.boundary import FluxProviderProtocol
from pyapprox.util.backends.protocols import Array, Backend


def _physics(
    bkd: Backend[Array], diffusion1: float = 3.0
) -> Tuple[Array, Array, TwoSpeciesReactionDiffusionPhysics[Array]]:
    """The node coordinates x, y and the physics."""
    mesh = TransformedMesh2D(6, 6, bkd)
    x, y = mesh.points()[0], mesh.points()[1]
    physics = TwoSpeciesReactionDiffusionPhysics(
        ChebyshevBasis2D(mesh, bkd),
        bkd,
        diffusion0=NodalField(1.0 + x / 2.0),
        diffusion1=diffusion1,
        reaction=LinearReaction(0.0, 0.0, 0.0, 0.0, bkd),
    )
    return x, y, physics


class _FluxFunction(Generic[Array]):
    """One flux component as a function of the state, for
    DerivativeChecker."""

    def __init__(
        self, flux: FluxProviderProtocol[Array], dim: int, bkd: Backend[Array]
    ) -> None:
        self._flux, self._dim, self._bkd = flux, dim, bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 72

    def nqoi(self) -> int:
        return 36

    def __call__(self, samples: Array) -> Array:
        return self._flux.compute_flux(samples[:, 0], 0.0)[self._dim][:, None]

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)

    def jacobian(self, sample: Array) -> Array:
        return self._flux.compute_flux_jacobian(sample[:, 0], 0.0)[self._dim]


def test_values(bkd: Backend[Array]) -> None:
    x, y, physics = _physics(bkd)
    state = bkd.hstack([x**2 * y, x * y**2])
    q0 = SpeciesDiffusiveFlux(physics, 0).compute_flux(state, 0.0)
    d0 = 1.0 + x / 2.0
    bkd.assert_allclose(q0[0], -d0 * 2.0 * x * y, atol=1e-12)
    bkd.assert_allclose(q0[1], -d0 * x**2, atol=1e-12)
    q1 = SpeciesDiffusiveFlux(physics, 1).compute_flux(state, 0.0)
    bkd.assert_allclose(q1[0], -3.0 * y**2, atol=1e-12)
    bkd.assert_allclose(q1[1], -3.0 * 2.0 * x * y, atol=1e-12)


@pytest.mark.parametrize("species", [0, 1])
@pytest.mark.parametrize("dim", [0, 1])
def test_jacobian(bkd: Backend[Array], species: int, dim: int) -> None:
    _, _, physics = _physics(bkd)
    flux = SpeciesDiffusiveFlux(physics, species)
    assert isinstance(flux, FluxProviderProtocol)
    state = bkd.asarray(np.random.default_rng(4).normal(0.0, 1.0, (72, 1)))
    jacobian = flux.compute_flux_jacobian(state[:, 0], 0.0)[dim]
    other = slice(36, 72) if species == 0 else slice(0, 36)
    bkd.assert_allclose(jacobian[:, other], bkd.zeros((36, 36)), atol=0.0)
    checker = DerivativeChecker(_FluxFunction(flux, dim, bkd))
    errors = checker.check_derivatives(state)
    assert checker.error_ratio(errors[0]) <= 1e-6


def test_normal_operator_on_stacked_state(bkd: Backend[Array]) -> None:
    _, _, physics = _physics(bkd)
    flux = SpeciesDiffusiveFlux(physics, 1)
    mesh = TransformedMesh2D(6, 6, bkd)
    indices = mesh.boundary_indices(1)
    normals = mesh.boundary_normals(1)
    operator = FluxNormalOperator(bkd, indices, normals, flux)
    state = bkd.asarray(np.random.default_rng(5).normal(0.0, 1.0, (72,)))
    jacobian = operator.jacobian(state, 0.0)
    assert jacobian.shape == (indices.shape[0], 72)
    q = flux.compute_flux_jacobian(state, 0.0)
    expected = normals[:, 0:1] * q[0][indices] + normals[:, 1:2] * q[1][indices]
    bkd.assert_allclose(jacobian, expected, atol=1e-12)


def test_non_diffusing_species(numpy_bkd: Backend[Array]) -> None:
    _, _, physics = _physics(numpy_bkd, diffusion1=0.0)
    flux = SpeciesDiffusiveFlux(physics, 1)
    state = numpy_bkd.ones((72,))
    for q in flux.compute_flux(state, 0.0):
        numpy_bkd.assert_allclose(q, numpy_bkd.zeros((36,)), atol=0.0)
    for jac in flux.compute_flux_jacobian(state, 0.0):
        numpy_bkd.assert_allclose(jac, numpy_bkd.zeros((36, 72)), atol=0.0)
    assert not flux.flux_is_time_dependent()


def test_refuses_a_third_species(numpy_bkd: Backend[Array]) -> None:
    _, _, physics = _physics(numpy_bkd)
    with pytest.raises(ValueError, match="species must be 0 or 1"):
        SpeciesDiffusiveFlux(physics, 2)
