"""The flux a shallow ice boundary can prescribe.

- On 10 Chebyshev nodes of [-1, 1] with A = rho = 1, friction 2, bed
  0.1 x and H = 1 + x/2 (so grad s = 0.6, which the derivative matrix
  differentiates exactly), the flux is -(gamma H^5 (0.36 + eps) +
  (g/2) H^2) 0.6, to rtol 1e-12.
- Each flux Jacobian component passes DerivativeChecker at a random
  positive state, in 1D and 2D, on a curved bed.
- A time-dependent bed makes the flux time-dependent; constant
  coefficients do not.
"""

from typing import Generic, Union

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh1D, TransformedMesh2D
from pyapprox.pde.collocation.physics.coefficients import NodalField, NodalSupplier
from pyapprox.pde.collocation.physics.shallow_ice import (
    ShallowIceFlux,
    ShallowIcePhysics,
)
from pyapprox.pde.collocation.protocols.boundary import FluxProviderProtocol
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.util.backends.protocols import Array, Backend

_GRAVITY = 9.81


def _rising_bed(coords: Array, time: float) -> Array:
    """b = t + 0.1 x, declared time-dependent by its caller."""
    return time + 0.1 * coords[0]


def _physics(
    bkd: Backend[Array], ndim: int, bed: str = "curved"
) -> ShallowIcePhysics[Array]:
    """A = rho = 1, friction 2, on 10 nodes per direction; the bed is
    ``sloped`` (0.1 x), ``curved`` (0.1 x + 0.05 x y, nodal) or
    ``transient`` (t + 0.1 x)."""
    if ndim == 1:
        basis: Union[ChebyshevBasis1D[Array], ChebyshevBasis2D[Array]] = (
            ChebyshevBasis1D(TransformedMesh1D(10, bkd), bkd)
        )
    else:
        basis = ChebyshevBasis2D(TransformedMesh2D(10, 10, bkd), bkd)
    points = basis.mesh().points()
    x = points[0]
    coefficient: Union[NodalField[Array], NodalSupplier[Array]]
    if bed == "sloped":
        coefficient = NodalField(0.1 * x)
    elif bed == "curved":
        y = points[1] if ndim == 2 else 0.0 * x
        coefficient = NodalField(0.1 * x + 0.05 * x * y)
    else:
        coefficient = NodalSupplier(TimeDependent(_rising_bed), points)
    return ShallowIcePhysics(basis, bkd, bed=coefficient, friction=2.0, A=1.0, rho=1.0)


class _FluxComponent(Generic[Array]):
    """One flux component as a function of the state, for
    DerivativeChecker."""

    def __init__(
        self,
        flux: FluxProviderProtocol[Array],
        dim: int,
        npts: int,
        bkd: Backend[Array],
    ) -> None:
        self._flux, self._dim, self._npts, self._bkd = flux, dim, npts, bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._npts

    def nqoi(self) -> int:
        return self._npts

    def __call__(self, samples: Array) -> Array:
        return self._flux.compute_flux(samples[:, 0], 0.0)[self._dim][:, None]

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)

    def jacobian(self, sample: Array) -> Array:
        return self._flux.compute_flux_jacobian(sample[:, 0], 0.0)[self._dim]


def test_values(bkd: Backend[Array]) -> None:
    physics = _physics(bkd, 1, "sloped")
    x = TransformedMesh1D(10, bkd).points()[0]
    thickness = 1.0 + x / 2.0
    (flux,) = ShallowIceFlux(physics).compute_flux(thickness, 0.0)
    gamma = 2.0 * _GRAVITY**3 / 5.0
    slope = 0.6
    diffusion = (
        gamma * thickness**5 * (slope**2 + 1e-12) + (_GRAVITY / 2.0) * thickness**2
    )
    bkd.assert_allclose(flux, -diffusion * slope, rtol=1e-12)


@pytest.mark.parametrize("ndim", [1, 2])
def test_jacobian(bkd: Backend[Array], ndim: int) -> None:
    physics = _physics(bkd, ndim)
    flux = ShallowIceFlux(physics)
    assert isinstance(flux, FluxProviderProtocol)
    npts = physics.npts()
    state = bkd.asarray(
        2.0 + 0.5 * np.random.default_rng(5).uniform(-1.0, 1.0, (npts, 1))
    )
    for dim in range(ndim):
        checker = DerivativeChecker(_FluxComponent(flux, dim, npts, bkd))
        errors = checker.check_derivatives(state, central=True)
        assert checker.error_ratio(errors[0]) <= 1e-6


def test_time_dependence(numpy_bkd: Backend[Array]) -> None:
    assert not ShallowIceFlux(_physics(numpy_bkd, 1)).flux_is_time_dependent()
    assert ShallowIceFlux(_physics(numpy_bkd, 1, "transient")).flux_is_time_dependent()
