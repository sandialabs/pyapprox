"""The fluxes a shallow shelf velocity boundary can prescribe.

- On 6 x 6 Chebyshev nodes of [-1, 1]^2 with A = 1, depth 2 and the
  linear velocity u = x + y/2, v = x/2 + y (constant strain, Q = 13/4),
  the fluxes are the constant -(tau_xx, tau_xy) and -(tau_xy, tau_yy)
  with tau = 2 mu H a, mu = (Q + eps)^(-1/3) / 2, to rtol 1e-12.
- Each flux direction's Jacobian passes DerivativeChecker at a random
  velocity, for both components, on a variable depth.
- A component other than 0 or 1 is refused; a time-dependent depth makes
  the flux time-dependent and a constant one does not.
"""

from typing import Generic, Union

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.coefficients import NodalField, NodalSupplier
from pyapprox.pde.collocation.physics.shallow_shelf import (
    ShallowShelfStressFlux,
    ShallowShelfVelocityPhysics,
)
from pyapprox.pde.collocation.protocols.boundary import FluxProviderProtocol
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.util.backends.protocols import Array, Backend

_N = 6


def _thickening(coords: Array, time: float) -> Array:
    """H = 1 + t + x/4, declared time-dependent by its caller."""
    return 1.0 + time + 0.25 * coords[0]


def _physics(
    bkd: Backend[Array], depth: str = "constant"
) -> ShallowShelfVelocityPhysics[Array]:
    """A = rho = g = 1, friction 1, flat bed; the depth is ``constant``
    (2), ``variable`` (1.5 + x/4 + y/8, nodal) or ``transient``."""
    mesh = TransformedMesh2D(_N, _N, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    points = mesh.points()
    coefficient: Union[float, NodalField[Array], NodalSupplier[Array]] = 2.0
    if depth == "variable":
        coefficient = NodalField(1.5 + 0.25 * points[0] + 0.125 * points[1])
    elif depth == "transient":
        coefficient = NodalSupplier(TimeDependent(_thickening), points)
    return ShallowShelfVelocityPhysics(
        basis, bkd, depth=coefficient, bed=0.0, friction=1.0, A=1.0, rho=1.0, g=1.0
    )


class _FluxDirection(Generic[Array]):
    """One direction of a flux as a function of the velocity, for
    DerivativeChecker."""

    def __init__(
        self, flux: FluxProviderProtocol[Array], dim: int, bkd: Backend[Array]
    ) -> None:
        self._flux, self._dim, self._bkd = flux, dim, bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2 * _N * _N

    def nqoi(self) -> int:
        return _N * _N

    def __call__(self, samples: Array) -> Array:
        return self._flux.compute_flux(samples[:, 0], 0.0)[self._dim][:, None]

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)

    def jacobian(self, sample: Array) -> Array:
        return self._flux.compute_flux_jacobian(sample[:, 0], 0.0)[self._dim]


def test_values(bkd: Backend[Array]) -> None:
    physics = _physics(bkd)
    points = TransformedMesh2D(_N, _N, bkd).points()
    x, y = points[0], points[1]
    state = bkd.hstack([x + 0.5 * y, 0.5 * x + y])
    mu = 0.5 * (13.0 / 4.0 + 1e-12) ** (-1.0 / 3.0)
    coef = 2.0 * mu * 2.0
    tau_xx, tau_yy, tau_xy = coef * 3.0, coef * 3.0, coef * 0.5
    ones = bkd.ones((_N * _N,))
    q_u = ShallowShelfStressFlux(physics, 0).compute_flux(state, 0.0)
    q_v = ShallowShelfStressFlux(physics, 1).compute_flux(state, 0.0)
    bkd.assert_allclose(q_u[0], -tau_xx * ones, rtol=1e-12)
    bkd.assert_allclose(q_u[1], -tau_xy * ones, rtol=1e-12)
    bkd.assert_allclose(q_v[0], -tau_xy * ones, rtol=1e-12)
    bkd.assert_allclose(q_v[1], -tau_yy * ones, rtol=1e-12)


@pytest.mark.parametrize("component", [0, 1])
def test_jacobian(bkd: Backend[Array], component: int) -> None:
    flux = ShallowShelfStressFlux(_physics(bkd, "variable"), component)
    assert isinstance(flux, FluxProviderProtocol)
    rng = np.random.default_rng(7)
    state = bkd.asarray(rng.normal(0.0, 1.0, (2 * _N * _N, 1)))
    for dim in range(2):
        checker = DerivativeChecker(_FluxDirection(flux, dim, bkd))
        errors = checker.check_derivatives(state, central=True)
        assert checker.error_ratio(errors[0]) <= 1e-6


def test_component_refused(numpy_bkd: Backend[Array]) -> None:
    with pytest.raises(ValueError, match="component must be 0 or 1"):
        ShallowShelfStressFlux(_physics(numpy_bkd), 2)


def test_time_dependence(numpy_bkd: Backend[Array]) -> None:
    constant = ShallowShelfStressFlux(_physics(numpy_bkd), 0)
    assert not constant.flux_is_time_dependent()
    transient = ShallowShelfStressFlux(_physics(numpy_bkd, "transient"), 1)
    assert transient.flux_is_time_dependent()
