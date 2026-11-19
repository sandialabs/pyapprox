"""The fluxes a Burgers boundary can prescribe.

- On 10 Chebyshev nodes of [-1, 1] with viscosity 2 and u = x**2 (which
  the derivative matrix differentiates exactly), the viscous flux is
  -4 x and the total flux x**4 / 2 - 4 x, to 1e-12.
- Each flux Jacobian passes DerivativeChecker at a random state, with
  variable viscosity 1 + x / 2.
- A time-dependent viscosity 1 + t + x**2 makes both fluxes
  time-dependent, and the viscous flux of u = x at t = 2 is
  -(3 + x**2); a constant viscosity makes neither time-dependent.
"""

from typing import Generic, Tuple, Union

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics.burgers import (
    BurgersPhysics1D,
    BurgersTotalFlux,
    BurgersViscousFlux,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField, NodalSupplier
from pyapprox.pde.collocation.protocols.boundary import FluxProviderProtocol
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.util.backends.protocols import Array, Backend

_Flux = Union[BurgersViscousFlux[Array], BurgersTotalFlux[Array]]


def _growing_viscosity(coords: Array, time: float) -> Array:
    """nu = 1 + t + x**2, declared time-dependent by its caller."""
    return 1.0 + time + coords[0] ** 2


def _physics(
    bkd: Backend[Array], viscosity: str
) -> Tuple[Array, BurgersPhysics1D[Array]]:
    """The nodes x and the physics with viscosity ``constant`` (2),
    ``variable`` (1 + x/2) or ``transient`` (1 + t + x**2)."""
    basis = ChebyshevBasis1D(TransformedMesh1D(10, bkd), bkd)
    nodes = basis.nodes()
    coefficient: Union[float, NodalField[Array], NodalSupplier[Array]] = 2.0
    if viscosity == "variable":
        coefficient = NodalField(1.0 + nodes / 2.0)
    elif viscosity == "transient":
        coefficient = NodalSupplier(
            TimeDependent(_growing_viscosity), bkd.reshape(nodes, (1, -1))
        )
    return nodes, BurgersPhysics1D(basis, bkd, viscosity=coefficient)


class _FluxFunction(Generic[Array]):
    """A flux component as a function of the state, for DerivativeChecker."""

    def __init__(self, flux: FluxProviderProtocol[Array], bkd: Backend[Array]):
        self._flux, self._bkd = flux, bkd
        self._npts = 10

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._npts

    def nqoi(self) -> int:
        return self._npts

    def __call__(self, samples: Array) -> Array:
        (flux,) = self._flux.compute_flux(samples[:, 0], 0.0)
        return flux[:, None]

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)

    def jacobian(self, sample: Array) -> Array:
        (jacobian,) = self._flux.compute_flux_jacobian(sample[:, 0], 0.0)
        return jacobian


def test_values(bkd: Backend[Array]) -> None:
    x, physics = _physics(bkd, "constant")
    state = x**2
    (viscous,) = BurgersViscousFlux(physics).compute_flux(state, 0.0)
    bkd.assert_allclose(viscous, -4.0 * x, atol=1e-12)
    (total,) = BurgersTotalFlux(physics).compute_flux(state, 0.0)
    bkd.assert_allclose(total, x**4 / 2.0 - 4.0 * x, atol=1e-12)


@pytest.mark.parametrize("total", [False, True], ids=["viscous", "total"])
def test_jacobian(bkd: Backend[Array], total: bool) -> None:
    _, physics = _physics(bkd, "variable")
    flux: _Flux[Array] = (
        BurgersTotalFlux(physics) if total else BurgersViscousFlux(physics)
    )
    assert isinstance(flux, FluxProviderProtocol)
    checker = DerivativeChecker(_FluxFunction(flux, bkd))
    state = bkd.asarray(np.random.default_rng(3).normal(0.0, 1.0, (10, 1)))
    # Central differences: the one-sided error curve is too shallow for the
    # 1e-6 ratio to hold across platform BLAS (CI measured up to 2.4e-6).
    errors = checker.check_derivatives(state, central=True)
    assert checker.error_ratio(errors[0]) <= 1e-6


def test_time_dependence(numpy_bkd: Backend[Array]) -> None:
    _, constant = _physics(numpy_bkd, "constant")
    assert not BurgersViscousFlux(constant).flux_is_time_dependent()
    assert not BurgersTotalFlux(constant).flux_is_time_dependent()
    x, transient = _physics(numpy_bkd, "transient")
    assert BurgersViscousFlux(transient).flux_is_time_dependent()
    assert BurgersTotalFlux(transient).flux_is_time_dependent()
    (flux,) = BurgersViscousFlux(transient).compute_flux(x, 2.0)
    numpy_bkd.assert_allclose(flux, -(3.0 + x**2), atol=1e-12)
