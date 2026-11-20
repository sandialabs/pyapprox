"""The collocation neo-Hookean hyperelasticity Jacobian, checked by
finite differences in 1D and 2D.

Residuals and solves against manufactured solutions are verified by the
collocation hyperelasticity catalog
(``tests/pde/models/collocation/test_verification_hyperelasticity.py``).
"""

from typing import Generic

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    AffineTransform2D,
    TransformedMesh1D,
    TransformedMesh2D,
)
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.collocation.protocols.basis import TensorProductBasisProtocol
from pyapprox.util.backends.protocols import Array, Backend


class _Residual(Generic[Array]):
    """The physics residual at time 0 as a function of the state, for
    DerivativeChecker: samples ``(nstates, 1)``, values ``(nstates, 1)``."""

    def __init__(self, physics: HyperelasticityPhysics[Array]) -> None:
        self._physics = physics

    def bkd(self) -> Backend[Array]:
        return self._physics.bkd()

    def nvars(self) -> int:
        return self._physics.nstates()

    def nqoi(self) -> int:
        return self._physics.nstates()

    def __call__(self, sample: Array) -> Array:
        return self._physics.residual(sample[:, 0], 0.0)[:, None]

    def jacobian(self, sample: Array) -> Array:
        return self._physics.jacobian(sample[:, 0], 0.0)

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)


def _basis(
    bkd: Backend[Array], ndim: int, npts_1d: int
) -> TensorProductBasisProtocol[Array]:
    """A Chebyshev basis on the unit line or square."""
    if ndim == 1:
        mesh1 = TransformedMesh1D(npts_1d, bkd, AffineTransform1D((0.0, 1.0), bkd))
        return ChebyshevBasis1D(mesh1, bkd)
    transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
    mesh2 = TransformedMesh2D(npts_1d, npts_1d, bkd, transform)
    return ChebyshevBasis2D(mesh2, bkd)


@pytest.mark.parametrize(
    "ndim, lamda, mu, npts_1d",
    [(1, 1.0, 1.0, 8), (1, 5.0, 10.0, 8), (2, 1.0, 1.0, 6), (2, 0.5, 2.5, 6)],
    ids=["1d", "1d-stiff", "2d", "2d-high-mu"],
)
def test_jacobian(
    bkd: Backend[Array], ndim: int, lamda: float, mu: float, npts_1d: int
) -> None:
    """The state Jacobian at a small displacement against finite
    differences: error ratio at most 1e-5. The steps stop at 1e-2, since
    larger ones can make det F negative."""
    physics = HyperelasticityPhysics(
        _basis(bkd, ndim, npts_1d), bkd, NeoHookeanLaw(bkd), mu=mu, lamda=lamda
    )
    nstates = physics.nstates()
    sample = bkd.asarray([[0.01 * float(i) for i in range(nstates)]]).T
    fd_eps = bkd.flip(bkd.logspace(-13, -2, 12))
    checker = DerivativeChecker(_Residual(physics))
    errors = checker.check_derivatives(sample, fd_eps=fd_eps, verbosity=0)
    assert checker.error_ratio(errors[0]) <= 1e-5
