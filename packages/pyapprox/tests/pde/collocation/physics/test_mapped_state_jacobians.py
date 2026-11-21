"""State Jacobians on curvilinear meshes, checked by finite differences.

The polar and elliptical maps over the full angle give the physics
non-constant gradient factors, which a box mesh does not exercise:
advection-diffusion-reaction with every term on, linear elasticity with
variable Lamé values, and neo-Hookean hyperelasticity.
"""

import math
from typing import Callable, Generic, Protocol

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms.elliptical import (
    EllipticalTransform,
)
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.collocation.protocols.basis import TensorProductBasisProtocol
from pyapprox.util.backends.protocols import Array, Backend


class _PhysicsProtocol(Protocol[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nstates(self) -> int: ...

    def residual(self, state: Array, time: float) -> Array: ...

    def jacobian(self, state: Array, time: float) -> Array: ...


class _Residual(Generic[Array]):
    """The physics residual at time 0 as a function of the state, for
    DerivativeChecker: samples ``(nstates, 1)``, values ``(nstates, 1)``."""

    def __init__(self, physics: _PhysicsProtocol[Array]) -> None:
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


def _polar(bkd: Backend[Array]) -> ChebyshevBasis2D[Array]:
    transform = PolarTransform((1.0, 2.0), (0.0, 2.0 * math.pi), bkd)
    return ChebyshevBasis2D(TransformedMesh2D(6, 7, bkd, transform), bkd)


def _elliptical(bkd: Backend[Array]) -> ChebyshevBasis2D[Array]:
    transform = EllipticalTransform((0.5, 1.0), (0.0, 2.0 * math.pi), 1.0, bkd)
    return ChebyshevBasis2D(TransformedMesh2D(6, 7, bkd, transform), bkd)


_Mesh = Callable[[Backend[Array]], ChebyshevBasis2D[Array]]


def _adr(
    basis: TensorProductBasisProtocol[Array], rng: np.random.Generator
) -> AdvectionDiffusionReaction[Array]:
    bkd = basis.bkd()
    npts = basis.npts()
    return AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=NodalField(bkd.asarray(rng.uniform(1.5, 2.5, npts))),
        velocity=[bkd.asarray(rng.uniform(-1.0, 1.0, npts)) for _ in range(2)],
        reaction=NodalField(bkd.asarray(rng.uniform(-1.0, 1.0, npts))),
    )


def _linear_elasticity(
    basis: TensorProductBasisProtocol[Array], rng: np.random.Generator
) -> LinearElasticityPhysics[Array]:
    bkd = basis.bkd()
    npts = basis.npts()
    return LinearElasticityPhysics(
        basis,
        bkd,
        lamda=NodalField(bkd.asarray(rng.uniform(1.3, 1.7, npts))),
        mu=NodalField(bkd.asarray(rng.uniform(0.7, 1.3, npts))),
    )


def _hyperelasticity(
    basis: TensorProductBasisProtocol[Array], rng: np.random.Generator
) -> HyperelasticityPhysics[Array]:
    bkd = basis.bkd()
    npts = basis.npts()
    return HyperelasticityPhysics(
        basis,
        bkd,
        NeoHookeanLaw(bkd),
        mu=NodalField(bkd.asarray(rng.uniform(0.7, 1.3, npts))),
        lamda=NodalField(bkd.asarray(rng.uniform(1.3, 1.7, npts))),
    )


_Physics = Callable[
    [TensorProductBasisProtocol[Array], np.random.Generator], _PhysicsProtocol[Array]
]


@pytest.mark.parametrize("mesh", [_polar, _elliptical], ids=["polar", "elliptical"])
@pytest.mark.parametrize(
    "physics, scale",
    [(_adr, 1.0), (_linear_elasticity, 1.0), (_hyperelasticity, 0.05)],
    ids=["adr", "linear-elasticity", "hyperelasticity"],
)
def test_state_jacobian(
    bkd: Backend[Array],
    mesh: _Mesh[Array],
    physics: _Physics[Array],
    scale: float,
) -> None:
    """6 x 7 nodes on the ring 1 <= r <= 2 and between the ellipses
    u = 0.5 and 1, coefficients random per node (D in [1.5, 2.5], v and
    r in [-1, 1]; lambda in [1.3, 1.7], mu in [0.7, 1.3]): error ratio
    at most 1e-6 at the smooth state ``scale`` sin(x + y + k) in field k
    (small for the hyperelastic one, so det F stays positive; a random
    nodal state has large spectral derivatives)."""
    rng = np.random.default_rng(1)
    basis = mesh(bkd)
    model = physics(basis, rng)
    checker = DerivativeChecker(_Residual(model))
    points = basis.mesh().points()
    phase = points[0] + points[1]
    nfields = model.nstates() // basis.npts()
    state = scale * bkd.concatenate([bkd.sin(phase + k) for k in range(nfields)])
    state = state[:, None]
    errors = checker.check_derivatives(state, verbosity=0)
    assert checker.error_ratio(errors[0]) <= 1e-6
