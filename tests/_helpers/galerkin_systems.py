"""Small transient Galerkin systems for testing assembly into stacks.

A scalar linear physics, a scalar nonlinear one and a mixed one (whose
mass is singular), each with essential constraints, on the NumPy
backend. Importing this module needs skfem; import it only after a
test module has skipped without it.
"""

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.protocols import SpatialOperatorProtocol
from pyapprox.pde.assembly import ElementDofMapProtocol
from pyapprox.pde.boundary import BoundaryConditionRole, ConstraintSetProtocol
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import LagrangeBasis, MixedBasis
from pyapprox.pde.galerkin.basis.mixed import MixedFieldBasisProtocol
from pyapprox.pde.galerkin.basis.vector_lagrange import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.physics.burgers import BurgersPhysics
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.steady_view import SteadyView
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.operators import SpaceDescriptor

NpArray = NDArray[Any]
Dofs = NDArray[np.int64]
NpSystem = GalerkinSystem[NpArray]

#: The names :func:`galerkin_system` accepts.
SYSTEM_NAMES = ["adr", "burgers", "stokes"]


def _zero(x: NpArray) -> NpArray:
    return np.zeros(x.shape[1])


def _ones(x: NpArray) -> NpArray:
    return np.ones(x.shape[-1])


def _zero_velocity(x: NpArray) -> NpArray:
    return np.zeros((2, x.shape[1]))


def adr_system(bkd: NumpyBkd) -> tuple[NpSystem, LagrangeBasis[NpArray]]:
    """Linear diffusion-reaction, P1 on a 3x2 mesh, Dirichlet on the left."""
    mesh = StructuredMesh2D(3, 2, [(0.0, 1.0), (0.0, 1.0)], bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=0.5,
        bkd=bkd,
        reaction=2.0,
        forcing=TimeIndependent(_zero),
    )
    bc = DirichletBC(basis, "left", TimeIndependent(_zero), bkd)
    return compose_galerkin_system(physics, [bc]), basis


def burgers_system(bkd: NumpyBkd) -> tuple[NpSystem, LagrangeBasis[NpArray]]:
    """Viscous Burgers, P2 on 6 elements, Dirichlet at both ends."""
    mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=2)
    physics = BurgersPhysics(
        basis=basis, viscosity=0.1, bkd=bkd, forcing=TimeIndependent(_ones)
    )
    bcs: Sequence[BoundaryConditionRole[NpArray]] = [
        DirichletBC(basis, name, TimeIndependent(_zero), bkd)
        for name in ("left", "right")
    ]
    return compose_galerkin_system(physics, bcs), basis


def stokes_system(bkd: NumpyBkd) -> tuple[NpSystem, MixedBasis[NpArray]]:
    """Taylor-Hood Stokes on a 3x3 mesh, velocity fixed on the left."""
    mesh = StructuredMesh2D(nx=3, ny=3, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
    fields: list[tuple[str, MixedFieldBasisProtocol[NpArray]]] = [
        ("velocity", VectorLagrangeBasis(mesh, degree=2)),
        ("pressure", LagrangeBasis(mesh, degree=1)),
    ]
    basis = MixedBasis(fields)
    bc = DirichletBC(
        basis.field("velocity"), "left", TimeIndependent(_zero_velocity), bkd
    )
    return compose_galerkin_system(StokesPhysics(basis, bkd), [bc]), basis


_Builder = Callable[[NumpyBkd], tuple[NpSystem, ElementDofMapProtocol[Dofs]]]
_BUILDERS: dict[str, _Builder] = {
    "adr": adr_system,
    "burgers": burgers_system,
    "stokes": stokes_system,
}


class MassScaledSystem:
    """A system whose mass can be rescaled, delegating everything else.

    Its mass matrix is the same object on every call until ``scale``
    assembles a new one, as mass suppliers cache theirs, so a consumer
    keyed on that object sees exactly one change per ``scale``.
    """

    def __init__(self, inner: NpSystem) -> None:
        self._inner = inner
        self._mass = inner.mass_matrix()

    def scale(self, factor: float) -> None:
        """Replace the mass by ``factor`` times the inner system's."""
        self._mass = factor * self._inner.mass_matrix()

    def bkd(self) -> Backend[NpArray]:
        return self._inner.bkd()

    def nstates(self) -> int:
        return self._inner.nstates()

    def spatial_operator(self) -> SpatialOperatorProtocol[NpArray]:
        return self._inner.spatial_operator()

    def constraint_set(self) -> ConstraintSetProtocol[NpArray]:
        return self._inner.constraint_set()

    def steady(self) -> SteadyView[NpArray]:
        return self._inner.steady()

    def steady_snapshot(self, time: float) -> SteadyView[NpArray]:
        return self._inner.steady_snapshot(time)

    def mass_matrix(self) -> NpArray:
        return self._mass


def galerkin_system(
    name: str, bkd: NumpyBkd
) -> tuple[NpSystem, ElementDofMapProtocol[Dofs], SpaceDescriptor]:
    """The named system, its basis as a DOF map, and its state space."""
    system, basis = _BUILDERS[name](bkd)
    return system, basis, SpaceDescriptor("state", system.nstates())
