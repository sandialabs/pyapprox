"""Flux balance identities for the beam, Stokes and hyperelasticity.

At the solution every residual row except the essential rows vanishes,
so weighting the rows by the interpolant of a field in the operator's
kernel turns the essential-boundary flux into minus the applied load
tested with that field, computed here independently of the assembly.
The weights must be the interpolant in the basis's own DOFs:

- Hermite beam: the interpolant of 1 is 1 on value DOFs and 0 on slope
  DOFs, giving the force balance; the interpolant of x is x_i on value
  DOFs and 1 on slope DOFs, giving the moment balance about x = 0.
- Stokes: a constant velocity e_c has no gradient and no divergence, so
  the velocity-component balance needs no pressure rows; a pinned
  pressure DOF is not part of it. Its own flux is the continuity
  equation's compatibility defect.
- Hyperelasticity: rigid translations have zero deformation gradient
  increment, so the balance is the elasticity force balance, at finite
  strain.

Stokes, the beam and Galerkin hyperelasticity have no differentiable
parameterization, so the beam and Stokes use an apply-only stub and
hyperelasticity its apply-only Lame parameterization; balance checks
need values only.
"""

from typing import Any

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import (
    LagrangeBasis,
    MixedBasis,
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.basis.hermite import CubicHermiteBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirectDirichletBC,
    DirichletBC,
    NeumannBC,
)
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics.composite_hyperelasticity import (
    CompositeHyperelasticityPhysics,
)
from pyapprox.pde.galerkin.physics.euler_bernoulli import (
    EulerBernoulliBeamFEM,
)
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    EssentialBoundaryFluxFunctional,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
)
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NumpyArray
from tests._helpers.parameterization_stubs import ApplyOnlyParameterization

_BEAM_LENGTH = 2.0


def _linear_load(x: Any) -> Any:
    """q(x) = 1 + x: integral 4 and first moment 14/3 on [0, 2]."""
    return 1.0 + x


class _ChannelInflow:
    """Parabolic inflow (4 y (1 - y), 0), zero at the walls."""

    def __call__(self, coords: Any) -> Any:
        y = coords[1]
        return np.stack([4.0 * y * (1.0 - y), 0.0 * y])


class _VelocityForcing:
    """f = (a + b y, c), shape (npts, 2) as StokesPhysics expects."""

    def __init__(self, a: float, b: float, c: float) -> None:
        self._a, self._b, self._c = a, b, c

    def __call__(self, coords: Any) -> Any:
        y = coords[1]
        return np.stack([self._a + self._b * y, np.full_like(y, self._c)], axis=1)


class _ConstantScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def __call__(self, coords: Any) -> Any:
        return np.full(coords.shape[-1], self._value)


class _ConstantVector:
    """A constant vector field, shape (2, npts)."""

    def __init__(self, value: Any) -> None:
        self._value = np.asarray(value, dtype=float)

    def __call__(self, coords: Any) -> Any:
        return np.outer(self._value, np.ones(coords.shape[1]))


def _solve(
    bkd: NumpyBkd,
    system: GalerkinSystem[NumpyArray],
    param_obj: Any,
    params: NumpyArray,
) -> NumpyArray:
    param_obj.apply(params)
    result = SteadyStateSolver(system.steady(), tol=1e-12).solve(
        bkd.zeros((system.nstates(),))
    )
    assert result.converged
    return result.solution[:, None]


def _flux(
    bkd: NumpyBkd,
    system: GalerkinSystem[NumpyArray],
    param_obj: Any,
    params: NumpyArray,
    state: NumpyArray,
    dofs: NumpyArray,
    weights: NumpyArray,
) -> NumpyArray:
    """The flux functional's value, shape (nqoi,)."""
    functional = EssentialBoundaryFluxFunctional(
        GalerkinParameterizedResidual(system.spatial_operator(), param_obj),
        system.steady(),
        bkd.asarray(np.asarray(dofs), dtype=bkd.int64_dtype()),
        bkd.asarray(np.asarray(weights)),
    )
    return functional(state, params[:, None])[:, 0]


def _index(bkd: NumpyBkd, dofs: NumpyArray) -> int:
    return int(bkd.to_numpy(dofs)[0])


class TestBeamBalance:
    """EI w'''' = q on [0, 2] with q = 1 + x, Hermite cubics on 8 cells,
    EI = 1.5. Both beams are statically determinate, so the support
    reactions follow from equilibrium alone."""

    def _beam(self, bkd: NumpyBkd) -> EulerBernoulliBeamFEM[NumpyArray]:
        basis = CubicHermiteBasis(StructuredMesh1D(8, (0.0, _BEAM_LENGTH), bkd))
        return EulerBernoulliBeamFEM(basis, 1.5, _linear_load, bkd)

    def test_cantilever_force_and_moment(self, numpy_bkd: NumpyBkd) -> None:
        """Clamped at x = 0, free at x = 2: the interpolant of 1 weights
        the root value DOF (force), that of x the root slope DOF (moment,
        x = 0 at the root). Q = (-integral q, -integral q x)."""
        bkd = numpy_bkd
        beam = self._beam(bkd)
        basis = beam.basis()
        system = compose_galerkin_system(beam, [DirichletBC(basis, "left", 0.0, bkd)])
        param_obj = ApplyOnlyParameterization(beam)
        params = bkd.zeros((1,))
        state = _solve(bkd, system, param_obj, params)
        value = _index(bkd, basis.get_dofs("left", components=(0,)))
        slope = _index(bkd, basis.get_dofs("left", components=(1,)))
        flux = _flux(
            bkd,
            system,
            param_obj,
            params,
            state,
            np.array([value, slope]),
            np.eye(2),
        )
        bkd.assert_allclose(flux, bkd.asarray([-4.0, -14.0 / 3.0]), rtol=1e-10)

    def test_simply_supported_reactions(self, numpy_bkd: NumpyBkd) -> None:
        """Pinned at both ends (value DOFs only): moments about x = 0 give
        the right reaction integral(q x) / L = 7/3, and the force balance
        the left one, 4 - 7/3 = 5/3."""
        bkd = numpy_bkd
        beam = self._beam(bkd)
        basis = beam.basis()
        system = compose_galerkin_system(
            beam,
            [
                DirichletBC(basis, "left", 0.0, bkd, components=(0,)),
                DirichletBC(basis, "right", 0.0, bkd, components=(0,)),
            ],
        )
        param_obj = ApplyOnlyParameterization(beam)
        params = bkd.zeros((1,))
        state = _solve(bkd, system, param_obj, params)
        left = _index(bkd, basis.get_dofs("left", components=(0,)))
        right = _index(bkd, basis.get_dofs("right", components=(0,)))
        flux = _flux(
            bkd,
            system,
            param_obj,
            params,
            state,
            np.array([left, right]),
            np.eye(2),
        )
        bkd.assert_allclose(flux, bkd.asarray([-5.0 / 3.0, -7.0 / 3.0]), rtol=1e-10)


def _stokes_basis(bkd: NumpyBkd) -> MixedBasis[NumpyArray]:
    mesh = StructuredMesh2D(nx=4, ny=4, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
    return MixedBasis(
        [
            ("velocity", VectorLagrangeBasis(mesh, degree=2)),
            ("pressure", LagrangeBasis(mesh, degree=1)),
        ]
    )


def _velocity_component_dofs(bkd: NumpyBkd, bcs: Any, component: int) -> NumpyArray:
    """The essential velocity DOFs of one component, without repeats
    (corners belong to two edges). Velocity DOFs come first and
    interleave the components."""
    dofs = np.unique(
        np.concatenate([bkd.to_numpy(bc.constrained_dofs()) for bc in bcs])
    )
    return dofs[dofs % 2 == component]


class TestStokesBalance:
    """-Laplacian(u) + grad(p) = f, -div(u) = f_p on the unit square,
    Taylor-Hood P2/P1 on 4 x 4 cells, viscosity 1."""

    @pytest.mark.parametrize("component", [0, 1])
    def test_channel_momentum_balance(
        self, numpy_bkd: NumpyBkd, component: int
    ) -> None:
        """Parabolic inflow on the left, no-slip walls, do-nothing outflow
        on the right (zero traction), f = (0.3 + y, -0.2). All momentum
        leaves through the essential boundary, so the velocity-component
        flux there is -integral f_c = (-0.8, 0.2)."""
        bkd = numpy_bkd
        basis = _stokes_basis(bkd)
        physics = StokesPhysics(
            basis,
            bkd,
            vel_forcing=TimeIndependent(_VelocityForcing(0.3, 1.0, -0.2)),
        )
        velocity = basis.field("velocity")
        bcs = [
            DirichletBC(velocity, "left", _ChannelInflow(), bkd),
            DirichletBC(velocity, "bottom", 0.0, bkd),
            DirichletBC(velocity, "top", 0.0, bkd),
        ]
        system = compose_galerkin_system(physics, bcs)
        param_obj = ApplyOnlyParameterization(physics)
        params = bkd.zeros((1,))
        state = _solve(bkd, system, param_obj, params)
        dofs = _velocity_component_dofs(bkd, bcs, component)
        flux = _flux(
            bkd,
            system,
            param_obj,
            params,
            state,
            dofs,
            np.ones((1, dofs.shape[0])),
        )
        expected = [-0.8, 0.2][component]
        bkd.assert_allclose(flux, bkd.asarray([expected]), rtol=1e-10)

    def test_cavity_with_pinned_pressure(self, numpy_bkd: NumpyBkd) -> None:
        """No-slip on every edge, pressure pinned at one DOF, f = (0.3 + y,
        -0.2) and f_p = 0.5. The velocity balance holds without the pin's
        row: Q = (-0.8, 0.2). The pin's own flux is the continuity
        defect: the residual -integral (f_p + div u_h) q vanishes on the
        other pressure rows, and u_h . n = 0 on the boundary, so the pin
        row carries -integral f_p and its flux is +0.5."""
        bkd = numpy_bkd
        basis = _stokes_basis(bkd)
        physics = StokesPhysics(
            basis,
            bkd,
            vel_forcing=TimeIndependent(_VelocityForcing(0.3, 1.0, -0.2)),
            pres_forcing=TimeIndependent(_ConstantScalar(0.5)),
        )
        velocity = basis.field("velocity")
        walls = [
            DirichletBC(velocity, name, 0.0, bkd)
            for name in ("left", "right", "bottom", "top")
        ]
        pin = physics.vel_ndofs()
        bcs = [*walls, DirectDirichletBC([pin], [0.0], bkd)]
        system = compose_galerkin_system(physics, bcs)
        param_obj = ApplyOnlyParameterization(physics)
        params = bkd.zeros((1,))
        state = _solve(bkd, system, param_obj, params)
        x_dofs = _velocity_component_dofs(bkd, walls, 0)
        y_dofs = _velocity_component_dofs(bkd, walls, 1)
        dofs = np.concatenate([x_dofs, y_dofs, [pin]])
        nx, ny = x_dofs.shape[0], y_dofs.shape[0]
        weights = np.zeros((3, dofs.shape[0]))
        weights[0, :nx] = 1.0
        weights[1, nx : nx + ny] = 1.0
        weights[2, -1] = 1.0
        flux = _flux(bkd, system, param_obj, params, state, dofs, weights)
        bkd.assert_allclose(flux, bkd.asarray([-0.8, 0.2, 0.5]), rtol=1e-10, atol=1e-12)


_BODY_FORCE = np.array([0.2, -0.1])
_TRACTION = np.array([0.1, 0.05])


class TestHyperelasticBalance:
    @pytest.mark.parametrize("component", [0, 1])
    def test_support_force_balance(self, numpy_bkd: NumpyBkd, component: int) -> None:
        """Neo-Hookean plate [0, 1]^2, P1 on 6 x 6, (E, nu) = (1, 0.3),
        clamped left, dead-load traction t = (0.1, 0.05) on the right,
        body force f = (0.2, -0.1). In the reference configuration the
        support force cancels the applied loads: Q_c = -(f_c + t_c)."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(nx=6, ny=6, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = CompositeHyperelasticityPhysics.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
            body_force=TimeIndependent(_ConstantVector(_BODY_FORCE)),
        )
        system = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 0.0, bkd),
                NeumannBC(
                    basis,
                    "right",
                    TimeIndependent(_ConstantVector(_TRACTION)),
                    bkd,
                ),
            ],
        )
        param_obj = create_galerkin_lame_parameterization(physics, bkd)
        params = bkd.asarray([1.0, 0.3])
        state = _solve(bkd, system, param_obj, params)
        dofs = bkd.to_numpy(basis.get_dofs("left", components=(component,)))
        flux = _flux(
            bkd,
            system,
            param_obj,
            params,
            state,
            dofs,
            np.ones((1, dofs.shape[0])),
        )
        expected = -(_BODY_FORCE[component] + _TRACTION[component])
        bkd.assert_allclose(flux, bkd.asarray([expected]), rtol=1e-10)
