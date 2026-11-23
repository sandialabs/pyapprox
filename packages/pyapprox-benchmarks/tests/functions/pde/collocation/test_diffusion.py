r"""Tests of the collocation 1D diffusion builders, on :math:`[0, 2]`.

Every case meshes :math:`[0, 2]` rather than the reference
:math:`[-1, 1]`, so a builder that ignored the mesh's transform would
fail.

Steady (``collocation_steady_function`` defaults):

- with :math:`D = 1 + p_0` and :math:`f = \pi^2 \sin \pi x`, every state
  equals :math:`\sin(\pi x)/(1 + p_0)` at :math:`p_0 = 0` and 0.5 to 1e-9
  (20 nodes);
- with a two-term lognormal KLE diffusivity, the Jacobian and HVP pass
  central finite differences (smallest relative error below 1e-6).

Transient (``collocation_transient_system`` through
``TransientImplicitMap``, backward Euler to 0.1 in steps of 0.02, the
step-end pin): with zero Dirichlet and with gradient Robin ends and the
KLE diffusivity, the middle node's final value has a Jacobian and HVP
passing central finite differences (smallest relative error below 1e-6).
"""

import math
from typing import Generic, Sequence, Tuple

import pytest

from pyapprox.implicitfunction.implicit_map import (
    ArrayParameters,
    ImplicitMapFunction,
)
from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.pinned_rows import AtStepEnd
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import backward_euler
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TransientImplicitMap,
    TransientPoint,
)
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    BoundarySide,
    TransformedMesh1D,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.kle_factory import create_lognormal_kle_field_map
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.models.collocation.steady_function import (
    collocation_steady_function,
)
from pyapprox.pde.models.collocation.transient_system import (
    collocation_transient_system,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton
from pyapprox_benchmarks.functions.pde.collocation.boundaries import (
    BoundarySpecProtocol,
    Dirichlet,
    FluxNeumann,
    GradientRobin,
    Periodic,
)
from pyapprox_benchmarks.functions.pde.collocation.diffusion import (
    build_steady_diffusion_1d,
    build_transient_diffusion_1d,
)

_DOMAIN = (0.0, 2.0)


def _mesh(bkd: Backend[Array], npts: int) -> TransformedMesh1D[Array]:
    return TransformedMesh1D(npts, bkd, AffineTransform1D(_DOMAIN, bkd))


class _SineForcing(Generic[Array]):
    r""":math:`\pi^2 \sin \pi x`."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, coords: Array) -> Array:
        return (math.pi**2) * self._bkd.sin(math.pi * coords[0])


def _kle_map(bkd: Backend[Array], points: Array) -> TransformedFieldMap[Array]:
    return create_lognormal_kle_field_map(
        points, bkd.zeros((points.shape[1],)), bkd, num_kle_terms=2, sigma=0.3
    )


def _every_state(bkd: Backend[Array], nstates: int) -> StateFunctional[Array, Array]:
    return StateFunctional.from_terms(
        bkd,
        nstates,
        state_terms=(ObservedValue(StateEntries(bkd, nstates, list(range(nstates)))),),
    )


def _middle(bkd: Backend[Array], nstates: int) -> StateFunctional[Array, Array]:
    return StateFunctional.from_terms(
        bkd,
        1,
        state_terms=(ObservedValue(StateEntries(bkd, nstates, [nstates // 2])),),
    )


def _close(bkd: Backend[Array], actual: Array, expected: Array, rtol: float) -> None:
    scale = max(bkd.to_float(bkd.max(bkd.abs(expected))), 1.0)
    bkd.assert_allclose(actual, expected, rtol=0.0, atol=rtol * scale)


class TestSteadyDiffusion:
    def test_exact_solution_on_the_physical_interval(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd, 20)
        configured = build_steady_diffusion_1d(
            mesh,
            TimeIndependent(_SineForcing(bkd)),
            BasisExpansion(bkd, 1.0, [bkd.ones((20,))]),
        )
        function = collocation_steady_function(
            configured.system,
            ParameterState(configured.parameterization),
            _every_state(bkd, 20),
        )
        x = mesh.points()[0]
        for p0 in (0.0, 0.5):
            exact = bkd.reshape(bkd.sin(math.pi * x) / (1.0 + p0), (20, 1))
            sample = bkd.reshape(bkd.asarray([p0]), (1, 1))
            bkd.assert_allclose(function(sample), exact, rtol=0.0, atol=1e-9)

    def test_derivatives_pass_finite_differences(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd, 20)
        configured = build_steady_diffusion_1d(
            mesh, TimeIndependent(_SineForcing(bkd)), _kle_map(bkd, mesh.points())
        )
        function = collocation_steady_function(
            configured.system,
            ParameterState(configured.parameterization),
            _middle(bkd, 20),
        )
        assert function.derivatives().hvp is not None
        errors = DerivativeChecker(function).check_derivatives(
            bkd.reshape(bkd.asarray([0.1, -0.1]), (2, 1)),
            relative=True,
            central=True,
        )
        assert len(errors) == 2
        for error in errors:
            assert bkd.to_float(bkd.min(error)) < 1e-6


def _transient_function(
    bkd: Backend[Array],
    boundaries: Sequence[BoundarySpecProtocol],
) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
    """The middle node's value after five BE steps of 0.02 from
    :math:`\\sin \\pi x` (15 nodes on [0, 2], KLE diffusivity), the
    boundary rows pinned at the step end."""
    mesh = _mesh(bkd, 15)
    configured = build_transient_diffusion_1d(
        mesh, _kle_map(bkd, mesh.points()), boundaries=boundaries
    )
    transient = collocation_transient_system(
        configured.system,
        AtStepEnd(),
        None,
        ParameterState(configured.parameterization),
    )
    newton: Newton[Array, Array] = Newton(atol=1e-12, rtol=0.0, maxiters=20)
    functional = FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd, 1, state_terms=(EndpointTerm(StateEntries(bkd, 15, [7])),)
        )
    )
    implicit = TransientImplicitMap(
        transient,
        transient,
        RungeKutta(backward_euler(), newton),
        RecomputeAlways(transient.solved.solver),
        FixedStep(0.02, (0.0, 0.1)),
        bkd.reshape(bkd.sin(math.pi * mesh.points()[0]), (-1, 1)),
        functional,
    )
    return ImplicitMapFunction(implicit, ArrayParameters(bkd))


class TestTransientDiffusion:
    @pytest.mark.parametrize("ends", ["dirichlet", "robin"])
    def test_derivatives_pass_central_differences(
        self, bkd: Backend[Array], ends: str
    ) -> None:
        boundaries: Tuple[BoundarySpecProtocol, ...] = (
            (Dirichlet(BoundarySide.LEFT, 0.0), Dirichlet(BoundarySide.RIGHT, 0.0))
            if ends == "dirichlet"
            else (
                GradientRobin(BoundarySide.LEFT, 1.0, 0.5, 0.0),
                GradientRobin(BoundarySide.RIGHT, 1.0, 0.5, 0.0),
            )
        )
        function = _transient_function(bkd, boundaries)
        assert isinstance(function, FunctionProtocol)
        assert function.derivatives().hvp is not None
        errors = DerivativeChecker(function).check_derivatives(
            bkd.reshape(bkd.asarray([0.1, -0.1]), (2, 1)),
            direction=bkd.reshape(bkd.asarray([0.6, -0.8]), (2, 1)),
            relative=True,
            central=True,
        )
        assert len(errors) == 2
        for error in errors:
            assert bkd.to_float(bkd.min(error)) < 1e-6


class _CosineForcing(Generic[Array]):
    r""":math:`\cos \pi x + 1`."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, coords: Array) -> Array:
        return self._bkd.cos(math.pi * coords[0]) + 1.0


def _constant_diffusion_solve(
    bkd: Backend[Array],
    boundaries: Sequence[BoundarySpecProtocol],
    p0: float,
    forcing: TimeIndependent[Array],
    reaction: float = 0.0,
) -> Tuple[Array, Array]:
    """The states at :math:`D = 1 + p_0` (20 nodes on [0, 2]) and the
    physical nodes."""
    mesh = _mesh(bkd, 20)
    configured = build_steady_diffusion_1d(
        mesh,
        forcing,
        BasisExpansion(bkd, 1.0, [bkd.ones((20,))]),
        boundaries=boundaries,
        reaction=reaction,
    )
    function = collocation_steady_function(
        configured.system,
        ParameterState(configured.parameterization),
        _every_state(bkd, 20),
    )
    sample = bkd.reshape(bkd.asarray([p0]), (1, 1))
    return function(sample)[:, 0], mesh.points()[0]


@pytest.mark.parametrize("p0", [0.0, 0.5])
class TestBoundarySpecs:
    r"""Each spec kind against an exact solution, :math:`\kappa = 1 + p_0`,
    :math:`s = \sin \pi x`, to 1e-9."""

    def test_dirichlet(self, bkd: Backend[Array], p0: float) -> None:
        """:math:`u(0) = 0.5`, :math:`u(2) = -0.3`:
        :math:`u = s/\\kappa + 0.5 - 0.4 x`."""
        state, x = _constant_diffusion_solve(
            bkd,
            (Dirichlet(BoundarySide.LEFT, 0.5), Dirichlet(BoundarySide.RIGHT, -0.3)),
            p0,
            TimeIndependent(_SineForcing(bkd)),
        )
        exact = bkd.sin(math.pi * x) / (1.0 + p0) + 0.5 - 0.4 * x
        bkd.assert_allclose(state, exact, rtol=0.0, atol=1e-9)

    def test_gradient_robin(self, bkd: Backend[Array], p0: float) -> None:
        r""":math:`u + 0.5\, u' n = g` with :math:`g = 0.2` at 0 and
        :math:`-0.4` at 2: :math:`u = s/\kappa + c + d x`,
        :math:`d = (g_1 - g_0 - k)/3`, :math:`c = g_0 + (k + d)/2`,
        :math:`k = \pi/\kappa`."""
        g0, g1 = 0.2, -0.4
        state, x = _constant_diffusion_solve(
            bkd,
            (
                GradientRobin(BoundarySide.LEFT, 1.0, 0.5, g0),
                GradientRobin(BoundarySide.RIGHT, 1.0, 0.5, g1),
            ),
            p0,
            TimeIndependent(_SineForcing(bkd)),
        )
        k = math.pi / (1.0 + p0)
        d = (g1 - g0 - k) / 3.0
        c = g0 + 0.5 * (k + d)
        exact = bkd.sin(math.pi * x) / (1.0 + p0) + c + d * x
        bkd.assert_allclose(state, exact, rtol=0.0, atol=1e-9)

    def test_flux_neumann(self, bkd: Backend[Array], p0: float) -> None:
        r""":math:`u(0) = 0`, :math:`-\kappa u'(2) = g = 0.7`:
        :math:`u = s/\kappa + d x`, :math:`d = -(g + \pi)/\kappa`."""
        g = 0.7
        state, x = _constant_diffusion_solve(
            bkd,
            (Dirichlet(BoundarySide.LEFT, 0.0), FluxNeumann(BoundarySide.RIGHT, g)),
            p0,
            TimeIndependent(_SineForcing(bkd)),
        )
        exact = bkd.sin(math.pi * x) / (1.0 + p0) - (g + math.pi) / (1.0 + p0) * x
        bkd.assert_allclose(state, exact, rtol=0.0, atol=1e-9)

    def test_periodic(self, bkd: Backend[Array], p0: float) -> None:
        r"""Periodic ends, :math:`r = -1` (the physics' rate sign, so
        :math:`-(\kappa u')' + u = f`), :math:`f = \cos \pi x + 1`:
        :math:`u = \cos \pi x / (\kappa \pi^2 + 1) + 1`."""
        state, x = _constant_diffusion_solve(
            bkd,
            (Periodic(),),
            p0,
            TimeIndependent(_CosineForcing(bkd)),
            reaction=-1.0,
        )
        exact = bkd.cos(math.pi * x) / ((1.0 + p0) * math.pi**2 + 1.0) + 1.0
        bkd.assert_allclose(state, exact, rtol=0.0, atol=1e-9)
