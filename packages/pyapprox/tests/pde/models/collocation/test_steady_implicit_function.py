r"""Collocation steady states as implicit functions of the parameters.

``collocation_steady_function`` with its defaults (``Newton`` atol 1e-10,
above the flux rows' roundoff floor near 1.5e-11, with a line search;
dense LU, ``RecomputeAlways``), the functional the state at the middle
node, on these problems:

- ``dirichlet``: 1D diffusion on :math:`[-1, 1]` (20 nodes), :math:`D = 2
  + p_0 + p_1 x`, forcing :math:`\pi^2 \sin \pi x`, zero Dirichlet ends;
- ``flux neumann``: on :math:`[0, 1]`, zero Dirichlet at 0 and the total
  flux 0 at 1, the same diffusion map;
- ``gradient robin``: on :math:`[0, 1]`, :math:`u + 0.5\, u' n = 0` at
  both ends, forcing 1, the same diffusion map;
- ``log kle``: on :math:`[-1, 1]`, :math:`D = \exp(W \theta)` with three
  modes :math:`0.4 \sin((k+1)\pi x)/(k+1)`, zero Dirichlet ends, a zero
  nodal reaction;
- ``composite``: ``log kle`` with also a forcing amplitude
  :math:`p_3 \sin \pi x` and a reaction :math:`-0.5 + p_4`, five
  parameters;
- ``first-order map``: 10 nodes, :math:`D = 2 + (p_0 + p_1 x)^2` from a
  map that declares no curvature, otherwise as ``log kle``;
- ``periodic``: 15 nodes, :math:`D = 1 + p_0 + p_1 \cos \pi x`, reaction
  1, forcing :math:`1 + \cos \pi x + 0.5 \sin \pi x`, periodic ends, the
  functional the value at the primary node.

Tests:

- the Jacobian and, where present, the HVP pass central finite
  differences (smallest relative error below 1e-6), and the HVP is
  symmetric, :math:`\langle H v, w \rangle = \langle H w, v \rangle` to
  1e-10;
- the HVP is present exactly when the closed right-hand side declares
  second derivatives. It is absent for the flux-Neumann rows, whose
  parameter Jacobian comes from the parameterization's flux sensitivity,
  which supplies no second derivative. It is absent for the map without
  curvature. It is present for the others: linear maps state their
  curvature zero;
- a parameterization declaring no derivatives gives values and no
  Jacobian; the scalar function runs inside the OUU ``InexactWrapper``.
"""

import dataclasses
import math
from typing import Callable, Dict, Generic, List, Optional, Tuple

import numpy as np
import pytest

from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import (
    SteadyImplicitFunction,
    TangentColumns,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsProtocol,
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic_rows import PeriodicPart, PeriodicRows
from pyapprox.pde.collocation.mesh import AffineTransform1D, TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField, NodalSupplier
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.scalar import ScalarAmplitude
from pyapprox.pde.field_maps.transformed import TransformedFieldMap, _ExpTransform
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_function import (
    ParameterizedCollocationSystem,
    collocation_steady_function,
)
from pyapprox.pde.parameterizations.collocation_advection_diffusion import (
    CollocationAdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.composite import CompositeParameterization
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.adjoint_checks import NoHVPQuadraticFieldMap

_CASES = [
    "dirichlet",
    "flux neumann",
    "gradient robin",
    "log kle",
    "composite",
    "first-order map",
    "periodic",
]
_WITHOUT_HVP = {"flux neumann", "first-order map"}


class _Problem(Generic[Array]):
    """A physics, its system, and its parameterization, sample and
    observed node."""

    def __init__(
        self,
        bkd: Backend[Array],
        physics: AdvectionDiffusionReaction[Array],
        essential: Optional[DirichletRows[Array]],
        closures: List[ClosureRowsProtocol[Array]],
        parameterization: ParameterizationProtocol[Array],
        param: List[float],
        node: int,
    ) -> None:
        self.bkd, self.physics = bkd, physics
        self.system = CollocationSystem(physics, essential, closures)
        self.parameterization = parameterization
        self.param = bkd.reshape(bkd.asarray(param), (len(param), 1))
        self.node = node
        self.nstates = physics.nstates()

    def function(self) -> SteadyImplicitFunction[Array, DenseFormat[Array]]:
        bkd = self.bkd
        parameters = ParameterState(self.parameterization)
        functional: StateFunctional[Array, Array] = StateFunctional.from_terms(
            bkd,
            1,
            state_terms=(ObservedValue(StateEntries(bkd, self.nstates, [self.node])),),
        )
        return collocation_steady_function(self.system, parameters, functional)


def _diffusion_map(
    bkd: Backend[Array], base: float, modes: List[Array]
) -> BasisExpansion[Array]:
    return BasisExpansion(bkd, base, modes)


def _sine_forcing(bkd: Backend[Array], nodes: Array) -> NodalField[Array]:
    return NodalField((math.pi**2) * bkd.sin(math.pi * nodes))


def _dirichlet(bkd: Backend[Array]) -> _Problem[Array]:
    mesh = TransformedMesh1D(20, bkd, AffineTransform1D((-1.0, 1.0), bkd))
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = mesh.points()[0]
    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=2.0, forcing=_sine_forcing(bkd, nodes)
    )
    left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
    return _Problem(
        bkd,
        physics,
        DirichletRows(
            bkd, 20, 1, [DirichletPart(left, 0.0), DirichletPart(right, 0.0)]
        ),
        [],
        create_diffusion_parameterization(
            physics, bkd, _diffusion_map(bkd, 2.0, [bkd.ones((20,)), nodes])
        ),
        [0.3, -0.2],
        10,
    )


def _flux_neumann(bkd: Backend[Array]) -> _Problem[Array]:
    mesh = TransformedMesh1D(20, bkd, AffineTransform1D((0.0, 1.0), bkd))
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = mesh.points()[0]
    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=2.0, forcing=_sine_forcing(bkd, nodes)
    )
    left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
    normals = mesh.boundary_normals(1)
    flux = RobinRows(
        bkd,
        20,
        1,
        [
            RobinPart(
                right, FluxNormalOperator(bkd, right, normals, physics), 0.0, 1.0, 0.0
            )
        ],
    )
    return _Problem(
        bkd,
        physics,
        DirichletRows(bkd, 20, 1, [DirichletPart(left, 0.0)]),
        [flux],
        create_diffusion_parameterization(
            physics, bkd, _diffusion_map(bkd, 2.0, [bkd.ones((20,)), nodes])
        ),
        [0.3, 0.1],
        10,
    )


def _gradient_robin(bkd: Backend[Array]) -> _Problem[Array]:
    mesh = TransformedMesh1D(20, bkd, AffineTransform1D((0.0, 1.0), bkd))
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = mesh.points()[0]
    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=2.0, forcing=NodalField(bkd.ones((20,)))
    )
    derivatives = [basis.derivative_matrix(1, 0)]
    parts = []
    for side in (0, 1):
        indices, normals = mesh.boundary_indices(side), mesh.boundary_normals(side)
        parts.append(
            RobinPart(
                indices,
                GradientNormalOperator(bkd, indices, normals, derivatives),
                1.0,
                0.5,
                0.0,
            )
        )
    return _Problem(
        bkd,
        physics,
        None,
        [RobinRows(bkd, 20, 1, parts)],
        create_diffusion_parameterization(
            physics, bkd, _diffusion_map(bkd, 2.0, [bkd.ones((20,)), nodes])
        ),
        [0.3, 0.1],
        10,
    )


class _ZeroEnds(Generic[Array]):
    r"""Diffusion 1, a zero nodal reaction (so a reaction map has a field
    to set), forcing :math:`\pi^2 \sin \pi x` and zero Dirichlet ends on
    :math:`[-1, 1]`."""

    def __init__(self, bkd: Backend[Array], npts: int) -> None:
        mesh = TransformedMesh1D(npts, bkd, AffineTransform1D((-1.0, 1.0), bkd))
        self.bkd, self.npts = bkd, npts
        self.nodes = mesh.points()[0]
        self.physics = AdvectionDiffusionReaction(
            ChebyshevBasis1D(mesh, bkd),
            bkd,
            diffusion=1.0,
            reaction=NodalField(bkd.zeros((npts,))),
            forcing=_sine_forcing(bkd, self.nodes),
        )
        self._left, self._right = mesh.boundary_indices(0), mesh.boundary_indices(1)

    def problem(
        self, parameterization: ParameterizationProtocol[Array], param: List[float]
    ) -> _Problem[Array]:
        bkd, left, right = self.bkd, self._left, self._right
        return _Problem(
            bkd,
            self.physics,
            DirichletRows(
                bkd,
                self.npts,
                1,
                [DirichletPart(left, 0.0), DirichletPart(right, 0.0)],
            ),
            [],
            parameterization,
            param,
            self.npts // 2,
        )


def _log_kle_map(bkd: Backend[Array], nodes: Array) -> TransformedFieldMap[Array]:
    r""":math:`D = \exp(W \theta)`, three modes."""
    modes = bkd.stack(
        [0.4 * bkd.sin((k + 1) * math.pi * nodes) / (k + 1) for k in range(3)],
        axis=1,
    )
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(
        MeshKLEFieldMap(bkd, bkd.zeros(tuple(nodes.shape)), modes),
        exp,
        exp,
        bkd,
        transform_deriv2=exp,
    )


def _log_kle(bkd: Backend[Array]) -> _Problem[Array]:
    base = _ZeroEnds(bkd, 20)
    return base.problem(
        create_diffusion_parameterization(
            base.physics, bkd, _log_kle_map(bkd, base.nodes)
        ),
        [0.4, -0.3, 0.2],
    )


def _composite(bkd: Backend[Array]) -> _Problem[Array]:
    base = _ZeroEnds(bkd, 20)
    physics, nodes = base.physics, base.nodes
    return base.problem(
        CompositeParameterization(
            [
                create_diffusion_parameterization(
                    physics, bkd, _log_kle_map(bkd, nodes)
                ),
                CollocationAdvectionDiffusionParameterization(
                    physics,
                    forcing_map=ScalarAmplitude(bkd, bkd.sin(math.pi * nodes)),
                    bkd=bkd,
                ),
                CollocationAdvectionDiffusionParameterization(
                    physics,
                    reaction_map=BasisExpansion(bkd, -0.5, [bkd.ones((20,))]),
                    bkd=bkd,
                ),
            ],
            bkd,
        ),
        [0.4, -0.3, 0.2, 1.0, 0.3],
    )


def _first_order_map(bkd: Backend[Array]) -> _Problem[Array]:
    base = _ZeroEnds(bkd, 10)
    modes = bkd.stack([bkd.ones((10,)), base.nodes], axis=1)
    return base.problem(
        create_diffusion_parameterization(
            base.physics,
            bkd,
            NoHVPQuadraticFieldMap(bkd, bkd.full((10,), 2.0), modes),
        ),
        [0.3, -0.2],
    )


def _periodic(bkd: Backend[Array]) -> _Problem[Array]:
    mesh = TransformedMesh1D(15, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = basis.nodes()
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=1.0,
        reaction=1.0,
        forcing=NodalField(
            1.0 + bkd.cos(math.pi * nodes) + 0.5 * bkd.sin(math.pi * nodes)
        ),
    )
    left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
    derivative = basis.derivative_matrix()
    return _Problem(
        bkd,
        physics,
        None,
        [
            PeriodicRows(
                bkd, mesh.reference_points(), 1, [PeriodicPart(left, right, derivative)]
            )
        ],
        create_diffusion_parameterization(
            physics,
            bkd,
            _diffusion_map(bkd, 1.0, [bkd.ones((15,)), bkd.cos(math.pi * nodes)]),
        ),
        [0.3, 0.2],
        bkd.to_int(left[0]),
    )


_BUILDERS: Dict[str, Callable[[Backend[Array]], _Problem[Array]]] = {
    "dirichlet": _dirichlet,
    "flux neumann": _flux_neumann,
    "gradient robin": _gradient_robin,
    "log kle": _log_kle,
    "composite": _composite,
    "first-order map": _first_order_map,
    "periodic": _periodic,
}


def _close(bkd: Backend[Array], actual: Array, expected: Array, rtol: float) -> None:
    """Equal to ``rtol`` relative to the larger entry."""
    scale = max(bkd.to_float(bkd.max(bkd.abs(expected))), 1.0)
    bkd.assert_allclose(actual, expected, rtol=0.0, atol=rtol * scale)


@pytest.mark.parametrize("name", _CASES)
class TestDerivatives:
    def test_finite_differences(self, bkd: Backend[Array], name: str) -> None:
        problem = _BUILDERS[name](bkd)
        function = problem.function()
        has_hvp = function.derivatives().hvp is not None
        assert has_hvp == (name not in _WITHOUT_HVP)
        errors = DerivativeChecker(function).check_derivatives(
            problem.param, relative=True, central=True
        )
        assert len(errors) == (2 if has_hvp else 1)
        for error in errors:
            assert bkd.to_float(bkd.min(error)) < 1e-6

    def test_hvp_is_symmetric(self, bkd: Backend[Array], name: str) -> None:
        problem = _BUILDERS[name](bkd)
        hvp = problem.function().derivatives().hvp
        if name in _WITHOUT_HVP:
            assert hvp is None
            return
        assert hvp is not None
        rng = np.random.default_rng(6)
        nparams = problem.param.shape[0]
        v = bkd.asarray(rng.normal(size=(nparams, 1)))
        w = bkd.asarray(rng.normal(size=(nparams, 1)))
        h_v, h_w = hvp(problem.param, v), hvp(problem.param, w)
        bkd.assert_allclose(bkd.sum(h_v * w), bkd.sum(h_w * v), rtol=1e-10)


def _growing(coords: Array, time: float) -> Array:
    """A forcing growing in time."""
    return (1.0 + time) * (2.0 + coords[0] ** 2)


def _middle_value(
    bkd: Backend[Array], nstates: int, node: int
) -> StateFunctional[Array, Array]:
    return StateFunctional.from_terms(
        bkd, 1, state_terms=(ObservedValue(StateEntries(bkd, nstates, [node])),)
    )


class TestBuilder:
    """``collocation_steady_function``'s arguments and refusals, on the
    ``dirichlet`` problem unless stated."""

    def test_configured_system_is_frozen(self, bkd: Backend[Array]) -> None:
        problem = _dirichlet(bkd)
        configured = ParameterizedCollocationSystem(
            problem.system, problem.parameterization, None
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            configured.system = problem.system  # type: ignore[misc]

    def test_overrides_reach_the_function(self, bkd: Backend[Array]) -> None:
        """Columns by tangents, a nonzero initial guess, and Newton without
        a line search give the default function's values and Jacobian to
        1e-12."""
        problem = _dirichlet(bkd)
        default = problem.function()
        parameters = ParameterState(problem.parameterization)
        overridden = collocation_steady_function(
            problem.system,
            parameters,
            _middle_value(bkd, problem.nstates, problem.node),
            init_state=bkd.ones((problem.nstates, 1)),
            setups=RecomputeAlways(BackendDenseLU(bkd)),
            nonlinear=Newton(atol=1e-12, rtol=0.0, maxiters=10),
            jacobian=TangentColumns(),
        )
        assert "TangentColumns" in repr(overridden)
        _close(bkd, overridden(problem.param), default(problem.param), 1e-12)
        new, old = overridden.derivatives().jacobian, default.derivatives().jacobian
        assert new is not None and old is not None
        _close(bkd, new(problem.param), old(problem.param), 1e-12)

    def test_refuses_a_wrong_initial_guess(self, bkd: Backend[Array]) -> None:
        problem = _dirichlet(bkd)
        with pytest.raises(ValueError, match="init_state"):
            collocation_steady_function(
                problem.system,
                ParameterState(problem.parameterization),
                _middle_value(bkd, problem.nstates, problem.node),
                init_state=bkd.zeros((problem.nstates,)),
            )

    def test_time_dependent_forcing_needs_the_time(self, bkd: Backend[Array]) -> None:
        """Omitted, the time is refused, naming the physics; given, the
        state is the one at that time, so later times differ."""
        mesh = TransformedMesh1D(20, bkd, AffineTransform1D((-1.0, 1.0), bkd))
        supplier: TimeDependent[Array] = TimeDependent(_growing)
        physics = AdvectionDiffusionReaction(
            ChebyshevBasis1D(mesh, bkd),
            bkd,
            diffusion=2.0,
            forcing=NodalSupplier(supplier, mesh.points()),
        )
        left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
        essential = DirichletRows(
            bkd, 20, 1, [DirichletPart(left, 0.0), DirichletPart(right, 0.0)]
        )
        system = CollocationSystem(physics, essential)
        parameterization = create_diffusion_parameterization(
            physics,
            bkd,
            _diffusion_map(bkd, 2.0, [bkd.ones((20,)), mesh.points()[0]]),
        )
        functional = _middle_value(bkd, 20, 10)
        param = bkd.reshape(bkd.asarray([0.3, -0.2]), (2, 1))
        with pytest.raises(ValueError, match="AdvectionDiffusionReaction"):
            collocation_steady_function(
                system, ParameterState(parameterization), functional
            )
        early, late = (
            collocation_steady_function(
                system, ParameterState(parameterization), functional, time=time
            )(param)
            for time in (0.0, 1.0)
        )
        # The forcing doubles, and the problem is linear.
        _close(bkd, late, 2.0 * early, 1e-10)


class _EvaluationOnly(Generic[Array]):
    """A parameterization writing what ``inner`` writes but declaring no
    derivatives."""

    def __init__(self, inner: ParameterizationProtocol[Array]) -> None:
        self._inner = inner

    def nparams(self) -> int:
        return self._inner.nparams()

    def targets(self) -> Tuple[object, ...]:
        return self._inner.targets()

    def owned_coefficients(self) -> Tuple[str, ...]:
        return self._inner.owned_coefficients()

    def apply(self, params_1d: Array) -> None:
        self._inner.apply(params_1d)

    def param_derivatives(self) -> ParamDerivatives[Array]:
        return ParamDerivatives.none()


class TestConsumers:
    """What the function's bundle gives a caller, on ``dirichlet``."""

    def test_no_jacobian_without_parameter_derivatives(
        self, bkd: Backend[Array]
    ) -> None:
        """Values equal the differentiable parameterization's; the bundle
        has neither a Jacobian nor an HVP."""
        problem = _dirichlet(bkd)
        function = collocation_steady_function(
            problem.system,
            ParameterState(_EvaluationOnly(problem.parameterization)),
            _middle_value(bkd, problem.nstates, problem.node),
        )
        derivatives = function.derivatives()
        assert derivatives.jacobian is None and derivatives.hvp is None
        samples = bkd.hstack([problem.param, 0.5 * problem.param])
        _close(bkd, function(samples), problem.function()(samples), 1e-12)

    def test_inside_inexact_wrapper(self, numpy_bkd: Backend[Array]) -> None:
        """The mean over three values of p_0 at the design p_1 = 0.1, by
        the OUU ``InexactWrapper``, equals the weighted values computed
        directly."""
        from pyapprox.optimization.minimize.inexact.fixed import (
            FixedSampleStrategy,
        )
        from pyapprox.optimization.minimize.inexact.wrapper import InexactWrapper
        from pyapprox.risk import SampleAverageMean

        bkd = numpy_bkd
        function = _dirichlet(bkd).function()
        random = bkd.asarray([[-0.2, 0.0, 0.2]])
        weights = bkd.asarray([0.25, 0.5, 0.25])
        wrapper = InexactWrapper(
            model=function,
            stat=SampleAverageMean(bkd),
            strategy=FixedSampleStrategy(random, weights, bkd),
            design_indices=[1],
            bkd=bkd,
        )
        value = wrapper(bkd.asarray([[0.1]]))
        direct = function(bkd.vstack([random, bkd.full((1, 3), 0.1)]))
        _close(bkd, value, bkd.sum(direct * weights[None, :], axis=1)[:, None], 1e-12)
