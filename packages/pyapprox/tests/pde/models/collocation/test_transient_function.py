r"""Transient collocation functions on the new path.

``collocation_transient_system`` through ``TransientImplicitMap`` and
``ImplicitMapFunction``: the ``FunctionProtocol`` face of a transient
collocation solve. 1D diffusion on 15 Chebyshev nodes, D = 2 + p_0 + p_1 x,
initial state sin(pi x), five steps of 0.02; boundary sets zero Dirichlet
at both ends, gradient Robin at both ends (alpha 1 left, 2 right, beta 1),
Robin left with Dirichlet right, and Dirichlet 0.5 left (the initial state
violates it) with zero right. Methods BE and CN, under the step-end pin,
the pin at every block and the step-end pin with differentiated essential
rows.

- the value equals a direct solve's final state; batches equal single
  samples; ``nvars``, ``nqoi`` and the ``FunctionProtocol`` check;
- the all-states Jacobian and a scalar QoI's Jacobian and HVP pass
  central differences, the HVP symmetric to 1e-12;
- with more parameters than states, the all-states Jacobian by adjoint
  rows equals the tangent columns to 1e-9;
- autodiff (``bkd.jacobian`` on torch) through the solve equals the
  Jacobian to 1e-6;
- the function is consumable by ``InexactWrapper``;
- a 6 x 6 polar mesh with zero Dirichlet sides: adjoint, tangent and
  central differences agree.
"""

import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest

from pyapprox.implicitfunction.implicit_map import (
    AdjointRows,
    ArrayParameters,
    ImplicitMapFunction,
    TangentColumns,
)
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.ode.differentiated_rows import (
    CoupledRows,
    RowRatesTreatmentProtocol,
)
from pyapprox.ode.integration import Integrator
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.pinned_rows import AtEveryBlock, AtStepEnd, PinPlacementProtocol
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import ButcherTableau, backward_euler, crank_nicolson
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TransientImplicitMap,
    TransientPoint,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
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
    GradientNormalOperator,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D, TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.transient_system import (
    CollocationTransientSystem,
    collocation_transient_system,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton

_NPTS = 15
_SAMPLE = [[0.3], [-0.2]]
_STEPS = FixedStep(0.02, (0.0, 0.1))

_Function = ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]
_Strategy = Optional[AdjointRows | TangentColumns]

_TABLEAUX: dict[str, Callable[[], ButcherTableau]] = {
    "backward_euler": backward_euler,
    "crank_nicolson": crank_nicolson,
}
_TREATMENTS: dict[
    str, Tuple[Callable[[], PinPlacementProtocol], Optional[RowRatesTreatmentProtocol]]
] = {
    "step_end": (AtStepEnd, None),
    "every_block": (AtEveryBlock, None),
    "differentiated": (AtStepEnd, CoupledRows()),
}


def _robin(
    bkd: Backend[Array], mesh: TransformedMesh1D[Array], side: int, alpha: float
) -> RobinPart[Array]:
    indices = mesh.boundary_indices(side)
    normal = GradientNormalOperator(
        bkd,
        indices,
        mesh.boundary_normals(side),
        [ChebyshevBasis1D(mesh, bkd).derivative_matrix(1, 0)],
    )
    return RobinPart(indices, normal, alpha, 1.0, 0.0)


def _system(bkd: Backend[Array], ends: str) -> CollocationSystem[Array]:
    """Diffusion with D = 2 on the boundary set ``ends``: ``dirichlet``,
    ``robin``, ``mixed`` or ``nonzero`` (Dirichlet 0.5 left)."""
    mesh = TransformedMesh1D(_NPTS, bkd)
    physics = AdvectionDiffusionReaction(
        ChebyshevBasis1D(mesh, bkd), bkd, diffusion=2.0
    )
    layout = StateLayout.uniform(_NPTS, 1)
    left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
    dirichlet = {
        "dirichlet": [DirichletPart(left, 0.0), DirichletPart(right, 0.0)],
        "robin": [],
        "mixed": [DirichletPart(right, 0.0)],
        "nonzero": [DirichletPart(left, 0.5), DirichletPart(right, 0.0)],
    }[ends]
    robin = {
        "dirichlet": [],
        "robin": [_robin(bkd, mesh, 0, 1.0), _robin(bkd, mesh, 1, 2.0)],
        "mixed": [_robin(bkd, mesh, 0, 1.0)],
        "nonzero": [],
    }[ends]
    closures: List[ClosureRowsProtocol[Array]] = (
        [RobinRows(bkd, layout, robin)] if robin else []
    )
    essential = DirichletRows(bkd, layout, dirichlet) if dirichlet else None
    return CollocationSystem(physics, essential, closures)


def _parameters(
    bkd: Backend[Array], system: CollocationSystem[Array], modes: List[Array]
) -> ParameterState[Array]:
    physics = system.physics()
    if not isinstance(physics, AdvectionDiffusionReaction):
        raise TypeError("the test systems are ADR")
    return ParameterState(
        create_diffusion_parameterization(physics, bkd, BasisExpansion(bkd, 2.0, modes))
    )


def _nodes(bkd: Backend[Array]) -> Array:
    return TransformedMesh1D(_NPTS, bkd).points()[0]


def _transient(
    bkd: Backend[Array],
    ends: str = "dirichlet",
    treatment: str = "step_end",
    modes: Optional[List[Array]] = None,
) -> CollocationTransientSystem[Array]:
    system = _system(bkd, ends)
    placement, rows = _TREATMENTS[treatment]
    if rows is not None and system.essential() is None:
        rows = None
    chosen = [bkd.ones((_NPTS,)), _nodes(bkd)] if modes is None else modes
    return collocation_transient_system(
        system, placement(), rows, _parameters(bkd, system, chosen)
    )


def _function(
    bkd: Backend[Array],
    transient: CollocationTransientSystem[Array],
    entries: List[int],
    method: str = "backward_euler",
    strategy: _Strategy = None,
    steps: FixedStep = _STEPS,
    init: Optional[Array] = None,
) -> _Function[Array]:
    nstates = transient.collocation.nstates()
    newton: Newton[Array, Array] = Newton(atol=1e-12, rtol=0.0, maxiters=20)
    functional = FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd,
            len(entries),
            state_terms=(EndpointTerm(StateEntries(bkd, nstates, entries)),),
        )
    )
    start = (
        bkd.reshape(bkd.sin(math.pi * _nodes(bkd)), (-1, 1)) if init is None else init
    )
    implicit = TransientImplicitMap(
        transient,
        transient,
        RungeKutta(_TABLEAUX[method](), newton),
        RecomputeAlways(transient.solved.solver),
        steps,
        start,
        functional,
    )
    if strategy is None:
        return ImplicitMapFunction(implicit, ArrayParameters(bkd))
    return ImplicitMapFunction(implicit, ArrayParameters(bkd), strategy)


def _all_states(
    bkd: Backend[Array],
    transient: CollocationTransientSystem[Array],
    strategy: _Strategy = None,
) -> _Function[Array]:
    return _function(bkd, transient, list(range(_NPTS)), strategy=strategy)


class TestTheFunction:
    def test_value_equals_a_direct_solve(self, bkd: Backend[Array]) -> None:
        transient = _transient(bkd)
        sample = bkd.asarray(_SAMPLE)
        value = _all_states(bkd, transient)(sample)
        transient.set_param(sample)
        newton: Newton[Array, Array] = Newton(atol=1e-12, rtol=0.0, maxiters=20)
        stepper = RungeKutta(backward_euler(), newton).stepper(
            transient.solved,
            RecomputeAlways(transient.solved.solver),
            transient.row_pinning,
        )
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        start = bkd.reshape(bkd.sin(math.pi * _nodes(bkd)), (-1, 1))
        Integrator(stepper).solve(transient.initial_state(start, 0.0), _STEPS, store)
        bkd.assert_allclose(
            value, store.state(store.ntimes() - 1), rtol=1e-10, atol=1e-14
        )

    def test_batches_equal_single_samples(self, bkd: Backend[Array]) -> None:
        function = _all_states(bkd, _transient(bkd))
        samples = bkd.asarray(np.random.default_rng(42).uniform(-0.1, 0.3, (2, 3)))
        values = function(samples)
        assert tuple(values.shape) == (_NPTS, 3)
        for k in range(3):
            bkd.assert_allclose(
                values[:, k : k + 1], function(samples[:, k : k + 1]), rtol=1e-10
            )

    def test_dimensions_and_protocol(self, bkd: Backend[Array]) -> None:
        function = _all_states(bkd, _transient(bkd))
        assert isinstance(function, FunctionProtocol)
        assert function.nvars() == 2 and function.nqoi() == _NPTS
        derivatives = function.derivatives()
        assert derivatives.jacobian is not None


_FD_CASES = [
    (ends, treatment, method)
    for ends in ("dirichlet", "robin", "mixed", "nonzero")
    for treatment in _TREATMENTS
    for method in _TABLEAUX
    # Robin at both ends has no essential rows to differentiate
    if not (ends == "robin" and treatment == "differentiated")
]


@pytest.mark.parametrize("ends, treatment, method", _FD_CASES)
class TestCentralDifferences:
    def test_all_states_jacobian(
        self, numpy_bkd: Backend[Array], ends: str, treatment: str, method: str
    ) -> None:
        bkd = numpy_bkd
        transient = _transient(bkd, ends, treatment)
        function = _function(bkd, transient, list(range(_NPTS)), method)
        checker = DerivativeChecker(function)
        errors = checker.check_derivatives(bkd.asarray(_SAMPLE), relative=True)
        assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-5

    def test_scalar_jacobian_and_hvp(
        self, numpy_bkd: Backend[Array], ends: str, treatment: str, method: str
    ) -> None:
        """Central differences along a fixed direction: best errors below
        1e-6, ratios below 1e-6 (forward differences floor the HVP near
        5e-6 here, the first-order truncation meeting the solve's noise)."""
        bkd = numpy_bkd
        transient = _transient(bkd, ends, treatment)
        function = _function(bkd, transient, [_NPTS // 2], method)
        sample = bkd.asarray(_SAMPLE)
        checker = DerivativeChecker(function)
        errors = checker.check_derivatives(
            sample, direction=bkd.asarray([[0.6], [-0.8]]), relative=True, central=True
        )
        for error in errors:
            assert bkd.to_float(bkd.min(error)) <= 1e-6
            assert bkd.to_float(checker.error_ratio(error)) <= 1e-6
        hvp = function.derivatives().hvp
        assert hvp is not None
        v, u = bkd.asarray([[0.7], [0.4]]), bkd.asarray([[-0.5], [0.9]])
        bkd.assert_allclose(
            bkd.sum(hvp(sample, v) * u), bkd.sum(hvp(sample, u) * v), rtol=1e-12
        )


def test_vector_adjoint_equals_tangent(bkd: Backend[Array]) -> None:
    """Ten parameters for 15 states, the modes 0.1 cos(k pi x): the
    all-states Jacobian by adjoint rows equals the tangent columns."""
    nodes = _nodes(bkd)
    modes = [bkd.ones((_NPTS,))] + [
        0.1 * bkd.cos(k * math.pi * nodes) for k in range(1, 10)
    ]
    transient = _transient(bkd, modes=modes)
    sample = bkd.asarray(np.random.default_rng(61).normal(0.0, 0.05, (10, 1)))
    adjoint = _all_states(bkd, transient, AdjointRows()).derivatives().jacobian
    tangent = _all_states(bkd, transient, TangentColumns()).derivatives().jacobian
    assert adjoint is not None and tangent is not None
    bkd.assert_allclose(adjoint(sample), tangent(sample), rtol=1e-9, atol=1e-12)


def test_autodiff_through_the_solve_equals_the_jacobian(
    torch_bkd: AutodiffBackend[Array],
) -> None:
    """The graph survives the solve: autodiff of the value equals the
    sweeps' Jacobian."""
    bkd = torch_bkd
    function = _all_states(bkd, _transient(bkd))
    sample = bkd.asarray([0.3, -0.2])

    def value(param: Array) -> Array:
        return function(param[:, None])[:, 0]

    autograd = bkd.jacobian(value, sample)
    jacobian = function.derivatives().jacobian
    assert jacobian is not None
    bkd.assert_allclose(jacobian(sample[:, None]), autograd, rtol=1e-6, atol=1e-12)


def test_inexact_wrapper_consumes_it(numpy_bkd: Backend[Array]) -> None:
    from pyapprox.optimization.minimize.inexact.fixed import FixedSampleStrategy
    from pyapprox.optimization.minimize.inexact.wrapper import InexactWrapper
    from pyapprox.risk import SampleAverageMean

    bkd = numpy_bkd
    function = _function(bkd, _transient(bkd), [_NPTS // 2])
    wrapper = InexactWrapper(
        model=function,
        stat=SampleAverageMean(bkd),
        strategy=FixedSampleStrategy(
            bkd.asarray([[-0.2, 0.0, 0.2]]), bkd.asarray([0.25, 0.5, 0.25]), bkd
        ),
        design_indices=[1],
        bkd=bkd,
    )
    value = wrapper(bkd.asarray([[0.1]]))
    assert value.shape[0] == 1
    assert math.isfinite(bkd.to_float(value[0, 0]))


def test_polar_adjoint_tangent_and_differences_agree(
    numpy_bkd: Backend[Array],
) -> None:
    """Diffusion 2 + p_0 + 0.3 p_1 sin(x + y) on a 6 x 6 polar mesh, zero
    Dirichlet sides, four BE steps of 0.0125: the adjoint gradient equals
    the tangent row to 1e-9 and passes central differences (old model's
    measured V-bottom 1.15e-7)."""
    bkd = numpy_bkd
    n1d = 6
    transform = PolarTransform((1.0, 2.0), (-math.pi / 2, math.pi / 2), bkd)
    mesh = TransformedMesh2D(n1d, n1d, bkd, transform)
    physics = AdvectionDiffusionReaction(ChebyshevBasis2D(mesh, bkd), bkd, 2.0)
    points = mesh.points()
    npts = points.shape[1]
    essential = DirichletRows(
        bkd,
        StateLayout.uniform(npts, 1),
        [DirichletPart(mesh.boundary_indices(side), 0.0) for side in range(4)],
    )
    system: CollocationSystem[Array] = CollocationSystem(physics, essential)
    modes = [bkd.ones((npts,)), 0.3 * bkd.sin(points[0] + points[1])]
    parameters = ParameterState(
        create_diffusion_parameterization(physics, bkd, BasisExpansion(bkd, 2.0, modes))
    )
    transient = collocation_transient_system(system, AtStepEnd(), None, parameters)
    observed = npts // 2 + n1d // 2
    init = bkd.reshape(
        bkd.sin(math.pi * (points[0] ** 2 + points[1] ** 2) / 4.0), (-1, 1)
    )
    steps = FixedStep(0.0125, (0.0, 0.05))
    scalar = _function(
        bkd, transient, [observed], strategy=AdjointRows(), steps=steps, init=init
    )
    final = _function(
        bkd,
        transient,
        list(range(npts)),
        strategy=TangentColumns(),
        steps=steps,
        init=init,
    )
    sample = bkd.asarray([[0.2], [-0.1]])
    adjoint, tangent = scalar.derivatives().jacobian, final.derivatives().jacobian
    assert adjoint is not None and tangent is not None
    bkd.assert_allclose(
        adjoint(sample), tangent(sample)[observed : observed + 1], rtol=1e-9
    )
    checker = DerivativeChecker(scalar)
    errors = checker.check_derivatives(sample, verbosity=0)
    assert bkd.to_float(bkd.min(errors[0])) <= 5e-7
    assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-5
