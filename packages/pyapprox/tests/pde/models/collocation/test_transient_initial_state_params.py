r"""Gradients when a parameter sets the initial state (collocation).

With Robin rows at both ends there are no essential rows, so every entry of
:math:`y_0 = u_0(p)` depends on :math:`p`; under Crank--Nicolson
:math:`F(y_0)` carries that dependence into the first step. The parameters
here set the diffusion :math:`2 + p_0 + p_1 x` and the initial state
:math:`\sin(\pi x) + p_0 (2 + x)`, on 15 Chebyshev nodes, five steps of
0.02. The shared map reads :math:`dy_0/dp` from the residual's bundle, which
takes it from the parameterization's ``initial_param_jacobian``.

The check needs no derivation: the gradient by the adjoint sweep is compared
with central differences of the function rebuilt with :math:`u_0(p)` for
every perturbation, under BE, CN and the implicit midpoint rule, with the
step-end pin and the pin at every block.
"""

import dataclasses
import math
from typing import Callable, Generic, Tuple

import pytest

from pyapprox.implicitfunction.implicit_map import (
    AdjointRows,
    ArrayParameters,
    ImplicitMapFunction,
)
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.pinned_rows import AtEveryBlock, AtStepEnd, PinPlacementProtocol
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TransientImplicitMap,
    TransientPoint,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary.closure_rows import RobinPart, RobinRows
from pyapprox.pde.collocation.boundary.normal_operators import (
    GradientNormalOperator,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.transient_system import (
    collocation_transient_system,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton

_NPTS = 15

_TABLEAUX: dict[str, Callable[[], ButcherTableau]] = {
    "backward_euler": backward_euler,
    "crank_nicolson": crank_nicolson,
    "implicit_midpoint": implicit_midpoint,
}
_PLACEMENTS: dict[str, Callable[[], PinPlacementProtocol]] = {
    "step_end": AtStepEnd,
    "every_block": AtEveryBlock,
}


class _InitialStateParameterization(Generic[Array]):
    """A parameterization whose parameters also set the initial state,
    ``u_0(p) = u_base + p_0 phi``, so ``dy_0/dp = [phi, 0]``."""

    def __init__(
        self, inner: ParameterizationProtocol[Array], ic_jacobian: Array
    ) -> None:
        self._inner = inner
        self._ic_jacobian = ic_jacobian

    def nparams(self) -> int:
        return self._inner.nparams()

    def targets(self) -> Tuple[object, ...]:
        return self._inner.targets()

    def owned_coefficients(self) -> Tuple[str, ...]:
        return self._inner.owned_coefficients()

    def apply(self, params_1d: Array) -> None:
        self._inner.apply(params_1d)

    def param_derivatives(self) -> ParamDerivatives[Array]:
        return dataclasses.replace(
            self._inner.param_derivatives(),
            initial_param_jacobian=self._initial_param_jacobian,
        )

    def _initial_param_jacobian(self, params_1d: Array) -> Array:
        return self._ic_jacobian


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


def _function(
    bkd: Backend[Array], method: str, placement: str, param: Array
) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
    """The endpoint value at the middle node, its initial state
    ``u_0(param)``."""
    mesh = TransformedMesh1D(_NPTS, bkd)
    nodes = mesh.points()[0]
    physics = AdvectionDiffusionReaction(
        ChebyshevBasis1D(mesh, bkd), bkd, diffusion=2.0
    )
    robin = RobinRows(
        bkd,
        StateLayout.uniform(_NPTS, 1),
        [_robin(bkd, mesh, 0, 1.0), _robin(bkd, mesh, 1, 2.0)],
    )
    system: CollocationSystem[Array] = CollocationSystem(physics, None, [robin])
    phi = 2.0 + nodes  # nonzero at both Robin nodes
    inner = create_diffusion_parameterization(
        physics, bkd, BasisExpansion(bkd, 2.0, [bkd.ones((_NPTS,)), nodes])
    )
    parameterization = _InitialStateParameterization(
        inner, bkd.stack([phi, bkd.zeros((_NPTS,))], axis=1)
    )
    transient = collocation_transient_system(
        system, _PLACEMENTS[placement](), None, ParameterState(parameterization)
    )
    init = bkd.reshape(bkd.sin(math.pi * nodes) + param[0, 0] * phi, (-1, 1))
    newton: Newton[Array, Array] = Newton(atol=1e-12, rtol=0.0, maxiters=20)
    functional = FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd,
            1,
            state_terms=(EndpointTerm(StateEntries(bkd, _NPTS, [_NPTS // 2])),),
        )
    )
    implicit = TransientImplicitMap(
        transient,
        transient,
        RungeKutta(_TABLEAUX[method](), newton),
        RecomputeAlways(transient.solved.solver),
        FixedStep(0.02, (0.0, 0.1)),
        init,
        functional,
    )
    return ImplicitMapFunction(implicit, ArrayParameters(bkd), AdjointRows())


_CASES = [
    (method, placement)
    for method in _TABLEAUX
    for placement in _PLACEMENTS
    # The implicit midpoint rule is not stiffly accurate: under the
    # step-end pin its one stage keeps the raw boundary rows, the
    # unconstrained operator's spurious growing modes are amplified, and
    # the trajectory diverges (1.6e12 after five steps).
    if not (method == "implicit_midpoint" and placement == "step_end")
]


@pytest.mark.parametrize("method, placement", _CASES)
def test_adjoint_gradient_matches_central_differences(
    bkd: Backend[Array], method: str, placement: str
) -> None:
    def value(samples: Array) -> Array:
        columns = [
            _function(bkd, method, placement, samples[:, k : k + 1])(
                samples[:, k : k + 1]
            )
            for k in range(samples.shape[1])
        ]
        return bkd.hstack(columns)

    def jacobian(sample: Array) -> Array:
        derivative = _function(bkd, method, placement, sample).derivatives().jacobian
        if derivative is None:
            raise AssertionError("the function has no Jacobian")
        return derivative(sample)

    wrapper = FunctionWithJacobianFromCallable(
        nqoi=1, nvars=2, fun=value, jacobian=jacobian, bkd=bkd
    )
    checker = DerivativeChecker(wrapper)
    errors = checker.check_derivatives(
        bkd.asarray([[0.3], [0.1]]),
        direction=bkd.asarray([[0.6], [-0.8]]),
        relative=True,
        central=True,
    )[0]
    assert bkd.to_float(bkd.min(errors)) <= 1e-6
    assert bkd.to_float(checker.error_ratio(errors)) <= 1e-6
