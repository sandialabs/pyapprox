"""The problem bundle the physics-agnostic projection-ROM tests run on.

Separate from the factories in ``tests/_helpers/rom_problems.py``, which
need skfem, so that factories built without an assembly library (the
mock element-wise producer in ``tests/_helpers/mock_elementwise.py``)
run the ROM tests without it.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import (
    PinnedValuesParamDerivatives,
    RowPin,
)
from pyapprox.implicitfunction.residual_derivatives import (
    ResidualDerivatives,
    ResidualJacobian,
)
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.state_equations.protocols import (
    SteadyStateEquationProtocol,
)
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.ode.protocols.spatial_operator import SpatialOperatorProtocol
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.elementwise import ElementwiseOperatorProtocol
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.pde.rom import (
    ComposedParameterizedResidualProtocol,
    LiftProtocol,
    SnapshotSet,
    homogenize,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    BackendRowSet,
    DenseFormat,
)
from pyapprox.util.linalg.inner_product import InnerProductProtocol
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    LinearizedOperator,
    RowSetProtocol,
)
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.adjoint_checks import NumpyArray


def steady_snapshots(
    steady_state: Callable[[NumpyArray], NumpyArray],
    samples: NumpyArray,
    bkd: NumpyBkd,
) -> SnapshotSet[NumpyArray]:
    """The steady state at each column of ``samples`` (nparams,
    nsamples), as snapshots."""
    states = [steady_state(samples[:, ii : ii + 1]) for ii in range(samples.shape[1])]
    return SnapshotSet(bkd.hstack(states), bkd.copy(samples))


class SolvedFromZero:
    """The steady state at a parameter column, by a state equation's solve
    from zero, so it does not depend on the order of the samples."""

    def __init__(
        self,
        state_equation: SteadyStateEquationProtocol[NumpyArray],
        bkd: NumpyBkd,
    ) -> None:
        self._state_equation = state_equation
        self._zeros = bkd.zeros((state_equation.nstates(), 1))

    def __call__(self, param: NumpyArray) -> NumpyArray:
        return self._state_equation.solve(self._zeros, param)


@dataclass(frozen=True)
class RomProblem:
    """Everything a physics-agnostic ROM test needs, and what to expect.

    ``affine``: the residual is affine in the state and the parameters,
    so the affine fit must succeed and be exact. ``coercive``: Galerkin
    quasi-optimality holds, so the ROM error is a bounded multiple of the
    best approximation error. ``elementwise``: the interior operator
    restricted to element subsets, for hyper-reduction; None when the
    discretization does not supply one. ``steady_state``: the full-order
    steady state at a parameter column (nparams, 1), shape (nstates, 1),
    computed however the factory chooses.
    """

    name: str
    bkd: NumpyBkd
    residual: ComposedParameterizedResidualProtocol[NumpyArray]
    steady_state: Callable[[NumpyArray], NumpyArray]
    constraint_set: ConstraintSetProtocol[NumpyArray]
    lift: LiftProtocol[NumpyArray]
    metric: InnerProductProtocol[NumpyArray]
    train: NumpyArray
    held_out: NumpyArray
    reference_params: NumpyArray
    affine: bool
    coercive: bool
    elementwise: Optional[ElementwiseOperatorProtocol[NumpyArray]] = None

    def nstates(self) -> int:
        return self.residual.nstates()

    def solve(self, samples: NumpyArray) -> SnapshotSet[NumpyArray]:
        """Full-order steady solutions at each column of ``samples``."""
        return steady_snapshots(self.steady_state, samples, self.bkd)

    def homogenized(self, samples: NumpyArray) -> SnapshotSet[NumpyArray]:
        """Full-order solutions with the lift removed."""
        return homogenize(self.solve(samples), self.lift, self.constraint_set)


class FixedValue:
    """An essential condition fixing one DOF to a constant."""

    def __init__(self, dof: int, value: float, bkd: NumpyBkd) -> None:
        self._dofs = bkd.asarray([dof], dtype=bkd.int64_dtype())
        self._values = bkd.asarray([value])
        self._bkd = bkd

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def constrained_dofs(self) -> NumpyArray:
        return self._dofs

    def constrained_values(self, time: float) -> NumpyArray:
        return self._values

    def is_time_invariant(self) -> bool:
        return True

    def constrained_values_derivative(
        self, order: int
    ) -> Callable[[float], NumpyArray]:
        zeros = self._bkd.zeros((1,))
        return lambda time: zeros


_Dense = DenseFormat[NumpyArray]


class _DenseOperatorResidual:
    r""":math:`F(u; p)` of a spatial operator with a dense Jacobian, at
    t = 0, as a time-free residual: value and state Jacobian only."""

    def __init__(
        self,
        operator: SpatialOperatorProtocol[NumpyArray],
        parameters: ParameterState[NumpyArray],
        bkd: NumpyBkd,
    ) -> None:
        self._operator, self._parameters, self._bkd = operator, parameters, bkd
        states, _ = backend_spaces(bkd, operator.nstates(), parameters.nparams())
        self._states = states.descriptor()
        self._bundle: ResidualDerivatives[NumpyArray, _Dense] = (
            ResidualDerivatives.state_only(ResidualJacobian.general(self._jacobian))
        )

    def __call__(self, state: NumpyArray) -> NumpyArray:
        return self._operator.spatial_residual(state[:, 0], 0.0)[:, None]

    def derivatives(self) -> ResidualDerivatives[NumpyArray, _Dense]:
        return self._bundle

    def parameters_version(self) -> int:
        return self._parameters.parameters_version()

    def _jacobian(
        self, state: NumpyArray, point: LinearizationPoint
    ) -> LinearizedOperator[NumpyArray, _Dense]:
        matrix = self._operator.spatial_jacobian(state[:, 0], 0.0)
        return LinearizedOperator(
            BackendDenseOperator(matrix, self._states, self._bkd), point
        )


class _ConstrainedValues:
    """A constraint set's rows and values at t = 0, independent of the
    parameters."""

    def __init__(
        self,
        constraint_set: ConstraintSetProtocol[NumpyArray],
        rows: RowSetProtocol[NumpyArray],
        nstates: int,
        bkd: NumpyBkd,
    ) -> None:
        self._rows = rows
        values = bkd.zeros((nstates, 1))
        values[constraint_set.dofs(), 0] = constraint_set.values(0.0)
        self._values = values

    def rows(self) -> RowSetProtocol[NumpyArray]:
        return self._rows

    def values(self) -> NumpyArray:
        return self._values

    def parameter_derivatives(self) -> PinnedValuesParamDerivatives[NumpyArray]:
        return PinnedValuesParamDerivatives.independent()


class DenseSteadyState:
    """The steady state of a spatial operator with a dense Jacobian and
    time-invariant essential constraints, for problems built without an
    assembly library: :math:`F(u; p)` with the constrained rows pinned,
    solved by Newton (``atol``, no relative tolerance) with dense LU from
    zero.

    Called with a parameter column (nparams, 1), returns the state
    (nstates, 1).
    """

    def __init__(
        self,
        operator: SpatialOperatorProtocol[NumpyArray],
        parameterization: ParameterizationProtocol[NumpyArray],
        constraint_set: ConstraintSetProtocol[NumpyArray],
        bkd: NumpyBkd,
        atol: float,
    ) -> None:
        if not constraint_set.is_time_invariant():
            raise ValueError("the constraints must be time-invariant")
        n = operator.nstates()
        self._parameters = ParameterState(parameterization)
        states, _ = backend_spaces(bkd, n, parameterization.nparams())
        rows: BackendRowSet[NumpyArray] = BackendRowSet(
            states.descriptor(),
            [int(dof) for dof in bkd.to_numpy(constraint_set.dofs())],
            bkd,
        )
        pinned = PinnedResidual(
            _DenseOperatorResidual(operator, self._parameters, bkd),
            RowPin(_ConstrainedValues(constraint_set, rows, n, bkd)),
            states,
        )
        self._system: SteadySystem[NumpyArray, NumpyArray, _Dense] = SteadySystem(
            pinned, states, RecomputeAlways(BackendDenseLU(bkd))
        )
        self._newton: Newton[NumpyArray, NumpyArray] = Newton(
            atol=atol, rtol=0.0, maxiters=50
        )
        self._zeros = bkd.zeros((n, 1))

    def __call__(self, param: NumpyArray) -> NumpyArray:
        self._parameters.set_param(param[:, 0])
        point = self._system.solve(self._newton, self._zeros)
        if not point.converged():
            raise RuntimeError(f"the steady solve did not converge: {point.info()}")
        return point.state()
