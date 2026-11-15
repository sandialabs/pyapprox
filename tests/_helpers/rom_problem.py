"""The problem bundle the physics-agnostic projection-ROM tests run on.

Separate from the factories in ``tests/_helpers/rom_problems.py``, which
need skfem, so that factories built without an assembly library (the
mock element-wise producer in ``tests/_helpers/mock_elementwise.py``)
run the ROM tests without it.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.elementwise import ElementwiseOperatorProtocol
from pyapprox.pde.galerkin.protocols.discretization import (
    SteadyDiscretizationProtocol,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.pde.rom import (
    ComposedParameterizedResidualProtocol,
    LiftProtocol,
    SnapshotSet,
    homogenize,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import InnerProductProtocol
from pyapprox.util.linalg.operators import FormatProtocol, InnerProductSpaceProtocol
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.adjoint_checks import NumpyArray

F = TypeVar("F", bound=FormatProtocol)


def steady_snapshots(
    steady_state: Callable[[NumpyArray], NumpyArray],
    samples: NumpyArray,
    bkd: NumpyBkd,
) -> SnapshotSet[NumpyArray]:
    """The steady state at each column of ``samples`` (nparams,
    nsamples), as snapshots."""
    states = [steady_state(samples[:, ii : ii + 1]) for ii in range(samples.shape[1])]
    return SnapshotSet(bkd.hstack(states), bkd.copy(samples))


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


class FixedValue(Generic[Array]):
    """An essential condition fixing one DOF to a constant."""

    def __init__(self, dof: int, value: float, bkd: Backend[Array]) -> None:
        self._dofs = bkd.asarray([dof], dtype=bkd.int64_dtype())
        self._values = bkd.asarray([value])
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def constrained_dofs(self) -> Array:
        return self._dofs

    def constrained_values(self, time: float) -> Array:
        return self._values

    def is_time_invariant(self) -> bool:
        return True

    def constrained_values_derivative(self, order: int) -> Callable[[float], Array]:
        zeros = self._bkd.zeros((1,))
        return lambda time: zeros


class SteadyStateAt(Generic[F]):
    """The steady state at a parameter column, on any stack: the steady
    residual of ``discretization`` solved by ``SteadySystem`` with the
    injected ``setups``, by Newton (``atol``, no relative tolerance) from
    zero, so the result does not depend on the order of the samples.

    Called with a parameter column (nparams, 1), returns the state
    (nstates, 1). ``setups`` is cleared after every parameter change.
    """

    def __init__(
        self,
        discretization: SteadyDiscretizationProtocol[NumpyArray, NumpyArray, F],
        parameterization: ParameterizationProtocol[NumpyArray],
        setups: SetupStrategyProtocol[SteadyRequest[NumpyArray, F], NumpyArray, F],
        bkd: NumpyBkd,
        atol: float,
    ) -> None:
        self._parameters = ParameterState(parameterization)
        space = discretization.space()
        if not isinstance(space, InnerProductSpaceProtocol):
            raise TypeError("the discretization's space has no inner product")
        self._system: SteadySystem[NumpyArray, NumpyArray, F] = SteadySystem(
            galerkin_steady_residual(discretization, self._parameters), space, setups
        )
        self._setups = setups
        self._newton: Newton[NumpyArray, NumpyArray] = Newton(
            atol=atol, rtol=0.0, maxiters=50
        )
        self._zeros = bkd.zeros((discretization.nstates(), 1))

    def __call__(self, param: NumpyArray) -> NumpyArray:
        self._parameters.set_param(param[:, 0])
        self._setups.clear()
        point = self._system.solve(self._newton, self._zeros)
        if not point.converged():
            raise RuntimeError(f"the steady solve did not converge: {point.info()}")
        return point.state()
