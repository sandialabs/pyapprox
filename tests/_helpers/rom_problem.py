"""The problem bundle the physics-agnostic projection-ROM tests run on.

Separate from the factories in ``tests/_helpers/rom_problems.py``, which
need skfem, so that factories built without an assembly library (the
mock element-wise producer in ``tests/_helpers/mock_elementwise.py``)
run the ROM tests without it.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    SteadyStateEquationProtocol,
)
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.elementwise import ElementwiseOperatorProtocol
from pyapprox.pde.rom import (
    ComposedParameterizedResidualProtocol,
    LiftProtocol,
    SnapshotSet,
    collect_steady_snapshots,
    homogenize,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.inner_product import InnerProductProtocol
from tests._helpers.adjoint_checks import NumpyArray


@dataclass(frozen=True)
class RomProblem:
    """Everything a physics-agnostic ROM test needs, and what to expect.

    ``affine``: the residual is affine in the state and the parameters,
    so the affine fit must succeed and be exact. ``coercive``: Galerkin
    quasi-optimality holds, so the ROM error is a bounded multiple of the
    best approximation error. ``elementwise``: the interior operator
    restricted to element subsets, for hyper-reduction; None when the
    discretization does not supply one.
    """

    name: str
    bkd: NumpyBkd
    residual: ComposedParameterizedResidualProtocol[NumpyArray]
    state_equation: SteadyStateEquationProtocol[NumpyArray]
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
        return collect_steady_snapshots(
            self.state_equation, samples, self.bkd.zeros((self.nstates(), 1))
        )

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
