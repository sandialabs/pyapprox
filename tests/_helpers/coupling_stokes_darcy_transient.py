r"""Harness for transient Stokes beside transient Darcy, coupled through
the interface pressure in a space in time per window.

The problem is :mod:`tests._helpers.coupling_stokes_darcy`'s with time
derivatives: :math:`\partial_t u - \nu \Delta u + \nabla p = f`,
:math:`\nabla\cdot u = f_p` on the Stokes side, :math:`\partial_t p_D -
\nabla\cdot(K \nabla p_D) = f_D` on the Darcy side. The manufactured
solution is :data:`~tests._helpers.coupling_stokes_darcy.SMOOTH` times
:math:`\phi(t) = 1 + \sin(3t)/2`: the three interface conditions are
linear and homogeneous, so they hold at every time, and at t = 0 it is
the steady solution.

Each window's skeleton unknown is the interface pressure in cG(q) in
time (:class:`~pyapprox.pde.coupling.transient.StartContinuousLagrange`
on q fractions, tested against the Lagrange basis of :math:`P_{q-1}`),
one trace per unknown at every skeleton DOF
(:class:`~pyapprox.pde.coupling.spaces.ComponentSkeletonSpace`), read by
each side's steady transfer component by component
(:class:`~pyapprox.pde.coupling.ports.TimeComponentTransfer`). Stokes
receives it as a load (:func:`~pyapprox.pde.coupling.stacks.galerkin.
skfem_spacetime_loaded_response`) and returns its velocity tested in time;
Darcy pins it (:func:`~pyapprox.pde.coupling.stacks.galerkin.
skfem_spacetime_response`) and returns its flux tested in time. Each side
steps its own number of steps of its own implicit, stiffly accurate
method per window (Stokes is a saddle-point system: no block solves with
its singular mass alone).

The initial states are the steady coupled solve with the solution at
t = 0: the discrete Stokes projection of the initial data, divergence
consistent, so the index-2 system starts on its constraint.

NumPy only: skfem is. Importers must skip when skfem is not installed.
"""

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import ButcherTableau, alexander_sdirk3
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.coupling.ports import PortTransferProtocol, TimeComponentTransfer
from pyapprox.pde.coupling.responses import (
    PortQuantity,
    ResponsePointProtocol,
    TimeWindow,
)
from pyapprox.pde.coupling.skeleton import CoupledSubdomain, DenseSchur
from pyapprox.pde.coupling.spaces import ComponentSkeletonSpace
from pyapprox.pde.coupling.stacks.galerkin import (
    skfem_spacetime_loaded_response,
    skfem_spacetime_response,
)
from pyapprox.pde.coupling.transient import (
    LagrangeTestFunctions,
    StartContinuousLagrange,
    WindowedResponseProtocol,
)
from pyapprox.pde.coupling.transient_skeleton import TransientSkeletonSolver
from pyapprox.pde.galerkin.protocols.basis import ComponentDofsBasisProtocol
from pyapprox.pde.models.galerkin.verification import SkfemPoints, exact_dof_values
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling_collocation import Coefficients
from tests._helpers.coupling_mixed import skfem_adr_system
from tests._helpers.coupling_stokes_darcy import (
    SMOOTH,
    InterfacePressure,
    K,
    NormalVelocity,
    Solution,
    StokesDarcyStrip,
    stokes_side,
)

Np = NDArray[Any]
_BKD = NumpyBkd()
PHI = "(1 + sin(3*t)/2)"
TRANSIENT = Solution(
    (
        f"{PHI}*({SMOOTH.stokes[0]})",
        f"{PHI}*({SMOOTH.stokes[1]})",
        f"{PHI}*({SMOOTH.stokes[2]})",
    ),
    f"{PHI}*({SMOOTH.darcy})",
)

_SPACES: dict[int, tuple[StartContinuousLagrange, LagrangeTestFunctions]] = {
    1: (StartContinuousLagrange(), LagrangeTestFunctions((0.5,))),
    2: (StartContinuousLagrange((0.5, 1.0)), LagrangeTestFunctions((0.25, 0.75))),
}


@dataclass
class _Recorded:
    """A windowed response recording each accepted point's output: the
    protocol satisfied by delegation."""

    inner: WindowedResponseProtocol[Np]
    outputs: list[Np]

    def bkd(self) -> NumpyBkd:
        return _BKD

    def nport(self) -> int:
        return self.inner.nport()

    def receives(self) -> PortQuantity:
        return self.inner.receives()

    def returns(self) -> PortQuantity:
        return self.inner.returns()

    def nparams(self) -> int:
        return 0

    def nqoi(self) -> int:
        return 0

    def linearize(
        self, port_data: Np, params: Np, window: TimeWindow
    ) -> ResponsePointProtocol[Np]:
        return self.inner.linearize(port_data, params, window)

    def time(self) -> float:
        return self.inner.time()

    def start(self) -> None:
        self.inner.start()

    def accept(self, point: ResponsePointProtocol[Np]) -> None:
        self.inner.accept(point)
        self.outputs.append(point.output())


def _method(tableau: Callable[[], ButcherTableau]) -> RungeKutta[Np, Np]:
    return RungeKutta(tableau(), Newton(atol=1e-12, rtol=0.0, maxiters=20))


class TransientStokesDarcy:
    """One coupled transient run.

    Parameters
    ----------
    ncells, nskel
        Cells per direction on each side; skeleton nodes per face.
    q
        The trial space's degree in time (1 or 2).
    nwindows, final
        Equal windows over [0, ``final``].
    nsubsteps
        Steps per window, Stokes then Darcy.
    methods
        The tableaus, Stokes then Darcy; implicit and stiffly accurate.
    bjs, stokes_sign
        As :class:`~tests._helpers.coupling_stokes_darcy.StokesDarcyStrip`
        (the controls).
    """

    def __init__(
        self,
        ncells: int,
        nskel: int,
        q: int,
        nwindows: int,
        final: float,
        nsubsteps: Sequence[int] = (2, 3),
        methods: Sequence[Callable[[], ButcherTableau]] = (
            alexander_sdirk3,
            alexander_sdirk3,
        ),
        bjs: bool = True,
        stokes_sign: float = -1.0,
    ) -> None:
        bkd = _BKD
        steady = StokesDarcyStrip(ncells, nskel, bjs, stokes_sign, SMOOTH)
        lam0 = steady.solve_skeleton()
        stokes0, darcy0 = steady.states(lam0)
        self._final = final
        trial, test = _SPACES[q]
        space = ComponentSkeletonSpace(steady.space(), q)
        stokes = stokes_side(
            ncells, bjs, stokes_sign, TRANSIENT, steady.skeleton(0), transient=True
        )
        darcy = skfem_adr_system(
            Coefficients(K, None, None),
            TRANSIENT.darcy,
            steady.skeleton(1),
            ncells,
            storage=1.0,
        )
        self.stokes_sink: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
        self.darcy_sink: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
        stokes_response = skfem_spacetime_loaded_response(
            stokes.system,
            stokes.ports,
            _method(methods[0]),
            RecomputeAlways(ScipySparseLU()),
            stokes0,
            self.stokes_sink,
            trial,
            test,
            stokes.transfer.scatter(lam0),
            nsubsteps=nsubsteps[0],
            receives=InterfacePressure(),
            returns=NormalVelocity(),
        )
        darcy_response = skfem_spacetime_response(
            darcy.system,
            darcy.ports,
            _method(methods[1]),
            RecomputeAlways(ScipySparseLU()),
            darcy0,
            self.darcy_sink,
            trial,
            test,
            nsubsteps=nsubsteps[1],
        )
        self.recorded = [_Recorded(stokes_response, []), _Recorded(darcy_response, [])]
        transfers: list[PortTransferProtocol[Np, Np]] = [
            TimeComponentTransfer(stokes.transfer, q, bkd),
            TimeComponentTransfer(darcy.transfer, q, bkd),
        ]
        self.subdomains = [
            CoupledSubdomain(r, t) for r, t in zip(self.recorded, transfers)
        ]
        lu: RecomputeAlways[Np, DenseFormat[Np]] = RecomputeAlways(BackendDenseLU(bkd))
        solver = TransientSkeletonSolver(
            BackendVectorSpace(space.descriptor(), bkd),
            self.subdomains,
            DenseSchur(bkd),
            lu,
            Newton(atol=1e-10, rtol=1e-12, maxiters=20),
        )
        windows = [
            TimeWindow(n * final / nwindows, (n + 1) * final / nwindows)
            for n in range(nwindows)
        ]
        self.interface: InMemoryTrajectoryStore[Np] = InMemoryTrajectoryStore()
        solver.solve(np.repeat(lam0, q, axis=0), windows, self.interface)
        self.q, self.nskel_dofs = q, steady.space().descriptor().size
        self._stokes, self._darcy = stokes, darcy

    def pressure_offset(self) -> int:
        """Where the Stokes pressure starts in the Stokes state; the
        velocity is before it."""
        return int(self._stokes.basis.offset("pressure"))

    def final_states(self) -> tuple[Np, Np]:
        """The Stokes and Darcy states at the final time."""
        return self.stokes_sink.state(-1), self.darcy_sink.state(-1)

    def velocity_error(self) -> float:
        """The largest nodal error of the Stokes velocity at the final
        time."""
        basis, problem = self._stokes.basis, self._stokes.problem
        velocity = basis.field("velocity")
        if not isinstance(velocity, ComponentDofsBasisProtocol):
            raise TypeError("the velocity field is not a vector field")
        # The field's DOFs are global, a run of the mixed state.
        offset = basis.offset("velocity")
        dofs = np.arange(offset, offset + int(velocity.skfem_basis().N))
        components = np.asarray(velocity.dof_components(dofs))
        coords = np.asarray(basis.dof_coordinates())[:, dofs]
        exact = SkfemPoints([problem.solution("u"), problem.solution("v")], problem)
        values = exact(coords, t=self._final)[components, np.arange(dofs.size)]
        state = self.stokes_sink.state(-1)[dofs, 0]
        return float(np.abs(state - values).max())

    def darcy_error(self) -> float:
        """The largest nodal error of the Darcy pressure at the final
        time."""
        exact = exact_dof_values(
            self._darcy.problem, self._darcy.basis, ("u",), self._final
        )
        return float(np.abs(self.darcy_sink.state(-1)[:, 0] - exact).max())
