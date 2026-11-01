"""ODE QoI function and functional classes.

A QoI function solves the ODE for each parameter sample and returns a
quantity of interest of the trajectory. The functionals are computed as
the trajectory is (each solve writes into a fresh sink from the
functional), so evaluating a QoI keeps no trajectory;
:meth:`ODEQoIFunction.solve_trajectory` returns the whole trajectory for
callers that want it.
"""

from dataclasses import dataclass
from typing import (
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.accumulating_sink import AccumulatingSinkProtocol, PerStateSink
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import DenseModelProtocol, dense_mass_form_system
from pyapprox.ode.protocols.ode_residual import ODEResidualWithParamJacobianProtocol
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta, SteppingMethodProtocol
from pyapprox.ode.tableau import backward_euler
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectorySinkProtocol,
    trajectory_array,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ODEResidualWithInitialStateProtocol(Protocol[Array]):
    """ODE residual that also defines its initial state from parameters.

    ``get_initial_condition`` returns ``y_0(p)`` for the parameters last
    passed to ``set_param``. Its derivatives are the residual's
    ``initial_param_jacobian`` (and ``initial_param_hvp``), so the forward
    map and the derivatives come from one definition.
    """

    def set_param(self, param: Array) -> None: ...

    def get_initial_condition(self) -> Array: ...


@runtime_checkable
class ODEFunctionalProtocol(Protocol[Array]):
    """A quantity of interest of an ODE trajectory, computed as the
    trajectory is: no Jacobians or HVPs.

    Each solve gets a fresh sink from :meth:`sink`; the integrator writes
    the stored states into it, and its ``value()`` is the QoI, shape
    ``(nqoi,)``. The trajectory itself is never kept.
    """

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        ...

    def sink(self) -> AccumulatingSinkProtocol[Array, Array]:
        """A fresh sink accumulating this QoI over one solve."""
        ...


class _FinalRows(Generic[Array]):
    """Keeps the latest state; its value is rows ``start`` to ``stop`` of
    it, shape ``(stop - start,)``."""

    def __init__(self, start: int, stop: int) -> None:
        self._start, self._stop = start, stop
        self._last: Optional[Array] = None

    def add(self, index: int, time: float, state: Array) -> None:
        self._last = state

    def value(self) -> Array:
        if self._last is None:
            raise ValueError("no state received")
        return self._last[self._start : self._stop, 0]


class _RunningMax(Generic[Array]):
    """The largest value each state entry has taken, shape ``(nstates,)``."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._max: Optional[Array] = None

    def add(self, index: int, time: float, state: Array) -> None:
        column = state[:, 0]
        self._max = column if self._max is None else self._bkd.maximum(
            self._max, column
        )

    def value(self) -> Array:
        if self._max is None:
            raise ValueError("no state received")
        return self._max


class SelectedStateEntries(Generic[Array]):
    """Chosen entries of the states at chosen steps, as the states arrive.

    The value lists, for each row in ``rows`` in turn, that row's entry
    at every kept state in time order.

    Parameters
    ----------
    rows : sequence of int
        The state entries to keep.
    steps : sequence of int, optional
        The stored-state indices to keep (0 the initial state); every
        state when omitted.
    bkd : Backend[Array]
        Joins the kept entries.
    """

    def __init__(
        self,
        rows: Sequence[int],
        bkd: Backend[Array],
        steps: Optional[Sequence[int]] = None,
    ) -> None:
        self._rows = list(rows)
        self._steps = None if steps is None else frozenset(steps)
        self._bkd = bkd
        self._kept: List[List[Array]] = [[] for _ in self._rows]

    def add(self, index: int, time: float, state: Array) -> None:
        if self._steps is not None and index not in self._steps:
            return
        for entries, row in zip(self._kept, self._rows):
            entries.append(state[row : row + 1, 0])

    def value(self) -> Array:
        return self._bkd.concatenate(
            [entry for entries in self._kept for entry in entries]
        )


class AllStatesEndpointODEFunctional(Generic[Array]):
    """Return all states at the final time.

    Parameters
    ----------
    nstates : int
        Number of state variables.
    """

    def __init__(self, nstates: int) -> None:
        self._nstates = nstates

    def nqoi(self) -> int:
        return self._nstates

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.none()

    def sink(self) -> PerStateSink[Array, Array]:
        """Keeps only the latest state."""
        return PerStateSink(_FinalRows(0, self._nstates))


class SingleStateEndpointODEFunctional(Generic[Array]):
    """Return a single state at the final time.

    Parameters
    ----------
    state_idx : int
        Index of the state variable to extract.
    """

    def __init__(self, state_idx: int) -> None:
        self._state_idx = state_idx

    def nqoi(self) -> int:
        return 1

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.none()

    def sink(self) -> PerStateSink[Array, Array]:
        """Keeps only the latest state."""
        return PerStateSink(_FinalRows(self._state_idx, self._state_idx + 1))


class MaxODEFunctional(Generic[Array]):
    """Return the maximum of each state over time.

    Parameters
    ----------
    nstates : int
        Number of state variables.
    bkd : Backend[Array]
        Backend for array operations.
    """

    def __init__(self, nstates: int, bkd: Backend[Array]) -> None:
        self._nstates = nstates
        self._bkd = bkd

    def nqoi(self) -> int:
        return self._nstates

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.none()

    def sink(self) -> PerStateSink[Array, Array]:
        """Keeps a running maximum."""
        return PerStateSink(_RunningMax(self._bkd))


@runtime_checkable
class ODEBenchmarkResidualProtocol(
    DenseModelProtocol[Array],
    ODEResidualWithParamJacobianProtocol[Array],
    Protocol[Array],
):
    """What an ODE benchmark's residual provides: a dense model with a
    mass and a derivatives bundle, which the steppers of
    :mod:`pyapprox.ode.stepping_methods` step, and the method-style
    parameter interface (``set_param``, ``nparams``, ``jacobian``,
    ``param_jacobian``) the earlier time integration reads."""


class ODEQoIFunction(Generic[Array]):
    """Callable QoI function wrapping an ODE residual.

    Integrates the ODE for each parameter sample and extracts a quantity
    of interest from the solution trajectory via a functional.

    Parameters
    ----------
    residual : ODEBenchmarkResidualProtocol
        :math:`M \\dot y = f(y, t; p)`; its backend and number of
        parameters are this function's.
    initial_condition : Array
        Initial condition. Shape: (nstates, 1) or (nstates,). Used only
        when the residual does not define its own initial state
        (``ODEResidualWithInitialStateProtocol``); one that does is asked
        for ``y_0(p)`` for every sample, so parameters that set the
        initial state reach the solution.
    time_config : ODETimeConfig
        The interval and step; the last step is shortened when the
        interval is not a whole number of steps.
    functional : ODEFunctionalProtocol
        Functional that extracts QoI from the solution trajectory.
    method : SteppingMethodProtocol, optional
        The time-stepping method, bound to the residual for every solve.
        Defaults to backward Euler with Newton at its defaults.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.

    Examples
    --------
    >>> qoi = ODEQoIFunction(
    ...     residual, init, time_config, functional,
    ...     method=RungeKutta(crank_nicolson()),
    ... )
    """

    def __init__(
        self,
        residual: ODEBenchmarkResidualProtocol[Array],
        initial_condition: Array,
        time_config: "ODETimeConfig",
        functional: ODEFunctionalProtocol[Array],
        method: Optional[SteppingMethodProtocol[Array, Array]] = None,
    ) -> None:
        if not isinstance(residual, ODEBenchmarkResidualProtocol):
            raise TypeError(
                "residual must satisfy ODEBenchmarkResidualProtocol, got "
                f"{type(residual).__name__}"
            )
        if not isinstance(functional, ODEFunctionalProtocol):
            raise TypeError(
                "functional must satisfy ODEFunctionalProtocol, got "
                f"{type(functional).__name__}"
            )
        if method is not None and not isinstance(method, SteppingMethodProtocol):
            raise TypeError(
                "method must satisfy SteppingMethodProtocol, got "
                f"{type(method).__name__}"
            )
        self._residual = residual
        self._functional = functional
        self._method: SteppingMethodProtocol[Array, Array] = (
            RungeKutta(backward_euler()) if method is None else method
        )
        self._bkd = residual.bkd()
        self._time_config = time_config
        self._nparams = residual.nparams()
        self._initial_state_residual: Optional[
            ODEResidualWithInitialStateProtocol[Array]
        ] = (
            residual
            if isinstance(residual, ODEResidualWithInitialStateProtocol)
            else None
        )
        # The steppers take states as single columns.
        self._init_state = self._bkd.reshape(initial_condition, (-1, 1))
        self._nstates = self._init_state.shape[0]

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables (parameters)."""
        return self._nparams

    def nqoi(self) -> int:
        """Return number of QoI outputs."""
        return self._functional.nqoi()

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.none()

    def __call__(self, samples: Array) -> Array:
        """Evaluate QoI for each parameter sample.

        Parameters
        ----------
        samples : Array
            Parameter samples. Shape: (nparams, nsamples)

        Returns
        -------
        Array
            QoI values. Shape: (nqoi, nsamples)
        """
        nsamples = samples.shape[1]
        qoi_values = []

        for ii in range(nsamples):
            param = samples[:, ii]
            qoi = self._evaluate_single(param)
            qoi_values.append(qoi)

        return self._bkd.stack(qoi_values, axis=1)

    def solve_trajectory(self, param: Array) -> Tuple[Array, Array]:
        """Solve the ODE and return full trajectory.

        Parameters
        ----------
        param : Array
            Single parameter sample. Shape: (nparams, 1) or (nparams,)

        Returns
        -------
        solutions : Array
            Solution trajectory. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        """
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        self.solve_into(param, store)
        return trajectory_array(store, self._bkd), self._bkd.asarray(store.times())

    def solve_into(self, param: Array, sink: TrajectorySinkProtocol[Array]) -> None:
        """Solve at ``param`` (shape ``(nparams, 1)`` or ``(nparams,)``),
        writing the stored states into ``sink``, which must be empty: a
        functional's sink, a store, or a
        :class:`~pyapprox.ode.accumulating_sink.MultiSink` feeding several
        from one solve."""
        if param.ndim == 2:
            param = param[:, 0]
        self._residual.set_param(param)
        # A new stepper per solve: the system is built at the parameters
        # just set.
        stepper = self._method.stepper(dense_mass_form_system(self._residual))
        init_state = (
            self._init_state
            if self._initial_state_residual is None
            else self._bkd.reshape(
                self._initial_state_residual.get_initial_condition(), (-1, 1)
            )
        )
        Integrator(stepper).solve(init_state, self._time_config.steps(), sink)

    def _evaluate_single(self, param: Array) -> Array:
        """Evaluate QoI for a single parameter sample, accumulated while
        solving; no trajectory is kept.

        Parameters
        ----------
        param : Array
            Single parameter sample. Shape: (nparams,)

        Returns
        -------
        Array
            QoI values. Shape: (nqoi,)
        """
        sink = self._functional.sink()
        self.solve_into(param, sink)
        return sink.value()


@dataclass
class ODETimeConfig:
    """Time configuration for ODE integration.

    Attributes
    ----------
    init_time : float
        Initial time for integration.
    final_time : float
        Final time for integration.
    deltat : float
        Time step for integration.
    """

    init_time: float
    final_time: float
    deltat: float

    def ntimes(self) -> int:
        """The number of stored times, :meth:`times` included the
        shortened last step's."""
        return len(self.times())

    def steps(self) -> FixedStep:
        """Steps of ``deltat`` from ``init_time``, landing exactly on
        ``final_time``: an interval that is a whole number of steps, to a
        relative 1e-9, has that many; otherwise the last step is
        shortened. The count is decided first, so rounding never adds a
        vanishing step."""
        return FixedStep(self.deltat, (self.init_time, self.final_time))

    def times(self) -> List[float]:
        """The step boundaries from ``init_time`` to ``final_time``,
        derived from :meth:`steps`' sizes."""
        return list(self.steps().times())


