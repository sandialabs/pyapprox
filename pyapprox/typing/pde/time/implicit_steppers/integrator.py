from typing import Generic, Tuple

from pyapprox.typing.util.backend import Array
from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
from pyapprox.typing.pde.time.implicit_steppers.protocols import (
    ImplicitTimeSteppingResidualProtocol,
)


class ImplicitTimeIntegrator(Generic[Array]):
    """
    Time integrator for solving time-dependent problems with impicit time stepping.

    This class uses a Newton solver to solve the residual equations at each
    time step.

    Parameters
    ----------
    time_residual : TimeIntegratorNewtonResidual
        Residual object implementing the TimeIntegratorNewtonResidual protocol.
    init_time : float
        Initial time.
    final_time : float
        Final time.
    deltat : float
        Time step size.
    newton_solver : NewtonSolver, optional
        Newton solver for solving the residual equations. Defaults to a new instance.
    verbosity : int, optional
        Verbosity level. Defaults to 0.
    """

    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        newton_solver: NewtonSolver,
        verbosity: int = 0,
    ):
        time_residual = newton_solver.residual()
        if not isinstance(time_residual, ImplicitTimeSteppingResidualProtocol):
            raise TypeError(
                "time_residual must be an instance of "
                "ImplicitTimeSteppingResidualProtocol, "
                "but received an object of type "
                "{type(time_residual).__name__}."
            )
        self._time_residual = time_residual
        self._bkd = self._time_residual.bkd()
        self._init_time = init_time
        self._final_time = final_time
        self._deltat = deltat
        self._verbosity = verbosity
        self._newton_solver = newton_solver

    def step(self, state: Array, deltat: float) -> Array:
        """
        Perform a single time step.

        Parameters
        ----------
        state : Array
            Current state.
        deltat : float
            Time step size.

        Returns
        -------
        Array
            State at the next time step.
        """
        self._time_residual.set_time(self._time, deltat, state)
        state = self._newton_solver.solve(state)
        self._time += deltat
        if self._verbosity >= 1:
            print("Time:", self._time)
            print(self._time_residual)
        return state

    def solve(self, init_state: Array) -> Tuple[Array, Array]:
        """
        Solve the time-dependent problem.

        Parameters
        ----------
        init_state : Array
            Initial state.

        Returns
        -------
        Tuple[Array, Array]
            States and corresponding time points.
        """
        states, times = [], []
        self._time = self._init_time
        times.append(self._time)
        state = init_state
        states.append(init_state)
        while self._time < self._final_time - 1e-12:
            deltat = min(self._deltat, self._final_time - self._time)
            state = self.step(state, deltat)
            states.append(state)
            times.append(self._time)
        states = self._bkd.stack(states, axis=1)
        return states, self._bkd.asarray(times)
