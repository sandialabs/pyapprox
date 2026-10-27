"""
Time integrator with optional adjoint and HVP support.

Provides forward solve for any time-stepping residual. Adjoint solve
and gradient computation require AdjointEnabledTimeSteppingResidualProtocol.
Hessian-vector products require HVPEnabledTimeSteppingResidualProtocol.
"""

from typing import Generic, Optional, Tuple

from pyapprox.ode.functionals.protocols import (
    TransientFunctionalProtocol,
    TransientJacobianFn,
)
from pyapprox.ode.protocols.state_constraint import StateConstraintProtocol
from pyapprox.ode.protocols.time_stepping import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    TimeSteppingResidualProtocol,
)
from pyapprox.ode.residual_derivatives import backend_spaces, backend_state_space
from pyapprox.ode.state_constraint import (
    IdentityConstraint,
    TangentFn,
    sweep_tangent,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver


class TimeIntegrator(Generic[Array]):
    r"""
    Time integrator with optional adjoint and HVP support.

    Forward solve works with any TimeSteppingResidualProtocol. Adjoint
    and gradient methods require AdjointEnabledTimeSteppingResidualProtocol;
    HVP methods require HVPEnabledTimeSteppingResidualProtocol. Capability
    is checked lazily when the method is called, not at construction.

    Parameters
    ----------
    init_time : float
        Initial time.
    final_time : float
        Final time.
    deltat : float
        Time step size.
    newton_solver : NewtonSolver
        Newton solver for solving the residual equations.
    verbosity : int, optional
        Verbosity level. Defaults to 0.
    constraint : StateConstraintProtocol, optional
        The constraint :math:`\Pi_t` applied to every stored state: the
        initial state is :math:`\Pi_{t_0}(y_0)` and each step stores
        :math:`y_{n+1} = \Pi_{t_{n+1}}(z_{n+1})`, :math:`z_{n+1}` being
        the step's Newton solution. Every step starts from the stored
        state :math:`y_n`, never from :math:`z_n`. Defaults to the
        identity, which stores exactly what it is given.
    """

    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        newton_solver: NewtonSolver[Array],
        verbosity: int = 0,
        constraint: Optional[StateConstraintProtocol[Array]] = None,
    ):
        residual = newton_solver.residual()
        if not isinstance(residual, TimeSteppingResidualProtocol):
            raise TypeError(
                "Newton solver residual must satisfy "
                "TimeSteppingResidualProtocol, "
                f"got {type(residual).__name__}"
            )
        if constraint is not None and not isinstance(
            constraint, StateConstraintProtocol
        ):
            raise TypeError(
                "constraint must satisfy StateConstraintProtocol, "
                f"got {type(constraint).__name__}"
            )
        self._constraint = constraint
        self._unprojected: Optional[Array] = None
        self._stored: Optional[Array] = None
        self._time_residual: TimeSteppingResidualProtocol[Array] = residual
        self._bkd = self._time_residual.bkd()
        self._init_time = init_time
        self._final_time = final_time
        self._deltat = deltat
        self._verbosity = verbosity
        self._newton_solver = newton_solver
        self._functional: Optional[TransientFunctionalProtocol[Array]] = None
        self._functional_state_jacobian: Optional[TransientJacobianFn[Array]] = None
        self._functional_param_jacobian: Optional[TransientJacobianFn[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def ntimes(self) -> int:
        """Return the number of time points (including initial time)."""
        import math

        nsteps = math.ceil((self._final_time - self._init_time - 1e-12) / self._deltat)
        return nsteps + 1

    def set_functional(self, functional: TransientFunctionalProtocol[Array]) -> None:
        """
        Set the functional for gradient computation.

        This integrator owns the time-integration scheme, so it injects
        the scheme's trajectory quadrature into the functional: here, on
        the planned step times, so the functional can answer which
        derivatives it provides (that depends only on the rule's
        structure); and after every :meth:`solve`, on the times actually
        stepped through. A caller that wants a different rule injects it
        after the solve.

        Parameters
        ----------
        functional : TransientFunctionalProtocol
            Functional defining the quantity of interest.

        Raises
        ------
        TypeError
            If the functional's ``derivatives()`` has no Jacobians, which
            the adjoint gradient needs.
        """
        functional.set_time_quadrature(
            self._time_residual.trajectory_quadrature(
                self._bkd.asarray(self._planned_times())
            )
        )
        derivs = functional.derivatives()
        if derivs.state_jacobian is None or derivs.param_jacobian is None:
            raise TypeError(
                f"{type(functional).__name__} provides no Jacobians "
                "(derivatives().state_jacobian is None); the adjoint "
                "gradient needs them"
            )
        self._functional = functional
        self._functional_state_jacobian = derivs.state_jacobian
        self._functional_param_jacobian = derivs.param_jacobian

    def step(self, state: Array, deltat: float) -> Array:
        """
        Perform a single forward time step.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        deltat : float
            Time step size.

        Returns
        -------
        Array
            State at the next time step, constrained. Shape: (nstates,)
        """
        return self._step(state, deltat, self.state_constraint(int(state.shape[0])))[1]

    def _step(
        self,
        state: Array,
        deltat: float,
        constraint: StateConstraintProtocol[Array],
    ) -> Tuple[Array, Array]:
        """One step: the Newton solution and the constrained state stored."""
        ctx = StepContext(t_prev=self._time, deltat=deltat, y_prev=state)
        self._time_residual.bind(ctx)
        unprojected = self._newton_solver.solve(state)
        self._time += deltat
        if self._verbosity >= 1:
            print("Time:", self._time)
        return unprojected, constraint.constrain(unprojected, self._time)

    def state_constraint(
        self, nstates: int, nparams: Optional[int] = None
    ) -> StateConstraintProtocol[Array]:
        """The given constraint, or the identity on ``nstates`` states.

        The identity has parameter derivatives (stated zeros) when
        ``nparams`` is given, as the sensitivity sweeps need.
        """
        if self._constraint is not None:
            return self._constraint
        if nparams is None:
            return IdentityConstraint(backend_state_space(self._bkd, nstates))
        return IdentityConstraint(*backend_spaces(self._bkd, nstates, nparams))

    def linearization_states(self, stored: Array) -> Array:
        r"""The states the steps of trajectory ``stored`` are linearized at.

        Step :math:`n` solves for :math:`z_{n+1}` from the stored
        :math:`y_n`, so its derivatives in its unknown are taken at
        :math:`z_{n+1}`. With no constraint that is ``stored`` itself;
        with one it is :meth:`unprojected_states`, which belongs to the
        last solve only.

        Raises
        ------
        ValueError
            With a constraint, if ``stored`` is not the last solve's
            trajectory: its unprojected states are not kept.
        """
        if self._constraint is None:
            return stored
        last = self._stored
        if (
            last is None
            or self._unprojected is None
            or tuple(stored.shape) != tuple(last.shape)
            or self._bkd.to_float(self._bkd.max(self._bkd.abs(stored - last)))
            != 0.0
        ):
            raise ValueError(
                "with a state constraint, sensitivities need the trajectory "
                "of the integrator's last solve, whose unprojected states it "
                "keeps; re-solve before differentiating another trajectory"
            )
        return self._unprojected

    def _sweep_tangent(
        self,
        fwd_sols: Array,
        times: Array,
        param: Array,
        functional: TransientFunctionalProtocol[Array],
    ) -> TangentFn[Array]:
        """The constraint's tangent for the adjoint sweeps (the identity's
        when none is given), sized by the residual's parameters."""
        nparams = int(param.shape[0]) - functional.nunique_params()
        return sweep_tangent(
            self.state_constraint(int(fwd_sols.shape[0]), nparams).derivatives(),
            fwd_sols[:, 0],
            float(times[0]),
        )

    def unprojected_states(self) -> Array:
        r"""The last solve's states before the constraint was applied.

        Column 0 is the initial state as given, column :math:`n+1` the
        Newton solution :math:`z_{n+1}` of step :math:`n`. Under the
        identity constraint they are the stored states. Linearizations
        of the steps are taken here, not at the stored states.

        Returns
        -------
        Array
            Shape: (nstates, ntimes)

        Raises
        ------
        RuntimeError
            Before the first :meth:`solve`.
        """
        if self._unprojected is None:
            raise RuntimeError("unprojected_states is available after solve()")
        return self._unprojected

    def solve(self, init_state: Array) -> Tuple[Array, Array]:
        """
        Solve the time-dependent problem forward in time.

        Parameters
        ----------
        init_state : Array
            Initial state. Shape: (nstates,)

        Returns
        -------
        Tuple[Array, Array]
            states : Array
                Solution trajectory. Shape: (nstates, ntimes)
            times : Array
                Time points. Shape: (ntimes,)
        """
        constraint = self.state_constraint(int(init_state.shape[0]))
        states_list: list[Array] = []
        unprojected_list: list[Array] = [init_state]
        times_list: list[float] = []
        self._time = self._init_time
        times_list.append(self._time)
        state = constraint.constrain(init_state, self._time)
        states_list.append(state)
        while self._time < self._final_time - 1e-12:
            deltat = min(self._deltat, self._final_time - self._time)
            unprojected, state = self._step(state, deltat, constraint)
            unprojected_list.append(unprojected)
            states_list.append(state)
            times_list.append(self._time)
        states = self._bkd.stack(states_list, axis=1)
        self._unprojected = self._bkd.stack(unprojected_list, axis=1)
        self._stored = states
        times = self._bkd.asarray(times_list)
        if self._functional is not None:
            self._functional.set_time_quadrature(
                self._time_residual.trajectory_quadrature(times)
            )
        return states, times

    def _planned_times(self) -> list[float]:
        """Return the step times :meth:`solve` will visit.

        Uses the same stepping arithmetic as :meth:`solve`, so the
        planned grid equals the stepped one.
        """
        times = [self._init_time]
        time = self._init_time
        while time < self._final_time - 1e-12:
            time += min(self._deltat, self._final_time - time)
            times.append(time)
        return times

    def _require_adjoint(
        self,
    ) -> AdjointEnabledTimeSteppingResidualProtocol[Array]:
        """Narrow to adjoint-capable, raising TypeError if unavailable."""
        res = self._time_residual
        if isinstance(res, AdjointEnabledTimeSteppingResidualProtocol):
            return res
        raise TypeError(
            "Adjoint/gradient methods require the stepper to satisfy "
            "AdjointEnabledTimeSteppingResidualProtocol, "
            f"got {type(res).__name__}"
        )

    def time_residual(self) -> AdjointEnabledTimeSteppingResidualProtocol[Array]:
        """Return the time stepping residual narrowed to adjoint-capable."""
        return self._require_adjoint()

    def hvp_time_residual(
        self,
    ) -> Optional[HVPEnabledTimeSteppingResidualProtocol[Array]]:
        """Return the time residual narrowed to HVP capability, or None."""
        res = self._time_residual
        if isinstance(res, HVPEnabledTimeSteppingResidualProtocol):
            return res
        return None

    # =========================================================================
    # Adjoint Methods
    # =========================================================================

    def adjoint_step(
        self,
        ctx: StepContext[Array],
        next_ctx: StepContext[Array],
        fsol_n: Array,
        asol_np1: Array,
        dqdu_n: Array,
        y_curr_of_next: Array,
        tangent: TangentFn[Array],
        time_n: float,
    ) -> Array:
        r"""
        Perform a single backward adjoint step.

        Solves :math:`(\partial R_n/\partial z_n)^\top \lambda_n =
        -\Pi'(z_n)^\top [B_{n+1}^\top \lambda_{n+1} + dQ/dy_n]`, where
        :math:`B_{n+1} = \partial R_{n+1}/\partial y_n` and the stored
        state is :math:`y_n = \Pi(z_n)`; :math:`\Pi' = I` without a
        constraint.

        Parameters
        ----------
        ctx : StepContext
            Context for step n.
        next_ctx : StepContext
            Context for step n+1.
        fsol_n : Array
            The state step n is linearized at, :math:`z_n`.
            Shape: (nstates,)
        asol_np1 : Array
            Adjoint solution at time step n+1. Shape: (nstates,)
        dqdu_n : Array
            Gradient dQ/dy at time step n. Shape: (nstates,)
        y_curr_of_next : Array
            The state step n+1 is linearized at, :math:`z_{n+1}`.
            Shape: (nstates,)
        tangent : TangentFn
            The constraint's tangent :math:`\Pi'`.
        time_n : float
            The time of step n's result, where :math:`\Pi` was applied.

        Returns
        -------
        Array
            Adjoint solution at time step n. Shape: (nstates,)
        """
        residual = self._require_adjoint()
        drduT_diag = residual.adjoint_diag_jacobian(ctx, fsol_n)
        # B_{n+1}^T lambda_{n+1}: the operator acts on one-column blocks.
        offdiag = residual.off_diag_operator(next_ctx, y_curr_of_next)
        coupling = offdiag.apply_transpose(
            self._bkd.reshape(asol_np1, (asol_np1.shape[0], 1))
        )
        dqdu_block = self._bkd.reshape(dqdu_n, (dqdu_n.shape[0], 1))
        rhs = -tangent(fsol_n, time_n).apply_transpose(coupling + dqdu_block)[:, 0]
        return drduT_diag.solve(rhs)

    def solve_adjoint(self, fwd_sols: Array, times: Array, param: Array) -> Array:
        """
        Solve the adjoint equations backward in time.

        Parameters
        ----------
        fwd_sols : Array
            Forward solution trajectory. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Adjoint solution trajectory. Shape: (nstates, ntimes)
        """
        residual = self._require_adjoint()

        if not self._bkd.allclose(
            times[-1], self._bkd.asarray(self._final_time), atol=1e-12
        ):
            raise ValueError("times array is inconsistent with final_time")

        functional = self._functional
        state_jacobian = self._functional_state_jacobian
        if functional is None or state_jacobian is None:
            raise RuntimeError("Must call set_functional() first")

        # Compute dQ/dy at all time steps
        dqdu = state_jacobian(fwd_sols, param)

        # Steps are linearized at their own unknowns z_n; contexts carry
        # the stored previous states.
        lin_sols = self.linearization_states(fwd_sols)
        tangent = self._sweep_tangent(fwd_sols, times, param, functional)

        adj_sols = self._bkd.zeros(fwd_sols.shape)

        # Initial condition for adjoint at final time
        deltat_n = float(times[-1] - times[-2])
        ctx_final = StepContext(
            t_prev=float(times[-2]), deltat=deltat_n, y_prev=fwd_sols[:, -2]
        )

        dqdu_final = residual.zero_adjoint_rhs(dqdu[:, -1])
        dqdu_final = tangent(lin_sols[:, -1], float(times[-1])).apply_transpose(
            self._bkd.reshape(dqdu_final, (dqdu_final.shape[0], 1))
        )[:, 0]

        adj_sols = self._bkd.copy(adj_sols)
        adj_sols[:, -1] = residual.adjoint_initial_condition(
            ctx_final, lin_sols[:, -1], dqdu_final
        )

        # Backward sweep from N-2 to 1
        for nn in range(fwd_sols.shape[1] - 2, 0, -1):
            deltat_n = float(times[nn] - times[nn - 1])
            deltat_np1 = float(times[nn + 1] - times[nn])

            ctx_n = StepContext(
                t_prev=float(times[nn - 1]),
                deltat=deltat_n,
                y_prev=fwd_sols[:, nn - 1],
            )
            next_ctx = StepContext(
                t_prev=float(times[nn]),
                deltat=deltat_np1,
                y_prev=fwd_sols[:, nn],
            )

            dqdu_n = residual.zero_adjoint_rhs(dqdu[:, nn])

            adj_sols[:, nn] = self.adjoint_step(
                ctx_n,
                next_ctx,
                lin_sols[:, nn],
                adj_sols[:, nn + 1],
                dqdu_n,
                lin_sols[:, nn + 1],
                tangent,
                float(times[nn]),
            )

        # Final adjoint step at t=0
        # ctx_1 describes R_1 (the first forward step):
        # ctx_1.y_prev = y_0, y_curr = y_1
        deltat_0 = float(times[1] - times[0])
        ctx_1 = StepContext(
            t_prev=float(times[0]),
            deltat=deltat_0,
            y_prev=fwd_sols[:, 0],
        )

        dqdu_0 = residual.zero_adjoint_rhs(dqdu[:, 0])

        # y_0 = Pi(given y_0) is projected data, not a projected unknown, so
        # no tangent here: it enters once, in the gradient's t = 0 term.
        adj_sols[:, 0] = residual.adjoint_final_solution(
            ctx_1,
            lin_sols[:, 1],
            adj_sols[:, 1],
            dqdu_0,
        )

        return adj_sols

    def gradient(self, fwd_sols: Array, times: Array, param: Array) -> Array:
        """
        Compute the gradient dQ/dp via the adjoint method.

        Parameters
        ----------
        fwd_sols : Array
            Forward solution trajectory. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Gradient dQ/dp. Shape: (1, nparams)
        """
        adj_sols = self.solve_adjoint(fwd_sols, times, param)
        return self.gradient_from_adjoint_sols(adj_sols, fwd_sols, times, param)

    def gradient_from_adjoint_sols(
        self,
        adj_sols: Array,
        fwd_sols: Array,
        times: Array,
        param: Array,
    ) -> Array:
        """
        Compute gradient from pre-computed adjoint solutions.

        dQ/dp = dQ/dp|_direct + Σ_n λ_n^T · dR_n/dp

        Parameters
        ----------
        adj_sols : Array
            Adjoint solution trajectory. Shape: (nstates, ntimes)
        fwd_sols : Array
            Forward solution trajectory. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Gradient dQ/dp. Shape: (1, nparams)
        """
        residual = self._require_adjoint()

        functional = self._functional
        state_jacobian = self._functional_state_jacobian
        param_jacobian = self._functional_param_jacobian
        if functional is None or state_jacobian is None or param_jacobian is None:
            raise RuntimeError("Must call set_functional() first")
        # Direct parameter dependence of functional
        dqdp = param_jacobian(fwd_sols, param)
        grad = self._bkd.copy(dqdp)

        # Initial-state contribution, for an initial state that depends on
        # the parameters. By the chain rule it is
        # (dQ/dy_0 + B_1^T lambda_1)^T dy_0/dp, where B_1 = dR_1/dy_0.
        # That bracket is minus the right-hand side of the t = 0 adjoint
        # equation M^T lambda_0 = -B_1^T lambda_1 - dQ/dy_0, so the term
        # equals -lambda_0^T M dy_0/dp; forming it from the right-hand
        # side avoids a mass solve and uses exactly what that equation uses.
        # With a constraint the stored y_0 is Pi(given y_0), so dy_0/dp is
        # Pi' times the given state's derivative: the only place the
        # tangent enters at t = 0.
        lin_sols = self.linearization_states(fwd_sols)
        tangent = self._sweep_tangent(fwd_sols, times, param, functional)
        ctx_init = StepContext(
            t_prev=float(times[0]),
            deltat=float(times[1] - times[0]),
            y_prev=fwd_sols[:, 0],
        )
        self._time_residual.bind(ctx_init)

        dy0dp = tangent(lin_sols[:, 0], float(times[0])).apply(
            residual.initial_param_jacobian()
        )
        dqdu_0 = residual.zero_adjoint_rhs(state_jacobian(fwd_sols, param)[:, 0])
        offdiag = residual.off_diag_operator(ctx_init, lin_sols[:, 1])
        init_sensitivity = offdiag.apply_transpose(adj_sols[:, 1:2])[:, 0] + dqdu_0
        grad_init = self._bkd.reshape(init_sensitivity @ dy0dp, (1, -1))
        # Prepend zeros for functional-only parameters
        n_unique = functional.nunique_params()
        if n_unique > 0:
            zeros = self._bkd.zeros((1, n_unique))
            grad_init = self._bkd.hstack((zeros, grad_init))
        grad += grad_init

        # Accumulate contributions from each time step
        for ii in range(len(times) - 1):
            ctx_ii = StepContext(
                t_prev=float(times[ii]),
                deltat=float(times[ii + 1] - times[ii]),
                y_prev=fwd_sols[:, ii],
            )

            drdp = residual.param_jacobian(ctx_ii, lin_sols[:, ii + 1])
            # Prepend zeros for functional-only parameters
            n_unique = functional.nunique_params()
            if n_unique > 0:
                zeros = self._bkd.zeros((drdp.shape[0], n_unique))
                drdp = self._bkd.hstack((zeros, drdp))
            grad += adj_sols[:, ii + 1 : ii + 2].T @ drdp

        return grad

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"init_time={self._init_time}, "
            f"final_time={self._final_time}, "
            f"deltat={self._deltat})"
        )


# Alias for backward compatibility
ImplicitTimeIntegrator = TimeIntegrator
