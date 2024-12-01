from abc import abstractmethod
from typing import Tuple

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.newton import (
    NewtonSolver,
    NewtonResidual,
    Functional,
    AdjointFunctional,
)
from pyapprox.surrogates.bases.univariate import (
    UnivariatePiecewisePolynomialNodeGenerator,
    UnivariatePiecewisePolynomialQuadratureRule,
)


class TransientNewtonResidual(NewtonResidual):
    # This is what user should derive from
    @abstractmethod
    def set_time(self, time: float):
        raise NotImplementedError

    def linsolve(self, sol: Array, res: Array):
        return self._bkd.solve(self.jacobian(sol), res)

    def mass_matrix(self, nstates):
        return self._bkd.eye(nstates)

    def initial_param_jacobian(self):
        raise NotImplementedError


class TimeIntegratorNewtonResidual(NewtonResidual):
    # This should only be derived from by developers implementing
    # new timestepping classes
    def __init__(self, residual: NewtonResidual):
        super().__init__(residual._bkd)
        self.native_residual = residual

    def _apply_constraints_to_residual(self, res_array: Array) -> Array:
        return res_array

    def _apply_constraints_to_jacobian(self, jac: Array) -> Array:
        return jac

    def _value(self, sol: Array) -> Array:
        raise NotImplementedError

    def __call__(self, sol: Array) -> Array:
        res_array = self._value(sol)
        return self._apply_constraints_to_residual(res_array)

    def _jacobian(self, sol: Array) -> Array:
        raise NotImplementedError

    def jacobian(self, sol: Array) -> Array:
        jac = self._jacobian(sol)
        return self._apply_constraints_to_jacobian(jac)

    def adjoint_implemented(self) -> bool:
        return False

    def set_time(self, time: float, deltat: float, prev_sol: Array):
        self._time = time
        self._deltat = deltat
        self._prev_sol = prev_sol

    def linsolve(self, sol: Array, res: Array) -> Array:

        # # leave around incase need to debug timestepping schemes and
        # # newton residuals
        # def autofun(sol_array):
        #     return self(sol_array)
        # jac_auto = self._bkd.jacobian(autofun, sol)
        # import torch
        # torch.set_printoptions(linewidth=1000, threshold=10000, sci_mode=False, precision=2)
        # print(res, "R")
        # print(self)
        # print(self._time, self._deltat)
        # print(jac_auto, "J")
        # print(self.jacobian(sol))
        # print((jac_auto-self.jacobian(sol)).max())
        # assert self._bkd.allclose(
        #     self.jacobian(sol),
        #     jac_auto,
        #     atol=1e-15
        # )

        # def autofun(sol_array):
        #     return self.native_residual(sol_array)
        # print("C", self.native_residual)
        # jac_auto = self._bkd.jacobian(autofun, sol)
        # import torch
        # torch.set_printoptions(linewidth=1000, precision=3, sci_mode=False)
        # # print(jac_auto)
        # # print(self.native_residual.jacobian(sol))
        # print((jac_auto-self.native_residual.jacobian(sol)).max())
        # assert self._bkd.allclose(
        #     self.native_residual.jacobian(sol),
        #     jac_auto,
        #     atol=1e-15
        # )
        # print("Warning using autograd to compute jacobian")
        # return self._bkd.solve(jac_auto, res)
        return self._bkd.solve(self.jacobian(sol), res)

    def _param_jacobian(self, fsol_nm1: Array, sol: Array) -> Array:
        raise NotImplementedError

    def param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """Gradient of residual with respect to parameters"""
        # this function assumes self._time has been set to time_nm1
        # and self._deltat = time_n-time_nm1
        if fsol_nm1.ndim != 1:
            raise ValueError("fsol_nm1 must be a 1d Array")
        if fsol_n.ndim != 1:
            raise ValueError("fsol_n must be a 1d Array")
        jac = self._param_jacobian(fsol_nm1, fsol_n)
        if jac.ndim != 2 or jac.shape[0] != fsol_n.shape[0]:
            raise RuntimeError(f"jac has the wrong shape {jac.shape}")
        return jac

    @abstractmethod
    def quadrature_samples_weights(self, times) -> Tuple[Array, Array]:
        raise NotImplementedError

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """
        Parameters
        ----------
        final_fwd_sol : Array (nstates)
            Solution of forward equations at final time at time step N

        final_dqdu : Array (nstates)
            Gradient of the QoI with respect to the solution of the
            forward equations at final time at time step N

        Return
        ------
        asol_N: Array
            The adjoint solution at the final time step N
            (assuming n increases with time)
        """
        drdu = self.jacobian(final_fwd_sol)
        return self._bkd.solve(drdu.T, -final_dqdu)

    def adjoint_final_solution(
        self, fsol_n: Array, asol_np1: Array, dqdu_n: Array, deltat_np1: float
    ) -> Array:
        # TODO if identity then no need to use solve below
        drduT_diag = self.native_residual.mass_matrix(fsol_n.shape[0]).T
        drduT_offdiag = self.adjoint_offdiag_jacobian(
            fsol_n,
            deltat_np1,
        )
        return self._bkd.solve(drduT_diag, -drduT_offdiag @ asol_np1 - dqdu_n)

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        """
        Compute the Jacobian of the residual with respect to the forward
        solution at time step n

        Parameters
        ----------
        fsol_n : Array (nstates)
            Solution of forward equations at final time at time step n
        """
        raise NotImplementedError

    def adjoint_offdiag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        """
        Compute the Jacobian of the residual with respect to the forward
        solutions at time step k != n used when time integrating

        Parameters
        ----------
        fsol_n : Array (nstates)
            Solution of forward equations at final time at time step n

        delta_np1: float
            the n+1 time step size
        """
        raise NotImplementedError

    def initial_param_jacobian(self) -> Array:
        jac = self.native_residual._initial_param_jacobian()
        if jac.ndim != 2:
            raise RuntimeError(f"jac has the wrong shape {jac.shape}")
        return jac

    def quadrature_rule(self):
        raise NotImplementedError


class ExplicitTimeIntegratorNewtonResidual(TimeIntegratorNewtonResidual):
    def _jacobian(self, sol: Array) -> Array:
        # todo: do not solve linear system when using explicit time integrators
        return self.native_residual.mass_matrix(sol.shape[0])

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        return -final_dqdu

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        return self.native_residual.mass_matrix(fsol_n.shape[0]).T


class UnivariateTransientNodeGenerator(
    UnivariatePiecewisePolynomialNodeGenerator
):
    def set_times(self, times: Array):
        self._times = times

    def _nodes(self, nnodes: int):
        return self._times[None, :]


class ForwardEulerResidual(ExplicitTimeIntegratorNewtonResidual):
    def adjoint_implemented(self) -> bool:
        return True

    def _value(self, sol: Array) -> Array:
        self.native_residual.set_time(self._time)
        return (
            sol
            - self._prev_sol
            - self._deltat * self.native_residual(self._prev_sol)
        )

    def _param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        self.native_residual.set_time(self._time)
        return -self._deltat * self.native_residual.param_jacobian(fsol_nm1)

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        node_gen = UnivariateTransientNodeGenerator(self._bkd)
        node_gen.set_times(times)
        quadx, quadw = UnivariatePiecewisePolynomialQuadratureRule(
            "leftconst", [times[0], times[-1]], node_gen, self._bkd, store=True
        )(times.shape[0])
        # left const rule does not return value at right end point so adjust
        return (
            self._bkd.hstack((quadx[0], times[-1])),
            self._bkd.hstack((quadw[:, 0], self._bkd.zeros((1,)))),
        )

    def adjoint_offdiag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        self.native_residual.set_time(self._time)
        return -(
            deltat_np1 * self.native_residual.jacobian(fsol_n)
            + self.native_residual.mass_matrix(fsol_n.shape[0])
        ).T


class BackwardEulerResidual(TimeIntegratorNewtonResidual):
    def adjoint_implemented(self) -> bool:
        return True

    def _value(self, sol: Array) -> Array:
        self.native_residual.set_time(self._time + self._deltat)
        return sol - self._prev_sol - self._deltat * self.native_residual(sol)

    def _jacobian(self, sol: Array) -> Array:
        self.native_residual.set_time(self._time + self._deltat)
        return self.native_residual.mass_matrix(
            sol.shape[0]
        ) - self._deltat * self.native_residual.jacobian(sol)

    def _param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        self.native_residual.set_time(self._time + self._deltat)
        return -self._deltat * self.native_residual.param_jacobian(fsol_n)

    def quadrature_samples_weights(self, times) -> Tuple[Array, Array]:
        node_gen = UnivariateTransientNodeGenerator(self._bkd)
        node_gen.set_times(times)
        quadx, quadw = UnivariatePiecewisePolynomialQuadratureRule(
            "rightconst",
            [times[0], times[-1]],
            node_gen,
            self._bkd,
            store=True,
        )(times.shape[0])
        # right const rule does not return value at left end point so adjust
        return (
            self._bkd.hstack((times[0], quadx[0])),
            self._bkd.hstack((self._bkd.zeros((1,)), quadw[:, 0])),
        )

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        # while forward solve requires setting
        # self.native_residual.set_time(self._time + self._deltat),
        # where self._time = time_nm1, self._deltat = time_n-times_nm1
        # adjoint solve passes in time_n so only do the following
        self.native_residual.set_time(self._time)
        return (
            self.native_residual.mass_matrix(fsol_n.shape[0])
            - self._deltat * self.native_residual.jacobian(fsol_n)
        ).T

    def adjoint_offdiag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        return -self.native_residual.mass_matrix(fsol_n.shape[0]).T


class HeunResidual(ExplicitTimeIntegratorNewtonResidual):
    def adjoint_implemented(self) -> bool:
        return True

    # Trapezoid integration
    def _value(self, sol: Array):
        self.native_residual.set_time(self._time)
        current_res = self.native_residual(self._prev_sol)
        next_sol = self._prev_sol + self._deltat * current_res
        self.native_residual.set_time(self._time + self._deltat)
        next_res = self.native_residual(next_sol)
        return (
            sol
            - self._prev_sol
            - 0.5 * self._deltat * (current_res + next_res)
        )

    # def param_hack(self, sol, param):
    #     self.native_residual.set_param(param)
    #     return self._value(sol)

    def _param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        self.native_residual.set_time(self._time)
        k1_param_jac = self.native_residual.param_jacobian(fsol_nm1)
        self.native_residual.set_time(self._time + self._deltat)
        #  d/dp g_p(x + d*g(x,p), p) =
        #      g_p(x + d*g(x,p))+d*g_x(x + d*g(x,p))g_p(x, p)
        k2 = fsol_nm1 + self._deltat * self.native_residual(fsol_nm1)
        k2_state_jac = self.native_residual.jacobian(k2)
        k2_param_jac = self.native_residual.param_jacobian(k2)
        jac = -(
            0.5
            * self._deltat
            * (
                k1_param_jac
                + k2_param_jac
                + self._deltat * (k2_state_jac @ k1_param_jac)
            )
        )
        # from functools import partial
        # print(self._bkd.abs(self._bkd.jacobian(partial(self.param_hack, fsol_n), self.native_residual._param)-jac).max(), "jp")
        # assert self._bkd.allclose(jac, self._bkd.jacobian(partial(self.param_hack, fsol_n), self.native_residual._param))
        return jac

    # def _state_hack(self, fsol_n, fsol_nm1):
    #     self._prev_sol = fsol_nm1
    #     return self._value(fsol_n)
    
    def adjoint_offdiag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        self.native_residual.set_time(self._time)
        k1_jac = self.native_residual.jacobian(fsol_n)
        self.native_residual.set_time(self._time + deltat_np1)
        #  d/dx g(x + d*g(x)) = (1 + d g'(x))g'(x + d g(x))
        #  k2 = x + d*g(x)
        k2 = fsol_n + deltat_np1 * self.native_residual(fsol_n)
        k2_jac = self.native_residual.jacobian(k2)
        mass = self.native_residual.mass_matrix(fsol_n.shape[0])
        jac = -(
            mass
            + 0.5
            * deltat_np1
            * (k1_jac + k2_jac @ (mass + deltat_np1 * k1_jac))
        )
        # from functools import partial
        # print(self._bkd.abs(jac-self._bkd.jacobian(partial(self._state_hack, fsol_n), self._prev_sol)).max(), "jx")
        # assert self._bkd.allclose(jac, self._bkd.jacobian(partial(self._state_hack, fsol_n), self._prev_sol))
        return jac.T

    def quadrature_samples_weights(self, times) -> Tuple[Array, Array]:
        node_gen = UnivariateTransientNodeGenerator(self._bkd)
        node_gen.set_times(times)
        quadx, quadw = UnivariatePiecewisePolynomialQuadratureRule(
            "linear",
            [times[0], times[-1]],
            node_gen,
            self._bkd,
            store=True,
        )(times.shape[0])
        return quadx[0], quadw[:, 0]


class CrankNicholsonResidual(TimeIntegratorNewtonResidual):
    def adjoint_implemented(self) -> bool:
        return True

    # Trapezoid integration
    def _value(self, sol: Array):
        self.native_residual.set_time(self._time)
        current_res = self.native_residual(self._prev_sol)
        self.native_residual.set_time(self._time + self._deltat)
        next_res = self.native_residual(sol)
        return (
            sol
            - self._prev_sol
            - 0.5 * self._deltat * (current_res + next_res)
        )

    def _jacobian(self, sol: Array):
        # self.native_residual.set_time(self._time)
        # current_jac = self.native_residual.jacobian(self._prev_sol)
        self.native_residual.set_time(self._time + self._deltat)
        next_jac = self.native_residual.jacobian(sol)
        return self.native_residual.mass_matrix(
            sol.shape[0]
        ) - 0.5 * self._deltat * (next_jac)

    def quadrature_samples_weights(self, times) -> Tuple[Array, Array]:
        node_gen = UnivariateTransientNodeGenerator(self._bkd)
        node_gen.set_times(times)
        quadx, quadw = UnivariatePiecewisePolynomialQuadratureRule(
            "linear",
            [times[0], times[-1]],
            node_gen,
            self._bkd,
            store=True,
        )(times.shape[0])
        return quadx[0], quadw[:, 0]

    def _param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        self.native_residual.set_time(self._time)
        current_param_jac = self.native_residual.param_jacobian(fsol_nm1)
        self.native_residual.set_time(self._time + self._deltat)
        next_param_jac = self.native_residual.param_jacobian(fsol_n)
        return -0.5 * self._deltat * (current_param_jac + next_param_jac)

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        # while forward solve requires setting
        # self.native_residual.set_time(self._time + self._deltat),
        # where self._time = time_nm1, self._deltat = time_n-times_nm1
        # adjoint solve passes in time_n so only do the following
        self.native_residual.set_time(self._time)
        return (
            self.native_residual.mass_matrix(fsol_n.shape[0])
            - self._deltat / 2 * self.native_residual.jacobian(fsol_n)
        ).T

    def adjoint_offdiag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        # while forward solve requires setting
        # self.native_residual.set_time(self._time + self._deltat),
        # where self._time = time_nm1, self._deltat = time_n-times_nm1
        # adjoint solve passes in time_n so only do the following
        self.native_residual.set_time(self._time)
        return -(
            +self.native_residual.mass_matrix(fsol_n.shape[0])
            + deltat_np1 / 2 * self.native_residual.jacobian(fsol_n)
        ).T


class RK4(ExplicitTimeIntegratorNewtonResidual):
    # Simpsons (piecewise quadratic) integration
    def _value(self, sol: Array):
        self.native_residual.set_time(self._time)
        k1_res = self.native_residual(self._prev_sol)
        self.native_residual.set_time(self._time + self._deltat / 2)
        k2_sol = self._prev_sol + self._deltat * k1_res / 2
        k2_res = self.native_residual(k2_sol)
        k3_sol = self._prev_sol + self._deltat * k2_res / 2
        k3_res = self.native_residual(k3_sol)
        self.native_residual.set_time(self._time + self._deltat)
        k4_sol = self._prev_sol + self._deltat * k3_res
        k4_res = self.native_residual(k4_sol)
        return (
            sol
            - self._prev_sol
            - self._deltat / 6 * (k1_res + 2 * k2_res + 2 * k3_res + k4_res)
        )

    def quadrature_samples_weights(self, times) -> Tuple[Array, Array]:
        node_gen = UnivariateTransientNodeGenerator(self._bkd)
        node_gen.set_times(times)
        quadx, quadw = UnivariatePiecewisePolynomialQuadratureRule(
            "quadratic",
            [times[0], times[-1]],
            node_gen,
            self._bkd,
            store=True,
        )(times.shape[0])
        return quadx[0], quadw[:, 0]


class SymplecticMidpointResidual(TimeIntegratorNewtonResidual):
    """
    First order implicit midpoint rule. It is the   lowest order
    Gauss-Legendre rule. All Gauss-Legendre rules are symplectic.

    Note
    ----
    Necause midpoint rule assumes solution is linear between two time steps
    y_{n+1/2} = 1/2(y_n+y_{n+1}).
    Thus rearranging
    y_{n+1}=2 y_{n+1/2} - y_n.
    We can use backward euler to compute y_{n+1/2} so that
    2 y_{n+1/2} - y_n = y_n + delta * G(t+delta/2, y_{n+1/2} so
    y_{n+1/2} = y_n + delta / 2 * G(t+delta/2, y_{n+1/2}
    so we can use newtons rule to solve for y_{n+1/2} i.e.
    y_{n+1/2} - y_n - delta/2*G(t+delta/2, y_{n+1/2} = 0 then
    correct via y_{n+1} = 2*y_{n+1/2}-y_n
    """

    def _value(self, sol: Array) -> Array:
        self.native_residual.set_time(self._time + self._deltat / 2)
        return (
            sol
            - self._prev_sol
            - self._deltat * self.native_residual((self._prev_sol + sol) / 2)
        )

    def _jacobian(self, sol: Array) -> Array:
        self.native_residual.set_time(self._time + self._deltat / 2)
        return self.native_residual.mass_matrix(
            sol.shape[0]
        ) - 0.5 * self._deltat * self.native_residual.jacobian(
            (self._prev_sol + sol) / 2
        )

    def quadrature_samples_weights(self, times) -> Tuple[Array, Array]:
        quadx = times
        deltat = times[1:] - times[:-1]
        quadw = self._bkd.hstack(
            [deltat[0] / 2, (deltat[1:] + deltat[-1:]) / 2, deltat[-1] / 2]
        )
        return quadx, quadw


class TransientFunctionalMixin:
    def set_quadrature_sample_weights(self, quadx, quadw):
        self._quadx = quadx
        self._quadw = quadw


class TransientFuncional(Functional, TransientFunctionalMixin):
    pass


class TransientAdjointFunctional(AdjointFunctional, TransientFunctionalMixin):
    def qoi_sol_jacobian(self, sol: Array) -> Array:
        """Gradient of qoi with respect to solution"""
        if sol.ndim != 2 or sol.shape[0] != self.nstates():
            raise ValueError("sol must be a 2d Array")
        jac = self._qoi_sol_jacobian(sol)
        if jac.ndim != 2 or jac.shape[1] != sol.shape[1]:
            raise RuntimeError("jac has the wrong shape")
        return jac

    def qoi_param_jacobian(self, sol: Array) -> Array:
        """Gradient of QoI with respect to parameters"""
        if sol.ndim != 2 or sol.shape[0] != self.nstates():
            raise ValueError("sol must be a 2d Array")
        jac = self._qoi_param_jacobian(sol)
        if jac.ndim != 1 or jac.shape[0] != self.nparams():
            raise RuntimeError("jac has the wrong shape")
        return jac


class ImplicitTimeIntegrator:
    def __init__(
        self,
        time_residual: TimeIntegratorNewtonResidual,
        init_time: float,
        final_time: float,
        deltat: float,
        newton_solver: NewtonSolver = None,
        verbosity: int = 0,
    ):
        if not isinstance(time_residual, TimeIntegratorNewtonResidual):
            raise ValueError(
                "residual must be an instance of TimeIntegratorNewtonResidual"
            )
        self._bkd = time_residual._bkd
        self._init_time = init_time
        self._final_time = final_time
        self._deltat = deltat
        self._verbosity = verbosity
        if newton_solver is None:
            newton_solver = NewtonSolver()
        if not isinstance(newton_solver, NewtonSolver):
            raise ValueError(
                "newton_solver must be an instance of NewtonSolver"
            )
        self.time_residual = time_residual
        self.newton_solver = newton_solver
        self.newton_solver.set_residual(time_residual)

    def set_functional(self, functional: TransientFunctionalMixin):
        if not isinstance(functional, TransientFunctionalMixin):
            raise ValueError(
                "functional must be an instance of TransientFunctionalMixin"
            )
        self._functional = functional

    def step(self, sol: Array, deltat: float) -> Array:
        self.time_residual.set_time(self._time, deltat, sol)
        sol = self.newton_solver.solve(self._bkd.copy(sol))
        self._time += deltat
        if self._verbosity >= 1:
            print("Time", self._time)
        return sol

    def solve(self, init_sol: Array) -> Tuple[Array, Array]:
        sols, times = [], []
        self._time = self._init_time
        times.append(self._time)
        sol = self._bkd.copy(init_sol)
        sols.append(init_sol)
        while self._time < self._final_time - 1e-12:
            deltat = min(self._deltat, self._final_time - self._time)
            sol = self.step(sol, deltat)
            sols.append(sol)
            times.append(self._time)
        sols = self._bkd.stack(sols, axis=1)
        return sols, self._bkd.array(times)

    def adjoint_step(
        self,
        fsol_n: Array,
        asol_np1: Array,
        dqdu_n: Array,
        deltat_n: float,
        deltat_np1: float,
        time_n: float,
    ) -> Array:
        """
        Parameters
        ----------
        fsol_n: Array
            The forward solution at the n time step

        asol_np1: Array
            The adjoint solution at the n+1 time step
            (assuming n increases with time),
            i.e. it is the most recently computed adjoint solution

        dqdu_n: Array
            The gradient of the QoI with respect to the solution at
            the n time step

        deltat_n: float
            the n time step size, deltat_n = time_n - time_nm1

        deltat_np1: float
            the n+1 time step size, deltat_np1 = time_np1 - time_n

        time_n: float
            The time at the n time step. time_nm1 = time_n - deltat_n.

        Return
        ------
        asol_n: Array
            The adjoint solution at time step n
            (assuming n increases with time)
        """

        # Jacobian of residual with respect to solution
        self._time = time_n
        self.time_residual.set_time(time_n, deltat_n, fsol_n)
        # Adjoint Jacobian functions already apply tranpose
        drduT_diag = self.time_residual.adjoint_diag_jacobian(fsol_n)
        drduT_offdiag = self.time_residual.adjoint_offdiag_jacobian(
            fsol_n,
            deltat_np1,
        )
        asol_n = self._bkd.solve(
            drduT_diag, -drduT_offdiag @ asol_np1 - dqdu_n
        )
        return asol_n

    def solve_adjoint(self, fwd_sols: Array, times: Array) -> Array:
        if not self._bkd.allclose(
            times[-1], self._bkd.atleast1d(self._final_time), atol=1e-12
        ):
            raise ValueError("times array is inconsistent with final_time")
        # copy required when using torch
        # todo compute dqdu at each time step rather than all upfront
        if not hasattr(self, "_functional"):
            raise RuntimeError("must call set_functional")
        dqdu = self._functional.qoi_sol_jacobian(fwd_sols)
        adj_sols = self._bkd.empty(fwd_sols.shape)
        # using notation deltat_n = t_n-tnm1, e.g. deltat_1 = t1 - t0
        # given times t0, t1, ..., t_Nm2, t_Nm1
        # start at n = Nm2, time = t_Nm2
        self._time = self._bkd.copy(times)[-2]
        deltat_n = times[-1] - times[-2]
        self.time_residual.set_time(self._time, deltat_n, fwd_sols[:, -1])
        adj_sols[:, -1] = self.time_residual.adjoint_initial_condition(
            fwd_sols[:, -1], dqdu[:, -1]
        )
        for nn in range(fwd_sols.shape[1] - 2, 0, -1):
            deltat_np1 = deltat_n
            deltat_n = times[nn] - times[nn - 1]
            # deltat_n = times[nn+1] - times[nn]
            adj_sols[:, nn] = self.adjoint_step(
                fwd_sols[:, nn],
                adj_sols[:, nn + 1],
                dqdu[:, nn],
                deltat_n,
                deltat_np1,
                times[nn],
            )
        deltat_np1 = deltat_n
        deltat_n = times[1] - times[0]
        self._time = times[0]
        self.time_residual.set_time(self._time, deltat_n, fwd_sols[:, 0])
        adj_sols[:, 0] = self.time_residual.adjoint_final_solution(
            fwd_sols[:, 0],
            adj_sols[:, 1],
            dqdu[:, 0],
            deltat_np1,
        )
        return adj_sols

    def gradient(self, fwd_sols: Array, times: Array) -> Array:
        adj_sols = self.solve_adjoint(fwd_sols, times)
        dqdp = self._functional.qoi_param_jacobian(fwd_sols)
        grad = dqdp
        # fwd_sols[:, 0] will never be used
        self.time_residual.set_time(
            times[0], times[1] - times[0], fwd_sols[:, 0]
        )
        drdp = self.time_residual.initial_param_jacobian()
        grad += adj_sols[:, 0] @ drdp
        for ii, time in enumerate(times[:-1], start=0):
            self.time_residual.set_time(
                times[ii], times[ii + 1] - times[ii], fwd_sols[:, ii]
            )
            drdp = self.time_residual.param_jacobian(
                fwd_sols[:, ii], fwd_sols[:, ii + 1]
            )
            grad += adj_sols[:, ii + 1] @ drdp
        return self._bkd.atleast2d(grad)

    def __repr__(self):
        return (
            "{0}(init_time={1}, final_time={2}, deltat={3}, residual={4})"
        ).format(
            self.__class__.__name__,
            self._init_time,
            self._final_time,
            self._deltat,
            self.time_residual._residual,
        )
