import numpy as np
from abc import ABC, abstractmethod
import torch


from pyapprox.surrogates.interp.tensorprod import (
    UnivariatePiecewiseLeftConstantBasis,
    UnivariatePiecewiseRightConstantBasis,
    UnivariatePiecewiseMidPointConstantBasis,
    UnivariatePiecewiseLinearBasis)


class TorchArrayMethods():
    @staticmethod
    def copy(array: torch.tensor) -> torch.tensor:
        return array.clone()

    @staticmethod
    def norm(array: torch.tensor, *args, **kwargs) -> float:
        return torch.linalg.norm(array, *args, **kwargs)

    @staticmethod
    def jacobian(fun: callable, array: torch.tensor) -> torch.tensor:
        # strict=True needed if computing adjoints and jac computation
        # needs to be part of graph
        if not array.requires_grad:
            raise ValueError("array must have requires_grad=True")
        if not array.ndim == 1:
            raise ValueError("array must be 1D tensor so AD can be used")
        return torch.autograd.functional.jacobian(
            fun, array, strict=True)

    @staticmethod
    def solve(mat: torch.tensor, vec: torch.tensor,
              *args, **kwargs) -> torch.tensor:
        return torch.linalg.solve(mat, vec)

    @staticmethod
    def eye(N):
        return torch.eye(N)

    @staticmethod
    def stack(arrays, axis=0):
        return torch.stack(arrays, dim=axis)


class NumpyArrayMethods():
    @staticmethod
    def copy(array) -> np.ndarray:
        return array.copy()

    @staticmethod
    def norm(array, *args, **kwargs) -> float:
        return np.linalg.norm(array, *args, **kwargs)

    @staticmethod
    def jacobian(fun: callable, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Numpy does not support automatic differentiation")

    @staticmethod
    def solve(mat: np.ndarray, vec: np.ndarray,
              *args, **kwargs) -> np.ndarray:
        return np.linalg.solve(mat, vec)

    @staticmethod
    def eye(N):
        return np.eye(N)

    @staticmethod
    def stack(arrays, axis=0):
        return np.stack(arrays, axis=axis)


class NewtonSolver():
    def __init__(self, atol=1e-7, maxiters=10, verbosity=0, step_size=1,
                 rtol=1e-7):
        self._linalg = NumpyArrayMethods()
        self._atol = atol
        self._rtol = rtol
        self._maxiters = maxiters
        self._verbosity = verbosity
        self._step_size = step_size

    def solve(self, residual_fun, init_guess):
        sol = self._linalg.copy(init_guess)
        residual_norms = []
        it = 0
        while True:
            residual, jac = residual_fun(sol)
            residual_norm = self._linalg.norm(residual)
            residual_norms.append(residual_norm)
            if self._verbosity > 1:
                print("Iter", it, "rnorm", residual_norm)
            if (it > 0 and (residual_norm <=
                            self._atol+self._rtol*residual_norms[0])):
                # must take at least one step for cases when residual
                # is under tolerance for init_guess
                exit_msg = "Tolerance {0} reached".format(
                    self._atol+self._rtol*residual_norms[0])
                break
            if it >= self._maxiters:
                exit_msg = f"Max iterations {self._maxiters} reached.\n"
                exit_msg += f"Rel residual norm is {residual_norm} "
                exit_msg += "Needs to be {0}".format(
                    self._atol+self._rtol*residual_norms[0])
                raise RuntimeError(exit_msg)
            sol = sol-self._step_size*self._linalg.solve(jac, residual)
            it += 1
        if self._verbosity > 0:
            print(exit_msg)
        return sol


class TimeIntegratorResidual(ABC):
    @abstractmethod
    def _value(self, sol, time):
        raise NotImplementedError

    def __call__(self, sol, time):
        if sol.ndim != 1:
            raise ValueError("sol must be 1D array")
        return self._value(sol, time), self.state_jacobian(sol, time)

    @abstractmethod
    def state_jacobian(self, sol, time):
        raise NotImplementedError

    def apply_constraints(self, sol, time):
        raise NotImplementedError

    def parameter_jacobian(self, fwd_sol, params):
        """
        Jacobian of the residual with respect to the parameters

        Necessary if computing adjoints
        """
        raise NotImplementedError


class CustomTimeIntegratorResidual(TimeIntegratorResidual):
    def __init__(self, fun, jac):
        self._fun = fun
        self._jac = jac

    def _value(self, sol, time):
        return self._fun(sol, time)

    def state_jacobian(self, sol, time):
        return self._jac(sol, time)


class TimeIntegratorUpdate(ABC):
    def __init__(self, residual: TimeIntegratorResidual):
        self._residual = residual
        self._basis = self._get_basis()

    @abstractmethod
    def _get_basis(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, prev_sol, prev_time, deltat):
        raise NotImplementedError

    def integrate(self, times, sols):
        return self._basis.integrate(times, sols)


class ImplicitTimeIntegratorUpdate(TimeIntegratorUpdate):
    def __init__(self, residual: TimeIntegratorResidual,
                 **newton_kwargs):
        super().__init__(residual)
        self._newton_solver = NewtonSolver(**newton_kwargs)

        self._linalg = NumpyArrayMethods()

        self._time = None
        self._deltat = None
        self._mass_matrix = None

        # todo add support for AD with torch by ammending residual to return
        # jac computed with
        # jac = self._linalg.jacobian(lambda s: self._residual(s)[0], sol)

    @abstractmethod
    def residual(self, sol_guess):
        raise NotImplementedError

    def _get_mass_matrix(self):
        return self._linalg.eye(self._nstates)

    def _set_time(self, prev_sol, prev_time, deltat):
        self._prev_sol = prev_sol
        self._prev_time = prev_time  # time corresponding to prev_sol
        self._deltat = deltat
        self._nstates = prev_sol.shape[0]
        if self._mass_matrix is None:
            self._mass_matrix = self._get_mass_matrix()

    def __call__(self, prev_sol, prev_time, deltat):
        self._set_time(prev_sol, prev_time, deltat)
        init_guess = prev_sol
        return self._newton_solver.solve(self.residual, init_guess)


class ExplicitTimeIntegratorUpdate(TimeIntegratorUpdate):
    def __init__(self, residual: TimeIntegratorResidual):
        super().__init__(residual)
        self._linalg = NumpyArrayMethods()
        self._mass_matrix_inv = self._get_mass_matrix_inv()

    def _get_mass_matrix_inv(self):
        return self._linalg.eye(self._nstates)

    def apply_full_mass_matrix_inv(self, vec):
        return self._mass_matrix_inv @ vec

    def apply_lumped_mass_matrix_inv(self, vec):
        return self._lumped_mass_matrix_inv[:, None] @ vec

    def apply_mass_matrix_inv(self, vec):
        if self._mass_matrix_inv.ndim == 2:
            return self.apply_full_mass_matrix_inv(vec)
        return self.apply_lumped_mass_matrix_inv(vec)


class ForwardEulerUpdate(ExplicitTimeIntegratorUpdate):
    def __call__(self, prev_sol, prev_time, deltat):
        # TODO must apply constrants (e.g. boundary conditions)
        # to residual before applying mass matrix
        return prev_sol + deltat * self.apply_mass_matrix_inv(self._residual(
            prev_sol, prev_time)[0])

    def update_adjoint(self, future_sol, future_time, rdeltat):
        """
        Parameters
        ----------
        rdeltat : float < 0
            A timestep backwards in time
        """
        state_jac = self._residual(future_sol, future_time)[1]
        # TODO must apply constrants (e.g. boundary conditions)
        # to state_jac before next line
        return future_sol + rdeltat * self.apply_mass_matrix_inv(
            state_jac.T @ future_sol)

    @staticmethod
    def _get_basis():
        return UnivariatePiecewiseLeftConstantBasis()


class ExpicitMidpointUpdate(ExplicitTimeIntegratorUpdate):
    def __call__(self, prev_sol, prev_time, deltat):
        stage_sol = prev_sol+deltat/2*self.apply_mass_matrix_inv(
            self._residual(prev_sol, prev_time)[0])
        return prev_sol + deltat * self.apply_mass_matrix_inv(
            self._residual(stage_sol, prev_time+deltat/2)[0])

    @staticmethod
    def _get_basis():
        return UnivariatePiecewiseMidPointConstantBasis()


class BackwardEulerUpdate(ImplicitTimeIntegratorUpdate):
    def residual(self, sol_guess):
        gres, gres_jac = self._residual(sol_guess, self._prev_time)
        jac = self._mass_matrix - self._deltat*gres_jac
        res = (self._mass_matrix @ (sol_guess - self._prev_sol) -
               self._deltat * gres)
        # TODO must apply constrants (e.g. boundary conditions)
        # to jac and res before next line
        return res, jac

    @staticmethod
    def _get_basis():
        return UnivariatePiecewiseRightConstantBasis()


class TrapezoidUpdate(ImplicitTimeIntegratorUpdate):
    def __init__(self, residual: TimeIntegratorResidual,
                 **newton_kwargs):
        super().__init__(residual, **newton_kwargs)

        self._prev_gres = None
        self._prev_gres_time = None

    def _get_prev_gres(self):
        if self._prev_gres is None or self._prev_gres_time != self._prev_time:
            # if first time or evaluating residual at different time from
            # when self._prev_gres was computed
            # WARNING: This will produce wrong answer if self._residual object
            # is changed. Though this should be avoided.
            self._prev_gres = self._residual(
                self._prev_sol, self._prev_time)[0]
            self._prev_gres_time = self._prev_time
        return self._prev_gres

    def residual(self, sol_guess):
        gres, res_jac = self._residual(sol_guess, self._prev_time+self._deltat)
        jac = self._mass_matrix - self._deltat/2*res_jac
        prev_gres = self._get_prev_gres()
        res = (self._mass_matrix @ (sol_guess - self._prev_sol) -
               self._deltat/2*(prev_gres + gres))
        self._update_gres(res, self._prev_time+self._deltat)
        return res, jac

    def _update_gres(self, res, time):
        # Gres computed in one timstep becomes gres_prev in next time step
        # so store to save computation
        self._prev_gres = self._linalg.copy(res)
        self._prev_gres_time = time

    @staticmethod
    def _get_basis():
        return UnivariatePiecewiseLinearBasis()


class TimeIntegrator(ABC):
    def __init__(self,
                 residual: TimeIntegratorResidual,
                 tableau_name: str,
                 **newton_kwargs):
        self._tableau_name = tableau_name
        self._newton_kwargs = newton_kwargs
        self._update = self.init_update(
            residual, tableau_name, newton_kwargs)
        self._linalg = NumpyArrayMethods()

    @staticmethod
    def init_update(residual, tableau_name, newton_kwargs):
        updates = {"imeuler1": BackwardEulerUpdate,
                   "imtrap2": TrapezoidUpdate,
                   "exeuler1": ForwardEulerUpdate,
                   "exmid2": ExpicitMidpointUpdate}
        if tableau_name not in updates:
            msg = f"tableau_name {0} not found must be in {1}".format(
                tableau_name, list(updates.keys()))
            raise ValueError(msg)
        return updates[tableau_name](residual, **newton_kwargs)

    def get_update(self):
        return self._update

    def update(self, sol, time, deltat):
        return self._update(sol, time, deltat)

    @abstractmethod
    def solve(selfinit_sol, init_time, final_time, deltat, **kwargs):
        raise NotImplementedError


class StateTimeIntegrator(TimeIntegrator):
    def solve(self, init_sol, init_time, final_time, deltat,
              verbosity=0, init_deltat=None):
        """
        Parameters
        ----------
        init_deltat : float
            The size of the first time step. If None then self.deltat will be
            used. This is needed for solving adjoint equations
        """
        sols, times = [], []
        time = init_time
        times.append(time)
        sol = self._linalg.copy(init_sol)
        if sol.ndim == 2:
            sol = sol[:, 0]
        sols.append(self._linalg.copy(sol))
        while time < final_time-1e-12:
            if verbosity >= 1:
                print("Time", time)
            if init_deltat is not None:
                if init_time+init_deltat > final_time:
                    raise ValueError("init_deltat is to large")
                _deltat = init_deltat
                init_deltat = None
            else:
                _deltat = min(deltat, final_time-time)
            sol = self._update(sol, time, _deltat)
            sols.append(self._linalg.copy(sol))
            time += _deltat
            times.append(time)
        if verbosity >= 1:
            print("Time", time)
        sols = self._linalg.stack(sols, axis=1).T
        times = self._linalg.stack(times)
        return sols, times


class QuantityOfInterest(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def parameter_jacobian(self, fwd_sols, params):
        """
        Jacobian of QoI with respect to the parameters

        Necessary if computing adjoint solutions
        """
        raise NotImplementedError

    def state_jacobian(self, fwd_sol, params):
        """
        Jacobian of QoI with respect to the states

        Necessary if computing adjoint solutions
        """
        raise NotImplementedError


class AdjointTimeIntegrator(TimeIntegrator):
    def __init__(self,
                 tableau_name: str,
                 **newton_kwargs):
        residual = self._init_residual()
        super().__init__(residual, tableau_name, **newton_kwargs)

    def _init_residual(self):
        pass

    def solve(self):
        return super().solve(init_time, final_time, deltat)
