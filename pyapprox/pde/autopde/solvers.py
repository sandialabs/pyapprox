import torch
from abc import ABC, abstractmethod
from functools import partial
import numpy as np

from pyapprox.pde.autopde.util import newton_solve
from pyapprox.pde.autopde.time_integration import ImplicitRungeKutta


class AbstractFunction(ABC):
    def __init__(self, name, requires_grad=False, oned=False):
        self._name = name
        self._requires_grad = requires_grad
        self._oned = oned

    @abstractmethod
    def _eval(self, samples):
        raise NotImplementedError()

    def __call__(self, samples):
        vals = self._eval(samples)
        if not self._oned and vals.ndim != 2:
            raise ValueError("Function must return a 2D np.ndarray")
        if self._oned and vals.ndim != 1:
            raise ValueError("Function must return a 1D np.ndarray")
        if type(vals) == np.ndarray:
            vals = torch.tensor(
                vals, requires_grad=self._requires_grad, dtype=torch.double)
            return vals
        return vals.clone().detach().requires_grad_(self._requires_grad)


class AbstractTransientFunction(AbstractFunction):
    @abstractmethod
    def set_time(self, time):
        raise NotImplementedError()


class Function(AbstractFunction):
    def __init__(self, fun, name='fun', requires_grad=False, oned=False):
        super().__init__(name, requires_grad, oned)
        self._fun = fun

    def _eval(self, samples):
        return self._fun(samples)


class TransientFunction(AbstractFunction):
    def __init__(self, fun, name='fun', requires_grad=False, oned=False):
        super().__init__(name, requires_grad, oned)
        self._fun = fun
        self._partial_fun = None
        self._time = None

    def _eval(self, samples):
        # print(self._time, self._name)
        return self._partial_fun(samples)

    def set_time(self, time):
        self._time = time
        self._partial_fun = partial(self._fun, time=time)


class SteadyStatePDE():
    def __init__(self, residual):
        self.residual = residual

    def solve(self, init_guess=None, **newton_kwargs):
        if init_guess is None:
            init_guess = torch.ones(
                (self.residual.mesh.nunknowns), dtype=torch.double)
        init_guess = init_guess.squeeze()
        # requires_grad = self._auto
        auto_jac = self.residual._auto_jac
        if type(init_guess) == np.ndarray:
            sol = torch.tensor(
                init_guess.clone(), requires_grad=auto_jac,
                dtype=torch.double)
        else:
            sol = init_guess.clone().detach().requires_grad_(auto_jac)
        sol = newton_solve(
            self.residual._residual, sol, **newton_kwargs)
        return sol.detach().numpy()[:, None]


class TransientPDE():
    def __init__(self, residual, deltat, tableau_name):
        self.residual = residual
        self.time_integrator = ImplicitRungeKutta(
            deltat, self.residual._transient_residual, tableau_name,
            constraints_fun=self._apply_boundary_conditions)

    def _apply_boundary_conditions(self, raw_residual, raw_jac, sol, time):
        # boundary conditions are updated when residual is computed
        # for bndry_cond in self.residual._bndry_conds:
        #     if hasattr(bndry_cond[0], "set_time"):
        #         bndry_cond[0].set_time(time)
        return self.residual.mesh._apply_boundary_conditions(
            self.residual._bndry_conds, raw_residual, raw_jac, sol)

    def solve(self, init_sol, init_time, final_time, verbosity=0,
              newton_opts={}):
        sols = self.time_integrator.integrate(
            init_sol, init_time, final_time, verbosity, newton_opts)
        return sols
