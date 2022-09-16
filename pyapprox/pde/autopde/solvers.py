import torch
import copy
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
    def __init__(self, physics):
        self.physics = physics

    def solve(self, init_guess=None, **newton_kwargs):
        if init_guess is None:
            init_guess = torch.ones(
                (self.physics.mesh.nunknowns), dtype=torch.double)
        init_guess = init_guess.squeeze()
        # requires_grad = self._auto
        auto_jac = self.physics._auto_jac
        if type(init_guess) == np.ndarray:
            sol = torch.tensor(
                init_guess.clone(), requires_grad=auto_jac,
                dtype=torch.double)
        else:
            sol = init_guess.clone().detach().requires_grad_(auto_jac)
        sol = newton_solve(
            self.physics._residual, sol, **newton_kwargs)
        return sol


class TransientPDE():
    def __init__(self, physics, deltat, tableau_name):
        self.physics = physics
        self.time_integrator = ImplicitRungeKutta(
            deltat, self.physics._transient_residual, tableau_name,
            constraints_fun=self._apply_boundary_conditions,
            auto=physics._auto_jac)

    def _apply_boundary_conditions(self, raw_residual, raw_jac, sol, time):
        # boundary conditions are updated when residual is computed
        # for bndry_cond in self.physics._bndry_conds:
        #     if hasattr(bndry_cond[0], "set_time"):
        #         bndry_cond[0].set_time(time)
        return self.physics.mesh._apply_boundary_conditions(
            self.physics._bndry_conds, raw_residual, raw_jac, sol,
            self.physics.flux_jac)

    def solve(self, init_sol, init_time, final_time, verbosity=0,
              newton_kwargs={}):
        sols, times = self.time_integrator.integrate(
            init_sol, init_time, final_time, verbosity, newton_kwargs)
        return sols, times


class SteadyStateAdjointPDE(SteadyStatePDE):
    def __init__(self, fwd_solver, functional, dqdu=None, dqdp=None,
                 dRdp=None):
        if type(fwd_solver) != SteadyStatePDE:
            raise ValueError("fwd_solver must be of type SteadyStatePDE")
        self._fwd_solver = fwd_solver
        self._functional = functional
        self._dqdu = dqdu
        self._dqdp = dqdp
        self._dRdp = dRdp

    def solve_adjoint(self, fwd_sol, param_vals, **newton_kwargs):
        jac = self._fwd_solver.physics._raw_residual(fwd_sol)[1]
        if jac is None:
            fwd_sol.requires_grad_(True)
            assert fwd_sol.requires_grad
            jac = torch.autograd.functional.jacobian(
                lambda s: self.physics._raw_residual(s)[0], fwd_sol,
                strict=True)
            fwd_sol.requires_grad_(False)
        adj_bndry_conds = copy.deepcopy(self._fwd_solver.physics._bndry_conds)
        for bndry_cond in adj_bndry_conds:
            # for now only support dirichlet boundary conds
            assert bndry_cond[1] == "D"
            bndry_cond[0] = lambda xx: np.zeros((xx.shape[1], 1))
        fwd_sol_copy = torch.clone(fwd_sol)
        param_vals_copy = torch.clone(param_vals)
        if self._dqdu is None:
            functional_vals = self._functional(
                fwd_sol_copy.requires_grad_(True),
                param_vals_copy.requires_grad_(True))
            functional_vals.backward()
            dqdu = fwd_sol_copy.grad
        else:
            functional_vals = self._functional(fwd_sol_copy, param_vals_copy)
            dqdu = self._dqdu(fwd_sol_copy, param_vals_copy)
        # need to pass in 0 instead of fwd_sol to apply boundary conditions
        # because we are not updating a residual rhs=sol-bndry_vals
        # but rather just want rhs=bndry_vals.
        # Techincally passing in we should be passing in negative bndry_vals
        # but since for dirichlet boundaries bndry_vals = 0 this is fine
        rhs, jac_adjoint = (
            self._fwd_solver.physics.mesh._apply_boundary_conditions(
                adj_bndry_conds, -dqdu, jac.clone().T, fwd_sol*0,
                self._fwd_solver.physics.flux_jac))
        adj_sol = torch.linalg.solve(jac_adjoint, rhs)
        return adj_sol
        # print("raw_adj_rhs", -dqdu)
        # print("adj_rhs", rhs)
        # print(jac_adjoint, 'jac_adj')
        # print("adj_sol", adj_sol)

        # # continous adjoint for pure diffusion
        # mesh = self._fwd_solver.physics.mesh
        # cont_jac = mesh._apply_boundary_conditions(
        #          adj_bndry_conds, -dqdu, jac, fwd_sol*0)[1]
        # cont_adj_sol = torch.linalg.solve(cont_jac, rhs)
        # print(cont_jac, 'continuous adj jac')
        # print(adj_sol, 'contuous adj')
        # #adj_sol = tmp
        # # import matplotlib.pyplot as plt
        # # ax = plt.subplots(1, 1)[1]
        # # mesh.plot(adj_sol, nplot_pts_1d=51, ax=ax, label="discrete")
        # # ax.plot(mesh.mesh_pts[0, :], adj_sol, 'o')
        # # mesh.plot(cont_adj_sol, nplot_pts_1d=51, ax=ax, label="cont")
        # # plt.legend()
        # # plt.show()
        # # r= D.dot(sol)-sol
        # # return adj_sol
        # return cont_adj_sol

    def _parameterized_raw_residual(
            self, sol, set_param_values, param_vals):
        set_param_values(self._fwd_solver.physics, param_vals)
        return self._fwd_solver.physics._raw_residual(sol)[0]
        #return self._fwd_solver.physics._residual(sol)[0]

    def compute_gradient(self, set_param_values, param_vals, return_obj=False,
                         **newton_kwargs):
        # use detach so that fwd_sol is not part of AD-graph
        set_param_values(self._fwd_solver.physics, param_vals.detach())
        fwd_sol = self._fwd_solver.solve(**newton_kwargs)
        adj_sol = self.solve_adjoint(fwd_sol, param_vals.detach())

        param_vals_copy = torch.clone(param_vals)
        fwd_sol_copy = torch.clone(fwd_sol)
        if self._dqdp is None:
            functional_vals = self._functional(
                fwd_sol_copy.requires_grad_(True),
                param_vals_copy).requires_grad_(True)
            functional_vals.backward()
            dqdp = param_vals_copy.grad
            if dqdp is None:
                dqdp = 0
        else:
            functional_vals = self._functional(fwd_sol_copy, param_vals_copy)
            dqdp = self._dqdp(fwd_sol_copy, param_vals_copy)

        if self._dRdp is None:
            dRdp = torch.autograd.functional.jacobian(
                partial(self._parameterized_raw_residual, fwd_sol,
                        set_param_values), param_vals, strict=True)
        else:
            dRdp = self._dRdp(
                self._fwd_solver.physics, fwd_sol_copy, param_vals)
        # print("dqdp", dqdp)
        # print(dRdp[:, 0], 'dRdp')
        # print(adj_sol, 'asol')
        # print(fwd_sol, 'fsol')
        grad = dqdp+torch.linalg.multi_dot((adj_sol[None, :], dRdp))
        if not return_obj:
            return grad
        return functional_vals.detach(), grad.detach()


class TransientAdjointPDE(TransientPDE):
    def __init__(self, physics, deltat, tableau_name, functional):
        if tableau_name != "im_beuler1":
            raise ValueError("Adjoints only supported for backward euler")
        super().__init__(physics, deltat, tableau_name)
        self._functional = functional

        self._adj_bndry_conds = copy.deepcopy(self.physics._bndry_conds)
        for bndry_cond in self._adj_bndry_conds:
            # for now only support dirichlet boundary conds
            assert bndry_cond[1] == "D"
            # bndry_cond[1] = "D"
            bndry_cond[0] = lambda xx: np.zeros((xx.shape[1], 1))

        # must change residual to adjoint residual and apply boundary conditions
        # to apply adjoint boundary conditions
        # self.adj_time_integrator = ImplicitRungeKutta(
        #     deltat, self.physics._transient_residual, tableau_name,
        #     constraints_fun=self._apply_boundary_conditions)

    def _dqdu(self, fwd_sols, time_index, param_vals):
        fwd_sols_copy = torch.clone(fwd_sols).requires_grad_(True)
        param_vals_copy = torch.clone(param_vals).requires_grad_(True)
        functional_vals = self._functional(fwd_sols_copy, param_vals_copy)
        functional_vals.backward()
        dqdu = fwd_sols_copy.grad
        return dqdu

    def solve_adjoint(self, fwd_sols, param_vals, times):
        adj_sols = torch.empty_like(fwd_sols)
        ndof, ntimes = fwd_sols.shape
        dqdu = self._dqdu(fwd_sols, ntimes-1, param_vals)
        adj_sols[:, ntimes-1] = 0
        Id = torch.eye(adj_sols.shape[0])
        for ii in range(ntimes-2, -1, -1):
            # print(ii, times[ii])
            jac = self.physics._transient_residual(
                fwd_sols[:, ii], times[ii])[1]
            deltat = times[ii+1]-times[ii]
            rhs = adj_sols[:, ii+1]+deltat*(-dqdu[:, ii+1])
            # need to pass in 0 instead of fwd_sol to apply boundary conditions
            # because we are not updating a residual rhs=sol-bndry_vals
            # but rather just want rhs=bndry_vals.
            # Techincally passing in we should be passing in negative bndry_vals
            # but since for dirichlet boundaries bndry_vals = 0 this is fine
            rhs, jac_adjoint = self.physics.mesh._apply_boundary_conditions(
                self._adj_bndry_conds, rhs, Id-deltat*jac.T, fwd_sols[:, ii]*0,
                self.physics.flux_jac)
            adj_sols[:, ii] = torch.linalg.solve(jac_adjoint, rhs)
            # print(rhs, 'rhs')
            # print(adj_sols[:, ii], 'phi', adj_sols[:, ii].sum())
        # print('a', adj_sols)
        return adj_sols

    # def solve_adjoint(self, fwd_sols, param_vals, times, verbosity=0,
    #                   newton_kwargs={}):
    #     init_sol = torch.zeros_like(fwd_sols[:, 0])
    #     adj_sols, times = self.adj_time_integrator.integrate(
    #         init_sol, times[0], times[-1], verbosity, newton_kwargs)
    #     return adj_sols

    def _parameterized_transient_raw_residual(
            self, sol, time, set_param_values, param_vals):
        set_param_values(self.physics, param_vals)
        return self.physics._transient_residual(sol, time)[0]

    def compute_gradient(self, init_sol, init_time, final_time,
                         set_param_values, param_vals, **newton_kwargs):
        set_param_values(self.physics, param_vals.detach())
        fwd_sols, times = self.solve(init_sol, init_time, final_time,
                                     newton_kwargs=newton_kwargs)
        adj_sols = self.solve_adjoint(fwd_sols, param_vals.detach(), times)

        param_vals_copy = torch.clone(param_vals).requires_grad_(True)
        fwd_sols_copy = torch.clone(fwd_sols).requires_grad_(True)
        functional_vals = self._functional(fwd_sols_copy, param_vals_copy)
        qoi = functional_vals.detach().numpy()
        # print(qoi, param_vals)
        # print(fwd_sols)
        # print(adj_sols)
        functional_vals.backward()
        dqdp = param_vals_copy.grad
        if dqdp is None:
            dqdp = 0

        phi_dRdp = 0
        for ii in range(fwd_sols.shape[1]-1):
            dRdp_ii = torch.autograd.functional.jacobian(
                partial(self._parameterized_transient_raw_residual,
                        fwd_sols[:, ii+1], times[ii], set_param_values),
                param_vals, strict=True)
            # print(ii, dRdp_ii.flatten(), 'dRdp')
            # print(adj_sols[:, ii:ii+1].T)
            phi_dRdp += torch.linalg.multi_dot(
                (adj_sols[:, ii:ii+1].T, dRdp_ii))
        grad = dqdp - phi_dRdp
        return np.atleast_1d(qoi), grad.numpy()
