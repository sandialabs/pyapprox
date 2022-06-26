import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from torch.linalg import multi_dot


class AbstractSpectralCollocationPhysics(ABC):
    def __init__(self, mesh, bndry_conds):
        self.mesh = mesh
        self._funs = None
        self._bndry_conds = self._set_boundary_conditions(
            bndry_conds)

    def _set_boundary_conditions(self, bndry_conds):
        # TODO add input checks
        return bndry_conds

    @abstractmethod
    def _raw_residual(self, sol):
        raise NotImplementedError()

    def _residual(self, sol):
        res, jac = self._raw_residual(sol)
        res, jac = self.mesh._apply_boundary_conditions(
            self._bndry_conds, res, jac, sol)
        return res, jac

    def _transient_residual(self, sol, time):
        # correct equations for boundary conditions
        for fun in self._funs:
            if hasattr(fun, "set_time"):
                fun.set_time(time)
        for bndry_cond in self._bndry_conds:
            if hasattr(bndry_cond[0], "set_time"):
                bndry_cond[0].set_time(time)
        return self._raw_residual(sol)


class AdvectionDiffusionReaction(AbstractSpectralCollocationPhysics):
    def __init__(self, mesh, bndry_conds, diff_fun, vel_fun, react_fun,
                 forc_fun, react_jac):
        super().__init__(mesh, bndry_conds)

        self._diff_fun = diff_fun
        self._vel_fun = vel_fun
        self._react_fun = react_fun
        self._forc_fun = forc_fun
        self._react_jac = react_jac

        self._funs = [
            self._diff_fun, self._vel_fun, self._react_fun, self._forc_fun]

    def _raw_residual(self, sol):
        diff_vals = self._diff_fun(self.mesh.mesh_pts)
        vel_vals = self._vel_fun(self.mesh.mesh_pts)
        linear_jac = 0
        for dd in range(self.mesh.nphys_vars):
            linear_jac += (
                multi_dot(
                    (self.mesh._dmat(dd), diff_vals*self.mesh._dmat(dd))) -
                vel_vals[:, dd:dd+1]*self.mesh._dmat(dd))
        res = multi_dot((linear_jac, sol))
        jac = linear_jac - self._react_jac(sol[:, None])
        res -= self._react_fun(sol[:, None])[:, 0]
        res += self._forc_fun(self.mesh.mesh_pts)[:, 0]
        return res, jac
