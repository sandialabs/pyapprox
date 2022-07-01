import torch
import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from torch.linalg import multi_dot
from pyapprox.pde.autopde.autopde import VectorMesh


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
        if type(self.mesh) == VectorMesh:
            import itertools
            bndry_conds = itertools.chain(*self._bndry_conds)
        else:
            bndry_conds = self._bndry_conds
        for bndry_cond in bndry_conds:
            print(time, hasattr(bndry_cond[0], "set_time"), bndry_cond[0])
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


class IncompressibleNavierStokes(AbstractSpectralCollocationPhysics):
    def __init__(self, mesh, bndry_conds, vel_forc_fun, pres_forc_fun,
                 unique_pres_data=(0, 1)):
        super().__init__(mesh, bndry_conds)

        self._navier_stokes = True
        self._vel_forc_fun = vel_forc_fun
        self._pres_forc_fun = pres_forc_fun
        self._unique_pres_data = unique_pres_data

    def _raw_residual(self, sol):
        split_sols = self.mesh.split_quantities(sol)
        vel_sols = torch.hstack([s[:, None] for s in split_sols[:-1]])
        vel_forc_vals = self._vel_forc_fun(self.mesh._meshes[0].mesh_pts)
        residual = [None for ii in range(len(split_sols))]
        jac = [[0 for jj in range(self.mesh.nphys_vars+1)]
               for ii in range(len(split_sols))]
        # assumes x and y velocity meshes are the same
        vel_dmats = [self.mesh._meshes[0]._dmat(dd)
                 for dd in range(self.mesh.nphys_vars)]
        for dd in range(self.mesh.nphys_vars):
            residual[dd] = (
                -self.mesh._meshes[dd].laplace(split_sols[dd]) +
                self.mesh._meshes[-1].partial_deriv(split_sols[-1], dd))
            residual[dd] -= vel_forc_vals[:, dd]
            for ii in range(self.mesh.nphys_vars):
                dmat = vel_dmats[ii] # self.mesh._meshes[dd]._dmat(ii)
                jac[dd][dd] += -multi_dot((dmat, dmat))
                if dd != ii:
                    jac[dd][ii] = torch.zeros_like(dmat)
            jac[dd][-1] = self.mesh._meshes[-1]._dmat(split_sols[-1], dd)
            if self._navier_stokes:
                residual[dd] += self.mesh._meshes[0].dot(
                    vel_sols, self.mesh._meshes[dd].grad(split_sols[dd]))
                # 1D
                # residual[0] += v[0]*(D[0]v[0])
                # d residual[0]/dv[0] += diag(D[0]v[dd]) + diag(v[0])*D[0]
                # 2D
                # residual[0] = v[0]*(D[0]v[0]) + v[1]*(D[1]v[0])
                # d residual[0] /dv[0] = diag(D[0]v[0]) + diag(v[0])D[0] + diag(v[1])D[1]
                # d residual[0] /dv[1] = diag(D[1]v[0])
                # residual[1] = v[0]*(D[0]v[1]) + v[1]*(D[1]v[1])
                # d residual[1] /dv[0] = diag(D[0]v[1])
                # d residual[1] /dv[1] = diag(v[0])D[0] + diag(D[1]v[1]) + diag(v[1])D[1]
                for ii in range(self.mesh.nphys_vars):
                    if ii != dd:
                        jac[dd][ii] += torch.diag(
                            multi_dot((vel_dmats[ii], split_sols[dd])))
                    jac[dd][dd] += split_sols[ii][:, None]*vel_dmats[ii]
                jac[dd][dd] += torch.diag(
                     multi_dot((vel_dmats[dd], split_sols[dd])))
            jac[dd] = torch.hstack(jac[dd])
        residual[-1] = (
            self.mesh._meshes[-1].div(vel_sols) -
            self._pres_forc_fun(self.mesh._meshes[-1].mesh_pts)[:, 0])
        for dd in range(self.mesh.nphys_vars):
            jac[-1][dd] = self.mesh._meshes[-1]._dmat(split_sols[dd], dd)
        jac[-1][-1] = torch.zeros(
            (self.mesh._meshes[-1].nunknowns, self.mesh._meshes[-1].nunknowns))
        jac[-1] = torch.hstack(jac[-1])
        residual[-1][self._unique_pres_data[0]] = (
            split_sols[-1][self._unique_pres_data[0]]-self._unique_pres_data[1])
        jac[-1][self._unique_pres_data[0], :] = 0
        jac[-1][self._unique_pres_data[0],
                self.mesh._meshes[0].nunknowns*self.mesh.nphys_vars +
                self._unique_pres_data[0]] = 1
        # Todo reverse sign of residual for time integration
        return torch.cat(residual), torch.vstack(jac)


class LinearIncompressibleStokes(IncompressibleNavierStokes):
    def __init__(self, mesh, bndry_conds, vel_forc_fun, pres_forc_fun,
                 unique_pres_data=(0, 1)):
        super().__init__(mesh, bndry_conds, vel_forc_fun, pres_forc_fun,
                         unique_pres_data)
        self._navier_stokes = False


class ShallowIce(AbstractSpectralCollocationPhysics):
    def __init__(self, mesh, bndry_conds, bed_fun, beta_fun, forc_fun,
                 A, rho, n=3, g=9.81):
        super().__init__(mesh, bndry_conds)

        self._bed_fun = bed_fun
        self._beta_fun = beta_fun
        self._forc_fun = forc_fun
        self._A = A
        self._rho = rho
        self._n = n
        self._g = g
        self._gamma = 2*A*(rho*g)**n/(n+2)
        self._eps = 1e-15 

        self._funs = [
            self._bed_fun, self._beta_fun, self._forc_fun]

    def _raw_residual(self, sol):
        bed_vals = self._bed_fun(self.mesh.mesh_pts)[:, 0]
        beta_vals = self._beta_fun(self.mesh.mesh_pts)[:, 0]
        surf_grad = self.mesh.grad(bed_vals+sol)
        res = self.mesh.div((self._gamma*sol[:, None]**(self._n+2)*(surf_grad**2+self._eps)**((self._n-1)/2) +
                            (1/(beta_vals)*self._rho*self._g*sol**2)[:, None])*surf_grad)
        # res1 = self.mesh.partial_deriv(
        #     ((self._gamma*sol**(self._n+2)*(surf_grad[:, 0]**2+self._eps)**((self._n-1)/2))*surf_grad[:, 0]), 0)
        # res2 = self.mesh.partial_deriv(((1/(beta_vals)*self._rho*self._g*sol**2))*surf_grad[:, 0], 0)
        res += self._forc_fun(self.mesh.mesh_pts)[:, 0]
        return res # res1+res2 + self._forc_fun(self.mesh.mesh_pts)[:, 0]

    def _raw_jacobian(self, sol):
        # C = g*h/beta
        # g(h) = C*h**2*D[0](h+b)
        # g(h) = C*h**2*D[0]h+C*h**2*D[0]b
        h, b = sol, self._bed_fun(self.mesh.mesh_pts)[:, 0]
        C = self._rho*self._g*h/self._beta_fun(self.mesh.mesh_pts)[:, 0]
        dmats = [self.mesh._dmat(dd) for dd in range(self.mesh.nphys_vars)]
        jac1, jac2 = 0, 0
        for dd in range(self.mesh.nphys_vars):
            jac2 += (2*C*multi_dot(
                (dmats[dd], torch.diag(h*multi_dot((dmats[dd], (h+b)))))) +
                     C*multi_dot((dmats[dd], (h[:, None]**2*(dmats[dd])))))
            surf_grad = self.mesh.grad(b+h)
            # multiplying A*b[:, None] is equivlanet to dot(diag(b), A)
            # multiplying A*b[None, :] is equivlanet to dot(A, diag(b))
            jac1 += (self._n+2)*self._gamma*dmats[dd]*((
                (h**(self._n+1)*(surf_grad[:, dd]**2+self._eps)**((self._n-1)/2)*surf_grad[:, dd]))[None, :])
            # d/dx((h(x)^2)^((n - 1)/2)) = (n - 1) h(x) (h(x)^2)^((n - 3)/2) h'(x)
            tmp = (surf_grad[:, dd]*(
                self._n-1)*surf_grad[:, dd]*(surf_grad[:, dd]**2+self._eps)**(
                    (self._n-3)/2))[:, None]*(dmats[dd])
            tmp += ((surf_grad[:, dd]**2+self._eps)**((self._n-1)/2))[:, None]*dmats[dd]
            jac1 += self._gamma*multi_dot((dmats[dd], (h[:, None]**(self._n+2)*(tmp))))
        jac = jac1 + jac2
        return jac



    # useful checks for computing jacobians
    # res = multi_dot((dmats[0], sol**2))
    # jac = 2*multi_dot((dmats[0], torch.diag(h))) = 2*dmats[0]*h[None, :]
    # res = multi_dot((dmats[0], sol**2*multi_dot((dmats[0], sol))))
    # jac = (
    #     2*multi_dot((dmats[0], torch.diag(h[:, None]*multi_dot((dmats[0], h)))))
    #     +multi_dot((dmats[0], (h[:, None]**2*(dmats[0])))))
