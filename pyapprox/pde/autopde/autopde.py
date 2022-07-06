import torch
from abc import ABC, abstractmethod
import numpy as np
from functools import partial
from pyapprox.pde.autopde.mesh import VectorMesh


class AbstractSpectralCollocationResidual(ABC):
    def __init__(self, mesh, bndry_conds):
        self.mesh = mesh
        self._funs = None
        self._bndry_conds = self._set_boundary_conditions(
            bndry_conds)

    @abstractmethod
    def _raw_residual(self, sol):
        raise NotImplementedError()

    def _residual(self, sol):
        # correct equations for boundary conditions
        raw_residual = self._raw_residual(sol)
        return self.mesh._apply_boundary_conditions_to_residual(
            self._bndry_conds, raw_residual, sol)

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
            # print(time, hasattr(bndry_cond[0], "set_time"), bndry_cond[0], bndry_cond[0]._name)
            if hasattr(bndry_cond[0], "set_time"):
                bndry_cond[0].set_time(time)
        return self._raw_residual(sol)

    def _set_boundary_conditions(self, bndry_conds):
        # if len(self._bndrys) != len(bndry_conds):
        #     raise ValueError(
        #         "Incorrect number of boundary conditions provided")
        # for bndry_cond in bndry_conds:
        #     if bndry_cond[1] not in ["D", "R", "P", None]:
        #         raise ValueError(
        #             "Boundary condition {bndry_cond[1} not supported")
        #     if (bndry_cond[1] not in [None, "P"] and
        #         not callable(bndry_cond[0])):
        #         raise ValueError("Boundary condition must be callable")
        return bndry_conds


class AdvectionDiffusionReaction(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, bndry_conds, diff_fun, vel_fun, react_fun, forc_fun):
        super().__init__(mesh, bndry_conds)

        self._diff_fun = diff_fun
        self._vel_fun = vel_fun
        self._react_fun = react_fun
        self._forc_fun = forc_fun

        self._funs = [
            self._diff_fun, self._vel_fun, self._react_fun, self._forc_fun]

    @staticmethod
    def _check_shape(vals, ncols, name=None):
        if vals.ndim != 2 or vals.shape[1] != ncols:
            if name is not None:
                msg = name
            else:
                msg = "The ndarray"
            msg += f' has the wrong shape {vals.shape}'
            raise ValueError(msg)

    def _raw_residual(self, sol):
        # torch requires 1d sol to be a 1D tensor so Jacobian can be
        # computed correctly. But each other quantity must be a 2D tensor
        # with 1 column
        # To make sure sol is applied to both velocity components use
        # sol[:, None]
        diff_vals = self._diff_fun(self.mesh.mesh_pts)
        vel_vals = self._vel_fun(self.mesh.mesh_pts)
        forc_vals = self._forc_fun(self.mesh.mesh_pts)
        react_vals = self._react_fun(sol[:, None])
        self._check_shape(diff_vals, 1, "diff_vals")
        self._check_shape(forc_vals, 1, "forc_vals")
        self._check_shape(vel_vals, self.mesh.nphys_vars, "vel_vals")
        self._check_shape(react_vals, 1, "react_vals")
        residual = (self.mesh.div(diff_vals*self.mesh.grad(sol)) -
                    self.mesh.div(vel_vals*sol[:, None]) -
                    react_vals[:, 0]+forc_vals[:, 0])
        return residual


class EulerBernoulliBeam(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, bndry_conds, emod_fun, smom_fun, forc_fun):
        if mesh.nphys_vars > 1:
            raise ValueError("Only 1D meshes supported")

        super().__init__(mesh, bndry_conds)

        self._emod_fun = emod_fun
        self._smom_fun = smom_fun
        self._forc_fun = forc_fun

        self._emod_vals, self._smom_vals, self._forc_vals = (
            self._precompute_data())

    def _precompute_data(self):
        return (self._emod_fun(self.mesh.mesh_pts),
                self._smom_fun(self.mesh.mesh_pts),
                self._forc_fun(self.mesh.mesh_pts))

    def _raw_residual(self, sol):
        emod_vals = self._emod_fun(self.mesh.mesh_pts)
        smom_vals = self._smom_fun(self.mesh.mesh_pts)
        forc_vals = self._forc_fun(self.mesh.mesh_pts)
        pderiv = self.mesh.partial_deriv
        residual = 0
        residual = pderiv(pderiv(
            emod_vals[:, 0]*smom_vals[:, 0]*pderiv(pderiv(sol, 0), 0), 0), 0)
        residual -= forc_vals[:, 0]
        return residual

    def _residual(self, sol):
        pderiv = self.mesh.partial_deriv
        pderiv2 = partial(self.mesh.high_order_partial_deriv, 2)
        pderiv3 = partial(self.mesh.high_order_partial_deriv, 3)
        # correct equations for boundary conditions
        raw_residual = self._raw_residual(sol)
        raw_residual[0] = sol[0]-0
        raw_residual[1] = pderiv(sol, 0, [0])-0
        raw_residual[-1] = pderiv2(sol, 0, [-1])-0
        raw_residual[-2] = pderiv3(sol, 0, [-1])-0
        return raw_residual


class Helmholtz(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, bndry_conds, wnum_fun, forc_fun):
        super().__init__(mesh, bndry_conds)

        self._wnum_fun = wnum_fun
        self._forc_fun = forc_fun

    def _raw_residual(self, sol):
        wnum_vals = self._wnum_fun(self.mesh.mesh_pts)
        forc_vals = self._forc_fun(self.mesh.mesh_pts)
        residual = (self.mesh.laplace(sol) + wnum_vals[:, 0]*sol -
                    forc_vals[:, 0])
        return residual


class NavierStokes(AbstractSpectralCollocationResidual):
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
        for dd in range(self.mesh.nphys_vars):
            residual[dd] = (
                -self.mesh._meshes[dd].laplace(split_sols[dd]) +
                self.mesh._meshes[-1].partial_deriv(split_sols[-1], dd))
            residual[dd] -= vel_forc_vals[:, dd]
            if self._navier_stokes:
                residual[dd] += self.mesh._meshes[0].dot(
                    vel_sols, self.mesh._meshes[dd].grad(split_sols[dd]))
        residual[-1] = (
            self.mesh._meshes[-1].div(vel_sols) -
            self._pres_forc_fun(self.mesh._meshes[-1].mesh_pts)[:, 0])
        residual[-1][self._unique_pres_data[0]] = (
            split_sols[-1][self._unique_pres_data[0]]-self._unique_pres_data[1])
        return torch.cat(residual)


class LinearStokes(NavierStokes):
    def __init__(self, mesh, bndry_conds, vel_forc_fun, pres_forc_fun,
                 unique_pres_data=(0, 1)):
        super().__init__(mesh, bndry_conds, vel_forc_fun, pres_forc_fun,
                         unique_pres_data)
        self._navier_stokes = False


class ShallowWaterWave(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, bndry_conds, depth_forc_fun, vel_forc_fun,
                 bed_fun):
        super().__init__(mesh, bndry_conds)

        self._depth_forc_fun = depth_forc_fun
        self._vel_forc_fun = vel_forc_fun
        self._g = 9.81

        self._funs = [self._depth_forc_fun, self._vel_forc_fun]
        self._bed_vals = bed_fun(self.mesh._meshes[0].mesh_pts)

    def _raw_residual_1d(self, depth, vels, depth_forc_vals, vel_forc_vals):
        pderiv = self.mesh._meshes[0].partial_deriv
        residual = [0, 0]
        residual[0] = -pderiv(depth*vels[:, 0], 0)+depth_forc_vals[:, 0]
        residual[1] = -pderiv(depth*vels[:, 0]**2+self._g*depth**2/2, 0)
        residual[1] -= self._g*depth*pderiv(self._bed_vals[:, 0], 0)
        residual[1] += vel_forc_vals[:, 0]
        print(self._g*depth*pderiv(self._bed_vals[:, 0], 0))
        print(depth_forc_vals[:, 0], 'd')
        print(vel_forc_vals[:, 0], 'v')
        return torch.cat(residual)

    def _raw_residual_2d(self, depth, vels, depth_forc_vals, vel_forc_vals):
        pderiv = self.mesh._meshes[0].partial_deriv
        residual = [0, 0, 0]
         # depth equation (mass balance)
        for dd in range(self.mesh.nphys_vars):
            residual[0] -= self.mesh._meshes[0].partial_deriv(
                depth*vels[:, dd], dd)
        residual[0] += depth_forc_vals[:, 0]
        # velocity equations (momentum equations)
        residual[1] = -pderiv(depth*vels[:, 0]**2+self._g*depth**2/2, 0)
        residual[1] -= pderiv(depth*torch.prod(vels, dim=1), 1)
        residual[1] -= self._g*depth*pderiv(self._bed_vals[:, 0], 0)
        residual[1] += vel_forc_vals[:, 0]
        residual[2] = -pderiv(depth*torch.prod(vels, dim=1), 0)
        residual[2] -= pderiv(depth*vels[:, 1]**2+self._g*depth**2/2, 1)
        residual[2] -= self._g*depth*pderiv(self._bed_vals[:, 0], 1)
        residual[2] += vel_forc_vals[:, 1]
        return torch.cat(residual)

    def _raw_residual(self, sol):
        split_sols = self.mesh.split_quantities(sol)
        depth = split_sols[0]
        #vels = torch.hstack([s[:, None] for s in split_sols[1:]])
        vels = torch.hstack([(s/depth)[:, None] for s in split_sols[1:]])
        depth_forc_vals = self._depth_forc_fun(self.mesh._meshes[0].mesh_pts)
        vel_forc_vals = self._vel_forc_fun(self.mesh._meshes[1].mesh_pts)

        if self.mesh.nphys_vars == 1:
            # return self._raw_residual_1d(
            #     depth, vels, depth_forc_vals, vel_forc_vals)
            return self._raw_residual_1d(
                depth, vels, depth_forc_vals, vel_forc_vals)

        return self._raw_residual_2d(
                depth, vels, depth_forc_vals, vel_forc_vals)

        # residual = [0 for ii in range(len(split_sols))]
        # # depth equation (mass balance)
        # for dd in range(self.mesh.nphys_vars):
        #     # split_sols = [q1, q2] = [h, u, v]
        #     residual[0] += self.mesh._meshes[0].partial_deriv(
        #         depth*vels[:, dd], dd)
        # residual[0] -= depth_forc_vals[:, 0]
        # # velocity equations (momentum equations)
        # for dd in range(self.mesh.nphys_vars):
        #     # split_sols = [q1, q2] = [h, u, v]
        #     residual[dd+1] += self.mesh._meshes[dd].partial_deriv(
        #         depth*vels[:, dd]**2+self._g*depth**2/2, dd)
        #     # split_sols = [q1, q2] = [h, uh, vh]
        #     # residual[dd+1] += self.mesh._meshes[dd].partial_deriv(
        #     #     vels[:, dd]**2/depth+self._g*depth**2/2, dd)
        #     residual[dd+1] += self._g*depth*self.mesh._meshes[dd].partial_deriv(
        #         self._bed_vals[:, 0], dd)
        #     residual[dd+1] -= vel_forc_vals[:, dd]
        # if self.mesh.nphys_vars > 1:
        #     residual[1] += self.mesh._meshes[1].partial_deriv(
        #         depth*torch.prod(vels, dim=1), 1)
        #     residual[2] += self.mesh._meshes[2].partial_deriv(
        #         depth*torch.prod(vels, dim=1), 0)
        # return torch.cat(residual)


class ShallowShelfVelocities(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, bndry_conds, forc_fun, bed_fun, beta_fun,
                 depth_fun, A, rho, homotopy_val=0):
        super().__init__(mesh, bndry_conds)

        self._forc_fun = forc_fun
        self._A = A
        self._rho = rho
        self._homotopy_val = homotopy_val
        self._g = 9.81
        self._n = 3

        self._depth_fun = depth_fun
        self._funs = [self._forc_fun]
        self._bed_vals = bed_fun(self.mesh._meshes[0].mesh_pts)
        self._beta_vals = beta_fun(self.mesh._meshes[0].mesh_pts)
        self._forc_vals = self._forc_fun(self.mesh._meshes[0].mesh_pts)

    def _derivs(self, split_sols):
        pderiv = self.mesh._meshes[0].partial_deriv
        dudx_ij = []
        for ii in range(len(split_sols)):
            dudx_ij.append([])
            for jj in range(self.mesh.nphys_vars):
                dudx_ij[-1].append(pderiv(split_sols[ii], jj))
        return dudx_ij

    def _effective_strain_rate_1d(self, dudx_ij):
        return (dudx_ij[0][0]**2+self._homotopy_val)**(1/2)

    def _effective_strain_rate_2d(self, dudx_ij):
        return (dudx_ij[0][0]**2 + dudx_ij[1][1]**2+dudx_ij[0][0]*dudx_ij[1][1]
                + dudx_ij[0][1]**2/4+self._homotopy_val)**(1/2)

    def _effective_strain_rate(self, dudx_ij):
        if self.mesh.nphys_vars == 2:
            return self._effective_strain_rate_2d(dudx_ij)
        elif self.mesh.nphys_vars == 1:
            return self._effective_strain_rate_1d(dudx_ij)
        raise NotImplementedError()

    def _viscosity(self, dudx_ij):
        return (1/2*self._A**(-1/self._n) *
                self._effective_strain_rate(dudx_ij)**((1-self._n)/(self._n)))

    def _vector_components(self, dudx_ij):
        if self.mesh.nphys_vars == 2:
            vec1 = torch.hstack([(2*dudx_ij[0][0] + dudx_ij[1][1])[:, None],
                                 ((dudx_ij[0][1] + dudx_ij[1][0])/2)[:, None]])
            vec2 = torch.hstack([((dudx_ij[0][1] + dudx_ij[1][0])/2)[:, None],
                                 (dudx_ij[0][0] + 2*dudx_ij[1][1])[:, None]])
            return vec1, vec2
        return (2*dudx_ij[0][0][:, None],)

    def _raw_residual_nD(self, split_sols, depth_vals):
        pderiv = self.mesh._meshes[0].partial_deriv
        div = self.mesh._meshes[0].div
        dudx_ij = self._derivs(split_sols)
        visc = self._viscosity(dudx_ij)
        C = 2*visc*depth_vals[:, 0]
        vecs = self._vector_components(dudx_ij)
        residual = [0 for ii in range(self.mesh.nphys_vars)]
        for ii in range(self.mesh.nphys_vars):
            residual[ii] = -div(C[:, None]*vecs[ii])
            residual[ii] += self._beta_vals[:, 0]*split_sols[ii]
            residual[ii] += self._rho*self._g*depth_vals[:, 0]*pderiv(
                self._bed_vals[:, 0]+depth_vals[:, 0], ii)
            residual[ii] -= self._forc_vals[:, ii]
        return torch.cat(residual)

    def _raw_residual(self, sol):
        depth_vals = self._depth_fun(self.mesh._meshes[0].mesh_pts)
        split_sols = self.mesh.split_quantities(sol)
        return self._raw_residual_nD(split_sols, depth_vals)


class ShallowShelf(ShallowShelfVelocities):
    def __init__(self, mesh, bndry_conds, forc_fun, bed_fun, beta_fun,
                 depth_forc_fun, A, rho, homotopy_val=0):
        if len(mesh._meshes) != mesh._meshes[0].nphys_vars+1:
            raise ValueError("Incorrect number of meshes provided")

        super().__init__(mesh, bndry_conds, forc_fun, bed_fun, beta_fun,
                         None, A, rho, homotopy_val)
        self._depth_forc_fun = depth_forc_fun

    def _raw_residual(self, sol):
         # depth is 3rd mesh
        split_sols = self.mesh.split_quantities(sol)
        depth_vals = split_sols[-1]
        residual = super()._raw_residual_nD(
            split_sols[:-1], depth_vals[:, None])
        vel_vals = torch.hstack(
            [s[:, None] for s in split_sols[:self.mesh.nphys_vars]])
        depth_residual = -self.mesh._meshes[self.mesh.nphys_vars].div(
            depth_vals[:, None]*vel_vals)
        depth_residual += self._depth_forc_fun(
            self.mesh._meshes[self.mesh.nphys_vars].mesh_pts)[:, 0]
        return torch.cat((residual, depth_residual))


class NaviersLinearElasticity(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, bndry_conds, forc_fun, lambda_fun, mu_fun, rho):
        super().__init__(mesh, bndry_conds)

        self._rho = rho
        self._forc_fun = forc_fun
        self._lambda_fun = lambda_fun
        self._mu_fun = mu_fun

        # only needs to be time dependent funs
        self._funs = [self._forc_fun]

        # assumed to be time independent
        self._lambda_vals = self._lambda_fun(self.mesh._meshes[0].mesh_pts)
        self._mu_vals = self._mu_fun(self.mesh._meshes[0].mesh_pts)

        # sol is the displacement field
        # beam length L box cross section with width W
        # lambda = Lamae elasticity parameter
        # mu = Lamae elasticity parameter
        # rho density of beam
        # g acceleartion due to gravity

    def _raw_residual_1d(self, sol_vals, forc_vals):
        pderiv = self.mesh.meshes[0].partial_deriv
        residual = -pderiv(
            (self._lambda_vals[:, 0]+2*self._mu_vals[:, 0]) *
            pderiv(sol_vals[:, 0], 0), 0) - self._rho*forc_vals[:, 0]
        return residual

    def _raw_residual_2d(self, sol_vals, forc_vals):
        pderiv = self.mesh.meshes[0].partial_deriv
        residual = [0, 0]
        mu = self._mu_vals[:, 0]
        lam = self._lambda_vals[:, 0]
        lp2mu = lam+2*mu
        # strains
        exx = pderiv(sol_vals[:, 0], 0)
        eyy = pderiv(sol_vals[:, 1], 1)
        exy = 0.5*(pderiv(sol_vals[:, 0], 1)+pderiv(sol_vals[:, 1], 0))
        # stresses
        tauxy = 2*mu*exy
        tauxx = lp2mu*exx+lam*eyy
        tauyy = lam*exx+lp2mu*eyy
        residual[0] = pderiv(tauxx, 0)+pderiv(tauxy, 1)
        residual[0] += self._rho*forc_vals[:, 0]
        residual[1] = pderiv(tauxy, 0)+pderiv(tauyy, 1)
        residual[1] += self._rho*forc_vals[:, 1]
        return torch.cat(residual)

    def _raw_residual(self, sol):
        split_sols = self.mesh.split_quantities(sol)
        sol_vals = torch.hstack(
            [s[:, None] for s in split_sols[:self.mesh.nphys_vars]])
        forc_vals = self._forc_fun(self.mesh._meshes[0].mesh_pts)
        if self.mesh.nphys_vars == 1:
            return self._raw_residual_1d(sol_vals, forc_vals)
        return self._raw_residual_2d(sol_vals, forc_vals)


class FirstOrderStokesIce(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, bndry_conds, forc_fun, bed_fun, beta_fun,
                 depth_fun, A, rho, homotopy_val=0):
        super().__init__(mesh, bndry_conds)

        self._forc_fun = forc_fun
        self._A = A
        self._rho = rho
        self._homotopy_val = homotopy_val
        self._g = 9.81
        self._n = 3

        self._depth_fun = depth_fun
        self._funs = [self._forc_fun]
        self._bed_vals = bed_fun(self.mesh._meshes[0].mesh_pts[:-1])
        self._beta_vals = beta_fun(self.mesh._meshes[0].mesh_pts[:-1])
        self._forc_vals = self._forc_fun(self.mesh._meshes[0].mesh_pts)

        # for computing boundary conditions
        self._surface_vals = None
        self._vecs = None

    def _derivs(self, split_sols):
        pderiv = self.mesh._meshes[0].partial_deriv
        dudx_ij = []
        for ii in range(len(split_sols)):
            dudx_ij.append([])
            for jj in range(self.mesh.nphys_vars):
                dudx_ij[-1].append(pderiv(split_sols[ii], jj))
        return dudx_ij

    def _effective_strain_rate_xz(self, dudx_ij):
        return (dudx_ij[0][0]**2+dudx_ij[0][1]**2/4+self._homotopy_val)**(1/2)

    def _effective_strain_rate(self, dudx_ij):
        if self.mesh.nphys_vars == 2:
            return self._effective_strain_rate_xz(dudx_ij)
        raise NotImplementedError()

    def _viscosity(self, dudx_ij):
        return (1/2*self._A**(-1/self._n) *
                self._effective_strain_rate(dudx_ij)**((1-self._n)/(self._n)))

    def _vector_components(self, dudx_ij):
        if self.mesh.nphys_vars == 2:
            vals = (torch.hstack(
                [2*dudx_ij[0][0][:, None], dudx_ij[0][1][:, None]/2]), )
            return vals
        raise NotImplementedError()

    def _raw_residual_nD(self, split_sols, depth_vals):
        div = self.mesh._meshes[0].div
        dudx_ij = self._derivs(split_sols)
        visc = self._viscosity(dudx_ij)
        vecs = [2*visc[:, None]*(self._vector_components(dudx_ij)[ii])
                for ii in range(self.mesh.nphys_vars-1)]
        self._vecs = vecs
        residual = [0 for ii in range(self.mesh.nphys_vars-1)]
        for ii in range(self.mesh.nphys_vars-1):
            residual[ii] = -div(vecs[ii])
            # idx = self.mesh._meshes[0]._bndry_indices[3]
            # mesh_pts = self.mesh._meshes[0].mesh_pts[:, idx]
            # fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
            # self.mesh._meshes[0].plot(
            #     residual[ii].detach().numpy(), nplot_pts_1d=50, ax=axs[0])
            # self.mesh._meshes[0].plot(
            #     self._forc_vals[:, ii].detach().numpy(), nplot_pts_1d=50,
            #     ax=axs[1])
            # plt.plot(mesh_pts[0], self._forc_vals[idx, ii], '-s')
            # plt.plot(mesh_pts[0], residual[ii][idx], '--o')
            # plt.plot(mesh_pts[0], (vecs[0][idx, 0]), '--o')
            # plt.plot(mesh_pts[0], (visc[idx]), '--o')
            # plt.plot(mesh_pts[0], self._effective_strain_rate(dudx_ij)[idx], '--o')
            # plt.plot(mesh_pts[0], dudx_ij[0][0][idx], '--o'
            # plt.plot(mesh_pts[0], split_sols[0][idx], '--o')
            # print(np.abs(residual[ii]-self._forc_vals[:, ii]).max())
            # print(np.abs(residual[ii]-self._forc_vals[:, ii]).max()/np.linalg.norm(residual[ii]))
            # plt.show()
            residual[ii] -= self._forc_vals[:, ii]
        return torch.cat(residual)

    def _raw_residual(self, sol):
        depth_vals = self._depth_fun(self.mesh._meshes[0].mesh_pts[:-1])
        self._surface_vals = self._bed_vals+depth_vals
        split_sols = self.mesh.split_quantities(sol)
        return self._raw_residual_nD(split_sols, depth_vals)

    def _strain_boundary_conditions(self, sol, idx, mesh, bndry_index):
        normal_vals = mesh._bndrys[bndry_index].normals(mesh.mesh_pts[:, idx])
        vals = mesh.dot(self._vecs[0][idx, :], normal_vals)
        # if bndry_index < 2:
        #     normal0 = (-1)**((bndry_index+1) % 2)
        #     return self._vecs[0][idx, 0]*normal0
        # if bndry_index == 3:
        #     return vals
        if bndry_index == 2:
            return vals + self._beta_vals[idx, 0]*sol[idx]
        return vals


def vertical_transform_2D_mesh(xdomain_bounds, bed_fun, surface_fun,
                               canonical_samples):
    samples = np.empty_like(canonical_samples)
    xx, yy = canonical_samples[0], canonical_samples[1]
    samples[0] = (xx+1)/2*(
        xdomain_bounds[1]-xdomain_bounds[0])+xdomain_bounds[0]
    bed_vals = bed_fun(samples[0:1])[:, 0]
    samples[1] = (yy+1)/2*(surface_fun(samples[0:1])[:, 0]-bed_vals)+bed_vals
    return samples


def vertical_transform_2D_mesh_inv(xdomain_bounds, bed_fun, surface_fun,
                                   samples):
    canonical_samples = np.empty_like(samples)
    uu, vv = samples[0], samples[1]
    canonical_samples[0] = 2*(uu-xdomain_bounds[0])/(
        xdomain_bounds[1]-xdomain_bounds[0])-1
    bed_vals = bed_fun(samples[0:1])[:, 0]
    canonical_samples[1] = 2*(samples[1]-bed_vals)/(
        surface_fun(samples[0:1])[:, 0]-bed_vals)-1
    return canonical_samples


def vertical_transform_2D_mesh_inv_dxdu(xdomain_bounds, samples):
    return np.full(samples.shape[1], 2/(xdomain_bounds[1]-xdomain_bounds[0]))


def vertical_transform_2D_mesh_inv_dydu(
        bed_fun, surface_fun, bed_grad_u, surf_grad_u, samples):
    surf_vals = surface_fun(samples[:1])[:, 0]
    bed_vals = bed_fun(samples[:1])[:, 0]
    return 2*(bed_grad_u(samples[:1])[:, 0]*(samples[1]-surf_vals) +
              surf_grad_u(samples[:1])[:, 0]*(bed_vals-samples[1]))/(
                  surf_vals-bed_vals)**2


def vertical_transform_2D_mesh_inv_dxdv(samples):
    return np.zeros(samples.shape[1])


def vertical_transform_2D_mesh_inv_dydv(bed_fun, surface_fun, samples):
    surf_vals = surface_fun(samples[:1])[:, 0]
    bed_vals = bed_fun(samples[:1])[:, 0]
    return 2/(surf_vals-bed_vals)
