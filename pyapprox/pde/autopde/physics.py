import torch
import itertools
from abc import ABC, abstractmethod
from torch.linalg import multi_dot
from pyapprox.pde.autopde.mesh import VectorMesh
from functools import partial


class AbstractSpectralCollocationPhysics(ABC):
    def __init__(self, mesh, bndry_conds):
        self.mesh = mesh
        self._funs = None
        self._bndry_conds = self._set_boundary_conditions(
            bndry_conds)
        self._auto_jac = True

    def _set_boundary_conditions(self, bndry_conds):
        if type(self.mesh) == VectorMesh:
            if len(bndry_conds) != len(self.mesh._meshes):
                msg = "Boundary conditions must be provided for each mesh"
                raise ValueError(msg)
            for ii in range(len(self.mesh._meshes)):
                if len(bndry_conds[ii]) != len(self.mesh._meshes[ii]._bndrys):
                    msg = "Boundary conditions must be provided for each "
                    msg += "boundary"
                    raise ValueError(msg)
            return bndry_conds

        if len(bndry_conds) != len(self.mesh._bndrys):
            msg = "Boundary conditions must be provided for each "
            msg += "boundary"
            raise ValueError(msg)
        return bndry_conds

    @abstractmethod
    def _raw_residual(self, sol):
        raise NotImplementedError()

    def _residual(self, sol):
        if sol.ndim != 1:
            raise ValueError("sol must be 1D tensor")
        res, jac = self._raw_residual(sol)
        if jac is None:
            assert self._auto_jac
        res, jac = self.mesh._apply_boundary_conditions(
            self._bndry_conds, res, jac, sol)
        return res, jac

    def _transient_residual(self, sol, time):
        # correct equations for boundary conditions
        for fun in self._funs:
            if hasattr(fun, "set_time"):
                fun.set_time(time)
        if type(self.mesh) == VectorMesh:
            bndry_conds = itertools.chain(*self._bndry_conds)
        else:
            bndry_conds = self._bndry_conds
        for bndry_cond in bndry_conds:
            if hasattr(bndry_cond[0], "set_time"):
                bndry_cond[0].set_time(time)
        res, jac = self._raw_residual(sol)
        if jac is None:
            assert self._auto_jac
        return res, jac


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

        self._auto_jac = False

    # def _parameterized_raw_residual(self, sol, params):
    #     ndof = sol.shape[0]
    #     diff_vals = params[:ndof]
    #     forc_vals = params[ndof:2*ndof]
    #     vel_vals = params[2*ndof:].reshape((ndof, 2))

    def _raw_residual(self, sol):
        diff_vals = self._diff_fun(self.mesh.mesh_pts)
        vel_vals = self._vel_fun(self.mesh.mesh_pts)
        linear_jac = 0
        for dd in range(self.mesh.nphys_vars):
            linear_jac += (
                multi_dot(
                    (self.mesh._dmats[dd], diff_vals*self.mesh._dmats[dd])) -
                vel_vals[:, dd:dd+1]*self.mesh._dmats[dd])
        res = multi_dot((linear_jac, sol))
        jac = linear_jac - self._react_jac(sol[:, None])
        res -= self._react_fun(sol[:, None])[:, 0]
        res += self._forc_fun(self.mesh.mesh_pts)[:, 0]
        # print(res)
        # print(self._forc_fun(self.mesh.mesh_pts)[:, 0])
        if not self._auto_jac:
            return res, jac
        return res, None


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
        # assumes x and y velocity meshes are the same
        for dd in range(self.mesh.nphys_vars):
            residual[dd] = (
                self.mesh._meshes[dd].laplace(split_sols[dd]) -
                self.mesh._meshes[-1].partial_deriv(split_sols[-1], dd))
            residual[dd] += vel_forc_vals[:, dd]
            if self._navier_stokes:
                residual[dd] -= self.mesh._meshes[0].dot(
                    vel_sols, self.mesh._meshes[dd].grad(split_sols[dd]))
        residual[-1] = (
            -self.mesh._meshes[-1].div(vel_sols) +
            self._pres_forc_fun(self.mesh._meshes[-1].mesh_pts)[:, 0])
        residual[-1][self._unique_pres_data[0]] = (
            split_sols[-1][self._unique_pres_data[0]]-self._unique_pres_data[1])
        # Todo reverse sign of residual for time integration
        return torch.cat(residual), self._raw_jacobian(sol)

    def _raw_jacobian(self, sol):
        split_sols = self.mesh.split_quantities(sol)
        vel_sols = torch.hstack([s[:, None] for s in split_sols[:-1]])
        vel_forc_vals = self._vel_forc_fun(self.mesh._meshes[0].mesh_pts)
        jac = [[0 for jj in range(self.mesh.nphys_vars+1)]
               for ii in range(len(split_sols))]
        # assumes x and y velocity meshes are the same
        vel_dmats = self.mesh._meshes[0]._dmats
        #[self.mesh._meshes[0]._dmat(dd)
        #for dd in range(self.mesh.nphys_vars)]
        for dd in range(self.mesh.nphys_vars):
            for ii in range(self.mesh.nphys_vars):
                dmat = vel_dmats[ii] # self.mesh._meshes[dd]._dmat(ii)
                jac[dd][dd] += multi_dot((dmat, dmat))
                if dd != ii:
                    jac[dd][ii] = torch.zeros_like(dmat)
            jac[dd][-1] = -self.mesh._meshes[-1]._dmat(split_sols[-1], dd)
            if self._navier_stokes:
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
                        jac[dd][ii] -= torch.diag(
                            multi_dot((vel_dmats[ii], split_sols[dd])))
                    jac[dd][dd] -= split_sols[ii][:, None]*vel_dmats[ii]
                jac[dd][dd] -= torch.diag(
                     multi_dot((vel_dmats[dd], split_sols[dd])))
            jac[dd] = torch.hstack(jac[dd])
        for dd in range(self.mesh.nphys_vars):
            jac[-1][dd] = -self.mesh._meshes[-1]._dmat(split_sols[dd], dd)
        jac[-1][-1] = torch.zeros(
            (self.mesh._meshes[-1].nunknowns, self.mesh._meshes[-1].nunknowns))
        jac[-1] = torch.hstack(jac[-1])
        jac[-1][self._unique_pres_data[0], :] = 0
        jac[-1][self._unique_pres_data[0],
                self.mesh._meshes[0].nunknowns*self.mesh.nphys_vars +
                self._unique_pres_data[0]] = 1
        # Todo reverse sign of residual for time integration
        return torch.vstack(jac)


class LinearIncompressibleStokes(IncompressibleNavierStokes):
    def __init__(self, mesh, bndry_conds, vel_forc_fun, pres_forc_fun,
                 unique_pres_data=(0, 1)):
        super().__init__(mesh, bndry_conds, vel_forc_fun, pres_forc_fun,
                         unique_pres_data)
        self._navier_stokes = False


class ShallowIce(AbstractSpectralCollocationPhysics):
    def __init__(self, mesh, bndry_conds, bed_fun, beta_fun, forc_fun,
                 A, rho, n=3, g=9.81, eps=1e-15):
        super().__init__(mesh, bndry_conds)

        self._bed_fun = bed_fun
        self._beta_fun = beta_fun
        self._forc_fun = forc_fun
        self._A = A
        self._rho = rho
        self._n = n
        self._g = g
        self._gamma = 2*A*(rho*g)**n/(n+2)
        self._eps = eps

        self._funs = [
            self._bed_fun, self._beta_fun, self._forc_fun]

    def _raw_residual(self, sol):
        if torch.any(sol <= 0):
            print(sol.detach().numpy())
            raise RuntimeError("Depth is negative")
        bed_vals = self._bed_fun(self.mesh.mesh_pts)[:, 0]
        beta_vals = self._beta_fun(self.mesh.mesh_pts)[:, 0]
        surf_grad = self.mesh.grad(bed_vals+sol)
        res = self.mesh.div(
            (self._gamma*sol[:, None]**(self._n+2)*(surf_grad**2+self._eps)**(
                (self._n-1)/2) +
             (1/(beta_vals)*self._rho*self._g*sol**2)[:, None])*surf_grad)
        res += self._forc_fun(self.mesh.mesh_pts)[:, 0]
        return res, self._raw_jacobian(sol)

    def _raw_jacobian(self, sol):
        # C = g*h/beta
        # g(h) = C*h**2*D[0](h+b)
        # g(h) = 2C*h*D[0]*h + C*h**2*D[0]
        h, b = sol, self._bed_fun(self.mesh.mesh_pts)[:, 0]
        C = self._rho*self._g*h/self._beta_fun(self.mesh.mesh_pts)[:, 0]
        # dmats = [self.mesh._dmat(dd) for dd in range(self.mesh.nphys_vars)]
        dmats = self.mesh._dmats
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


class EulerBernoulliBeam(AbstractSpectralCollocationPhysics):
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
        return residual, None

    def _residual(self, sol):
        pderiv = self.mesh.partial_deriv
        pderiv2 = partial(self.mesh.high_order_partial_deriv, 2)
        pderiv3 = partial(self.mesh.high_order_partial_deriv, 3)
        # correct equations for boundary conditions
        raw_residual, raw_jac = self._raw_residual(sol)
        raw_residual[0] = sol[0]-0
        raw_residual[1] = pderiv(sol, 0, [0])-0
        raw_residual[-1] = pderiv2(sol, 0, [-1])-0
        raw_residual[-2] = pderiv3(sol, 0, [-1])-0
        return raw_residual, raw_jac


class Helmholtz(AbstractSpectralCollocationPhysics):
    def __init__(self, mesh, bndry_conds, wnum_fun, forc_fun):
        super().__init__(mesh, bndry_conds)

        self._wnum_fun = wnum_fun
        self._forc_fun = forc_fun

    def _raw_residual(self, sol):
        wnum_vals = self._wnum_fun(self.mesh.mesh_pts)
        forc_vals = self._forc_fun(self.mesh.mesh_pts)
        residual = (self.mesh.laplace(sol) + wnum_vals[:, 0]*sol -
                    forc_vals[:, 0])
        return residual, None


class ShallowWaterWave(AbstractSpectralCollocationPhysics):
    def __init__(self, mesh, bndry_conds, depth_forc_fun, vel_forc_fun,
                 bed_fun):
        super().__init__(mesh, bndry_conds)

        self._depth_forc_fun = depth_forc_fun
        self._vel_forc_fun = vel_forc_fun
        self._g = 9.81

        self._funs = [self._depth_forc_fun, self._vel_forc_fun]
        self._bed_vals = bed_fun(self.mesh._meshes[0].mesh_pts)

        self._auto_jac = False

    def _raw_residual_1d(self, depth, vels, depth_forc_vals, vel_forc_vals):
        pderiv = self.mesh._meshes[0].partial_deriv
        residual = [0, 0]
        residual[0] = -pderiv(depth*vels[:, 0], 0)+depth_forc_vals[:, 0]
        residual[1] = -pderiv(depth*vels[:, 0]**2+self._g*depth**2/2, 0)
        residual[1] -= self._g*depth*pderiv(self._bed_vals[:, 0], 0)
        residual[1] += vel_forc_vals[:, 0]
        if not self._auto_jac:
            return torch.cat(residual), self._raw_jacobian_1d(depth, vels)
        return torch.cat(residual), None

    def _raw_jacobian_1d(self, depth, vels):
        # dmats = [self.mesh._meshes[0]._dmat(dd)
        #          for dd in range(self.mesh.nphys_vars)]
        dmats = self.mesh._meshes[0]._dmats
        jac = [0, 0]
        # recall taking jac with respect to u and uh
        jac[0] = [-dmats[0]*0, -dmats[0]]
        # r[1] = -D[0](uh**2/h) - g*D[0]*h*2/2
        jac[1] = [dmats[0]*(vels[:, 0][None, :]**2)-dmats[0]*(self._g*depth[None, :]),
                  -2*dmats[0]*((vels[:, 0])[None, :])]
        jac[1][0] -= self._g*self.mesh._meshes[0].partial_deriv(
            self._bed_vals[:, 0], 0)
        return torch.vstack([torch.hstack(j) for j in jac])

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
        if not self._auto_jac:
            return torch.cat(residual), self._raw_jacobian_2d(depth, vels)
        return torch.cat(residual), None

    def _raw_jacobian_2d(self, depth, vels):
        # dmats = [self.mesh._meshes[0]._dmat(dd)
        #          for dd in range(self.mesh.nphys_vars)]
        dmats = self.mesh._meshes[0]._dmats
        jac = [0, 0, 0]
        # recall taking jac with respect to u and uh
        jac[0] = [-dmats[0]*0, -dmats[0], -dmats[1]]
        # r[1] = -D[0](uh**2/h) - g*D[0]*h*2/2 - D[1]*(uh*vh/h)
        jac[1] = [
            dmats[0]*(vels[:, 0][None, :]**2)-dmats[0]*(self._g*depth[None, :])
            + dmats[1]*((vels[:, 0]*vels[:, 1])[None, :]),
            - 2*dmats[0]*((vels[:, 0])[None, :])
            - dmats[1]*(vels[:, 1][None, :]),
            - dmats[1]*(vels[:, 0][None, :])]
        jac[2] = [
            dmats[0]*((vels[:, 0]*vels[:, 1])[None, :])
            + dmats[1]*(vels[:, 1][None, :]**2)
            - dmats[1]*(self._g*depth[None, :]),
            -dmats[0]*(vels[:, 1][None, :]),
            -2*dmats[1]*((vels[:, 1])[None, :])-dmats[0]*(vels[:, 0][None, :])]
        jac[1][0] -= self._g*self.mesh._meshes[0].partial_deriv(
            self._bed_vals[:, 0], 0)
        jac[2][0] -= self._g*self.mesh._meshes[0].partial_deriv(
            self._bed_vals[:, 0], 1)
        return torch.vstack([torch.hstack(j) for j in jac])

    def _raw_residual(self, sol):
        split_sols = self.mesh.split_quantities(sol)
        depth = split_sols[0]
        if torch.any(depth <= 0):
            raise RuntimeError("Depth is negative")
        vels = torch.hstack([(s/depth)[:, None] for s in split_sols[1:]])
        depth_forc_vals = self._depth_forc_fun(self.mesh._meshes[0].mesh_pts)
        vel_forc_vals = self._vel_forc_fun(self.mesh._meshes[1].mesh_pts)

        if self.mesh.nphys_vars == 1:
            return self._raw_residual_1d(
                depth, vels, depth_forc_vals, vel_forc_vals)

        return self._raw_residual_2d(
                depth, vels, depth_forc_vals, vel_forc_vals)


class ShallowShelfVelocities(AbstractSpectralCollocationPhysics):
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

        self._auto_jac = False

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
                + (dudx_ij[0][1]+dudx_ij[1][0])**2/4+self._homotopy_val)**(1/2)

    def _effective_strain_rate(self, dudx_ij):
        if self.mesh.nphys_vars == 2:
            return self._effective_strain_rate_2d(dudx_ij)
        elif self.mesh.nphys_vars == 1:
            return self._effective_strain_rate_1d(dudx_ij)
        raise NotImplementedError()

    def _effective_strain_rate_jac(self, dudx_ij):
        # d/dx sqrt(g(x)) =dg/df(g(x)) df/dx(x)
        # d/du (ux**2 + vy**2 + ux*vy + 0.25*(uy+vx)**2)**(1/2)
        # = 0.5*srate**(-0.5)*d/du(ux**2+...)
        srate = self._effective_strain_rate(dudx_ij)
        # dmats = [self.mesh._meshes[0]._dmat(dd)
        #          for dd in range(self.mesh.nphys_vars)]
        dmats = self.mesh._meshes[0]._dmats
        tmp = [2*dudx_ij[dd][dd][:, None]*dmats[dd]
               for dd in range(self.mesh.nphys_vars)]
        if self.mesh.nphys_vars == 2:
            tmp[0] += (
                (dudx_ij[1][1][:, None])*dmats[0] +
                0.25*(2*(dudx_ij[0][1]+dudx_ij[1][0])[:, None]*dmats[1]))
            tmp[1] += (dudx_ij[0][0][:, None])*dmats[1]+0.25*(2*(
                dudx_ij[1][0]+dudx_ij[0][1])[:, None]*dmats[0])
        return [.5/srate[:, None]*t for t in tmp]

    def _viscosity(self, dudx_ij):
        return (1/2*self._A**(-1/self._n) *
                self._effective_strain_rate(dudx_ij)**((1-self._n)/(self._n)))

    def _viscosity_jac(self, dudx_ij):
        sjac = self._effective_strain_rate_jac(dudx_ij)
        tmp = (1/2*self._A**(-1/self._n) *
               self._effective_strain_rate(dudx_ij)**((1-2*self._n)/(self._n)))
        jac = [(1-self._n)/(self._n)*tmp[:, None]*j for j in sjac]
        return jac

    def _vector_components(self, dudx_ij):
        if self.mesh.nphys_vars == 2:
            vec1 = torch.hstack([(2*dudx_ij[0][0] + dudx_ij[1][1])[:, None],
                                 ((dudx_ij[0][1] + dudx_ij[1][0])/2)[:, None]])
            vec2 = torch.hstack([((dudx_ij[0][1] + dudx_ij[1][0])/2)[:, None],
                                 (dudx_ij[0][0] + 2*dudx_ij[1][1])[:, None]])
            return vec1, vec2
        return (2*dudx_ij[0][0][:, None],)

    def _vector_components_jac(self, dudx_ij):
        # dmats = [self.mesh._meshes[0]._dmat(dd)
        #          for dd in range(self.mesh.nphys_vars)]
        dmats = self.mesh._meshes[0]._dmats
        if self.mesh.nphys_vars == 1:
            return ([[2*dmats[0]]], )
        # vec1_jac = [[dv1[0]/du, dv1[1]/du], dv1[0]/dv, dv[1]/dv]
        vec1_jac = [[2*dmats[0], dmats[1]/2], [dmats[1], dmats[0]/2]]
        vec2_jac = [[dmats[1]/2, dmats[0]], [dmats[0]/2, 2*dmats[1]]]
        return vec1_jac, vec2_jac

    def _raw_residual_nD(self, split_sols, depth_vals):
        pderiv = self.mesh._meshes[0].partial_deriv
        div = self.mesh._meshes[0].div
        dudx_ij = self._derivs(split_sols)
        visc = self._viscosity(dudx_ij)
        C = 2*visc*depth_vals[:, 0]
        vecs = self._vector_components(dudx_ij)
        residual = [0 for ii in range(self.mesh.nphys_vars)]
        for ii in range(self.mesh.nphys_vars):
            residual[ii] = div(C[:, None]*vecs[ii])
            residual[ii] -= self._beta_vals[:, 0]*split_sols[ii]
            residual[ii] -= self._rho*self._g*depth_vals[:, 0]*pderiv(
                self._bed_vals[:, 0]+depth_vals[:, 0], ii)
            residual[ii] += self._forc_vals[:, ii]
        if self._auto_jac is True:
            return torch.cat(residual), None
        return torch.cat(residual), self._raw_jacobian_nD(
            split_sols, depth_vals)

    def _raw_jacobian_nD(self, split_sols, depth_vals):
        dudx_ij = self._derivs(split_sols)
        visc = self._viscosity(dudx_ij)
        visc_jac = self._viscosity_jac(dudx_ij)
        vecs = self._vector_components(dudx_ij)
        vecs_jac = self._vector_components_jac(dudx_ij)
        # dmats = [self.mesh._meshes[0]._dmat(dd)
        #          for dd in range(self.mesh.nphys_vars)]
        dmats = self.mesh._meshes[0]._dmats
        jac = []
        # loop over jacobian rows (blocks)
        for ii in range(self.mesh.nphys_vars):
            jac.append([0 for jj in range(self.mesh.nphys_vars)])
            # loop over jacobian columns (blocks)
            for jj in range(self.mesh.nphys_vars):
                # loop over phys_vars to compute divergence
                for dd in range(self.mesh.nphys_vars):
                    jac[ii][jj] += multi_dot((dmats[dd], (
                        (2*depth_vals[:, :1]*vecs[ii][:, dd:dd+1]*visc_jac[jj]) +
                        ((2*visc*depth_vals[:, 0])[:, None])*vecs_jac[ii][jj][dd])))
            jac[ii][ii] -= torch.diag(self._beta_vals[:, 0])
        return torch.vstack([torch.hstack(j) for j in jac])

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
        residual, jac = super()._raw_residual_nD(
            split_sols[:-1], depth_vals[:, None])
        vel_vals = torch.hstack(
            [s[:, None] for s in split_sols[:self.mesh.nphys_vars]])
        depth_residual = -self.mesh._meshes[self.mesh.nphys_vars].div(
            depth_vals[:, None]*vel_vals)
        depth_residual += self._depth_forc_fun(
            self.mesh._meshes[self.mesh.nphys_vars].mesh_pts)[:, 0]
        return (torch.cat((residual, depth_residual)),
                self._raw_jacobian(split_sols[:-1], depth_vals))

    def _raw_jacobian(self, vel_vals, depth_vals):
        pderiv = self.mesh._meshes[0].partial_deriv
        # dmats = [self.mesh._meshes[0]._dmat(dd)
        #          for dd in range(self.mesh.nphys_vars)]
        dmats = self.mesh._meshes[0]._dmats
        vel_jac = super()._raw_jacobian_nD(
            vel_vals, depth_vals[:, None])
        dudx_ij = self._derivs(vel_vals)
        visc = self._viscosity(dudx_ij)
        vecs = self._vector_components(dudx_ij)
        dvdh_jac = []
        for ii in range(self.mesh.nphys_vars):
            dvdh_jac.append(0)
            for dd in range(self.mesh.nphys_vars):
                dvdh_jac[ii] += dmats[dd]*((2*visc*vecs[ii][:, dd])[None, :])
            dvdh_jac[ii] -= (
                (self._rho*self._g*depth_vals)[:, None]*dmats[ii] +
                self._rho*self._g*torch.diag(pderiv(
                    self._bed_vals[:, 0]+depth_vals, ii)))
        jac = torch.hstack((vel_jac, torch.vstack(dvdh_jac)))
        depth_jacs = []
        for ii in range(self.mesh.nphys_vars):
            depth_jacs.append(-dmats[ii]*depth_vals[None, :])
        depth_jacs.append(0)
        for ii in range(self.mesh.nphys_vars):
            depth_jacs[-1] -= dmats[ii]*vel_vals[ii][None, :]
        jac = torch.vstack((jac, torch.hstack(depth_jacs)))
        return jac


class FirstOrderStokesIce(AbstractSpectralCollocationPhysics):
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
        return self._raw_residual_nD(split_sols, depth_vals), None

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
