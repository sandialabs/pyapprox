import numpy as np
from functools import partial
import torch


from pyapprox.interface.wrappers import evaluate_1darray_function_on_2d_array
from pyapprox.pde.autopde.mesh import TransformedCollocationMesh
from pyapprox.pde.autopde.solvers import SteadyStatePDE, TransientPDE
from pyapprox.pde.autopde.physics import AdvectionDiffusionReaction
from pyapprox.pde.hdg.pde_coupling import (
    TransientDomainDecompositionSolver,
    SteadyStateDomainDecompositionSolver, GappyRectangularDomainDecomposition,
    get_active_subdomain_indices)


def full_fun_axis_0(fill_val, xx, oned=True):
    vals = torch.full((xx.shape[0], ), fill_val, dtype=torch.double)
    if oned:
        return vals
    else:
        return vals[:, None]


def full_fun_axis_1(fill_val, xx, oned=True):
    vals = torch.full((xx.shape[1], ), fill_val, dtype=torch.double)
    if oned:
        return vals
    else:
        return vals[:, None]


def obstructed_flow_boundary_fun(amp, const, aa, bb, xx):
    # assumes len of domain in xx[1] direction is 1
    # aa, bb = 8, 3
    vals = const + amp*xx[1]**(aa-1)*(1-xx[1])**(bb-1)/(0.5)**(aa+bb)
    return vals[:, None]


def obstructed_flow_forcing_fun(amp, loc, scale, x):
    assert loc.ndim == 1
    return amp*np.exp(-np.sum((x-loc[:, None])**2/scale**2, axis=0))[:, None]


def update_obstructed_tracer_flow_boundary_conds(
        alpha, nominal_concentration, subdomain_id, nsubdomains_1d,
        active_subdomain_indices, bndry_conds):
    for ii in range(2):
        if active_subdomain_indices[0, subdomain_id] == 0 and ii == 0:
            # on left bounary
            # bndry_conds[ii] = [
            #     partial(full_fun_axis_1, uleft, oned=False), "D"]
            # if a boundary is not zero flux when generating velocity field
            # then this must be also true in transient solve for concentration
            # material must be able to leave domain if velocity is pushing it to
            bndry_conds[ii] = [
                partial(full_fun_axis_1, alpha*nominal_concentration,
                        oned=False), "R", alpha]
        elif (active_subdomain_indices[0, subdomain_id] == nsubdomains_1d[0]-1
              and ii == 1):
            # on right bounary
            # bndry_conds[ii] = [
            #     partial(full_fun_axis_1, uright, oned=False), "D"]
            bndry_conds[ii] = [
                partial(full_fun_axis_1, alpha*nominal_concentration,
                        oned=False), "R", alpha]
    return bndry_conds


def init_steady_pressure_induced_flow_subdomain_model(
        domain_decomp, active_subdomain_indices, nsubdomains_1d, orders,
        aa, bb, pleft, pright, mesh_transform, subdomain_id):

    # top and bottom boundaries are neumann
    bndry_conds = [
        None, None,
        [partial(full_fun_axis_1, 0., oned=False), "N"],
        [partial(full_fun_axis_1, 0., oned=False), "N"]]
    for ii in range(2):
        if active_subdomain_indices[0, subdomain_id] == 0 and ii == 0:
            # on left bounary
            bndry_conds[ii] = [
                partial(full_fun_axis_1, pleft, oned=False), "D"]
        elif (active_subdomain_indices[0, subdomain_id] == nsubdomains_1d[0]-1
              and ii == 1):
            # on right bounary
            bndry_conds[ii] = [
                # partial(full_fun_axis_1, pright, oned=False), "D"]
                partial(obstructed_flow_boundary_fun, -pright/2,
                        pright, aa, bb), "D"]
        else:
            # boundary of obstruction
            # or interface (the latter will be ovewritten)
            bndry_conds[ii] = [
                partial(full_fun_axis_1, 0., oned=False), "N"]

    diff_fun = partial(full_fun_axis_1, 1, oned=False)
    forc_fun = partial(full_fun_axis_1, 0., oned=False)
    react_funs = [partial(full_fun_axis_0, 0, oned=False),
                  partial(full_fun_axis_0, 0, oned=True)]

    mesh = TransformedCollocationMesh(orders, mesh_transform)
    solver = SteadyStatePDE(
        AdvectionDiffusionReaction(
            mesh, bndry_conds, diff_fun, zero_vel_fun, react_funs[0], forc_fun,
            react_funs[1]))
    return solver


def init_obstructed_tracer_flow_subdomain_model(
        domain_decomp, active_subdomain_indices, nsubdomains_1d,
        subdomain_vels, deltat, orders, nominal_concentration,
        amp, loc, scale, mesh_transform, subdomain_id):

    # top and bottom boundaries are neumann
    bndry_conds = [
        [partial(full_fun_axis_1, 0., oned=False), "N"],
        [partial(full_fun_axis_1, 0., oned=False), "N"],
        [partial(full_fun_axis_1, 0., oned=False), "N"],
        [partial(full_fun_axis_1, 0., oned=False), "N"]]
    alpha = 0.1
    bndry_conds = update_obstructed_tracer_flow_boundary_conds(
        alpha, nominal_concentration, subdomain_id, nsubdomains_1d,
        active_subdomain_indices, bndry_conds)

    # forc_fun = partial(full_fun_axis_1, 0., oned=False)
    forc_fun = partial(obstructed_flow_forcing_fun, amp, loc, scale)

    # smaller diffusivity slows diffusion down
    diff_fun = partial(full_fun_axis_1, .1, oned=False)
    react_funs = [partial(full_fun_axis_0, 0, oned=False),
                  partial(full_fun_axis_0, 0, oned=True)]

    mesh = TransformedCollocationMesh(orders, mesh_transform)
    if deltat is not None:
        return TransientPDE(
            AdvectionDiffusionReaction(
                mesh, bndry_conds, diff_fun,
                partial(fixed_vel_fun, subdomain_vels[subdomain_id]),
                react_funs[0], forc_fun, react_funs[1]), deltat, "im_beuler1")
    return SteadyStatePDE(
        AdvectionDiffusionReaction(
            mesh, bndry_conds, diff_fun,
            partial(fixed_vel_fun, subdomain_vels[subdomain_id]),
            react_funs[0], forc_fun, react_funs[1]))


def zero_vel_fun(xx):
    return torch.zeros((xx.shape[1], 2), dtype=torch.double)


def fixed_vel_fun(vel_vals, xx):
    assert vel_vals.shape[0] == xx.shape[1]
    return vel_vals


class SteadyObstructedFlowModel():
    def __init__(self, L, orders, bndry_info,
                 source_info, functional=None):
        self.domain_bounds = [0, L, 0, 1]
        self.orders = orders
        aa, bb, pleft, pright = bndry_info
        self.nominal_concentration = 1.0
        amp, scale = source_info[:2]
        loc = np.array(source_info[2:])
        self._functional = functional

        nsubdomains_1d = [5, 3]
        # missing_subdomain_indices = [[1, 0], [1, 2], [3, 1]]
        missing_subdomain_indices = [[1, 1], [3, 0], [3, 2]]
        # missing_subdomain_indices = [[1, 0], [2, 2], [3, 1]]
        intervals = [
            np.array([0, 2*L/7, 3*L/7, 4*L/7, 5*L/7, L]),
            np.linspace(*self.domain_bounds[2:], nsubdomains_1d[1]+1)]
        pressure_domain_decomp = GappyRectangularDomainDecomposition(
            self.domain_bounds, nsubdomains_1d, orders[0]-1,
            missing_subdomain_indices, 12, intervals)

        active_subdomain_indices = get_active_subdomain_indices(
            nsubdomains_1d, missing_subdomain_indices)

        self.pressure_solver = SteadyStateDomainDecompositionSolver(
            pressure_domain_decomp)
        self.pressure_solver._decomp.init_subdomains(
            partial(init_steady_pressure_induced_flow_subdomain_model,
                    pressure_domain_decomp, active_subdomain_indices,
                    nsubdomains_1d, orders, aa, bb, pleft, pright))

        self.psols = self.pressure_solver.solve()
        self.subdomain_vels = self._get_velocities(self.psols)[0]

        tracer_domain_decomp = GappyRectangularDomainDecomposition(
            self.domain_bounds, nsubdomains_1d, orders[0]-1,
            missing_subdomain_indices, 12, intervals)
        self.tracer_solver = self._set_tracer_solver(
            tracer_domain_decomp, active_subdomain_indices, nsubdomains_1d,
            orders, amp, loc, scale)

    def _set_tracer_solver(self, tracer_domain_decomp,
                           active_subdomain_indices, nsubdomains_1d,
                           orders, amp, loc, scale):
        tracer_solver = SteadyStateDomainDecompositionSolver(
            tracer_domain_decomp)
        tracer_solver._decomp.init_subdomains(
            partial(
                init_obstructed_tracer_flow_subdomain_model,
                tracer_domain_decomp, active_subdomain_indices, nsubdomains_1d,
                self.subdomain_vels, None, orders, self.nominal_concentration,
                amp, loc, scale))
        return tracer_solver

    def _get_velocities(self, psols):
        norm_vels = []
        vel_mags = []
        subdomain_vels = []
        for jj, model in enumerate(
                self.pressure_solver._decomp._subdomain_models):
            mesh = model.physics.mesh
            idx = np.arange(mesh.mesh_pts.shape[1])
            flux_jac_vals = model.physics.flux_jac(idx)
            # only works because diffusivity field is
            # constant across each subdomain
            subdomain_vels.append(torch.hstack([
                -torch.linalg.multi_dot(
                    (flux_jac_vals[dd], psols[jj]))[:, None]
                for dd in range(len(flux_jac_vals))]))
            vel_mags.append(
                np.linalg.norm(subdomain_vels[-1], axis=1)[:, None])
            norm_vels.append(subdomain_vels[-1]/vel_mags[-1])
        return subdomain_vels, vel_mags, norm_vels

    def plot_velocities(self, ax, **kwargs):
        subdomain_vels, vel_mags, norm_vels = self._get_velocities(
            self.psols)
        return self._plot_velocities(
            subdomain_vels, vel_mags, norm_vels, ax, **kwargs)

    def _plot_velocities(self, subdomain_vels, vel_mags, norm_vels,
                         ax, **kwargs):
        velmag_min = np.min([v.min() for v in vel_mags])
        velmag_max = np.max([v.max() for v in vel_mags])
        tmp = np.hstack(vel_mags).flatten()
        tmp = tmp[tmp < np.quantile(tmp, 0.99)]
        velmag_max = tmp.max()
        print(velmag_min, velmag_max)
        levels = np.linspace(velmag_min, velmag_max, 51)
        for jj, model in enumerate(
                self.pressure_solver._decomp._subdomain_models):
            mesh = model.physics.mesh
            mesh.plot(vel_mags[jj], nplot_pts_1d=100, ax=ax, levels=levels,
                      **kwargs)
            jdx = np.arange(mesh.mesh_pts.shape[1]).reshape(
                np.asarray(self.orders)+1)
            mesh_pts = mesh.mesh_pts[:, jdx.flatten()]
            nvels = norm_vels[jj][jdx.flatten()]
            ax.quiver(*mesh_pts, *nvels.T)

    def _set_random_sample(self, sample):
        assert sample.shape[0] == 4
        amp, scale = sample[:2]
        loc = sample[2:]
        forc_fun = partial(obstructed_flow_forcing_fun, amp, loc, scale)
        for model in self.tracer_solver._decomp._subdomain_models:
            model.physics._forc_fun = forc_fun
            model.physics._funs[-1] = forc_fun

    def _solve(self, sample):
        self._set_random_sample(sample)
        tracer_sols = self.tracer_solver.solve()
        return tracer_sols

    def _eval(self, sample):
        tracer_sols = self._solve(sample)
        qoi = self._functional(tracer_sols, sample.copy)
        if isinstance(qoi, np.ndarray):
            if qoi.ndim == 1:
                return qoi
            assert qoi.shape[1] == 1
            return qoi[:, 0]
        return np.asarray([qoi])

    def __call__(self, samples, return_grad=False):
        return evaluate_1darray_function_on_2d_array(
            self._eval, samples, return_grad=return_grad)


class TransientObstructedFlowModel(SteadyObstructedFlowModel):
    def __init__(self, deltat, final_time, L, orders, bndry_info,
                 source_info, functional=None):
        self.deltat = deltat
        self.final_time = final_time
        super().__init__(L, orders, bndry_info, source_info, functional)

    def _set_tracer_solver(self, tracer_domain_decomp,
                           active_subdomain_indices, nsubdomains_1d,
                           orders, amp, loc, scale):
        tracer_solver = TransientDomainDecompositionSolver(
            tracer_domain_decomp)
        tracer_solver._decomp.init_subdomains(
            partial(init_obstructed_tracer_flow_subdomain_model,
                    tracer_domain_decomp, active_subdomain_indices,
                    nsubdomains_1d, self.subdomain_vels,
                    self.deltat, orders, self.nominal_concentration,
                    amp, loc, scale))
        return tracer_solver
