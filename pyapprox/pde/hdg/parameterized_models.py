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
from skfem.element import ElementVector
from pyapprox.pde.galerkin.util import _get_element
from pyapprox.pde.galerkin.physics import Stokes, Basis
from pyapprox.pde.galerkin.solvers import SteadyStatePDE as FEMSteadyStatePDE
from skfem.visuals.matplotlib import draw, plot, plt


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
            mesh, bndry_conds, diff_fun, zero_vel_fun, react_funs[0],
            forc_fun, react_funs[1]))
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


from skfem import MeshQuad
from pyapprox.pde.galerkin.meshes import init_gappy

def _gappy_bndry_tests(intervals):
    e = 1e-8
    return {
        "left": lambda x: np.isclose(x[0], intervals[0][0]),
        "right": lambda x: np.isclose(x[0], intervals[0][-1]),
        "bottom": lambda x: np.isclose(x[1], intervals[1][0]),
        "top": lambda x: np.isclose(x[1], intervals[1][-1]),
        "obs0": lambda x: (
            (x[0] >= (intervals[0][1]-e)) &
            (x[0] <= (intervals[0][2]+e)) &
            (x[1] >= (intervals[1][1]-e)) &
            (x[1] <= (intervals[1][2]+e))),
        "obs1": lambda x: (
            (x[0] >= (intervals[0][3]-e)) &
            (x[0] <= (intervals[0][4]+e)) &
            (x[1] >= (intervals[1][0]-e)) &
            (x[1] <= (intervals[1][1]+e))),
        "obs2": lambda x: (
            (x[0] >= (intervals[0][3]-e)) &
            (x[0] <= (intervals[0][4]+e)) &
            (x[1] >= (intervals[1][2]-e)) &
            (x[1] <= (intervals[1][3]+e))),}

def _fem_gappy_mesh(nrefine, intervals):
    MeshQuad.init_gappy = init_gappy
    mesh = (
        MeshQuad.init_gappy(*intervals).refined(nrefine).with_boundaries(
            _gappy_bndry_tests(intervals)))
    return mesh


def _stokes_no_slip_bndry_fun(x):
    vals = np.stack([0 * x[1], np.zeros_like(x[1])])
    # print(x[0].min(), x[0].max(), x[1].min(), x[1].max())
    # print(vals.shape, 'v')
    return vals


def _fem_gappy_stokes_inlet_bndry_fun(x):
    """return the plane Poiseuille parabolic inlet profile"""
    vals = np.stack([4 * x[1] * (1. - x[1]), np.zeros_like(x[1])])
    # print(x[0].min(), x[0].max(), x[1].min(), x[1].max(),
    #       vals.min(), vals.max())
    return vals


def _fem_gappy_stokes_bndry_conds(keys):
    D_bndry_conds = dict()
    for key in keys:
        if key != "left" and key != "right":
            D_bndry_conds[key] = [_stokes_no_slip_bndry_fun]
        elif key in ["left"]:
            D_bndry_conds[key] = [_fem_gappy_stokes_inlet_bndry_fun]
        # else:
        #   apply zero neumann on right boundary, i.e. do nothing
            
    return [D_bndry_conds, {}, {}]

        
def _vector_forcing_full(fill_value, xx):
    # return shape (xx.shape[1], xx.shape[0], xx.shape[2])
    return np.hstack([xx[0][:, None]*0+fill_value for x in xx])


def _forcing_full(fill_value, xx):
    shape = (xx.shape[1], xx.shape[0], xx.shape[2])
    return np.full((shape), fill_value)



class SteadyObstructedFlowModel():
    def __init__(self, L, orders, bndry_info,
                 source_info, functional=None, flow_type="darcy"):
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

        if flow_type == "darcy":
            self.pressure_solver = SteadyStateDomainDecompositionSolver(
                pressure_domain_decomp)
            self.pressure_solver._decomp.init_subdomains(
                partial(init_steady_pressure_induced_flow_subdomain_model,
                        pressure_domain_decomp, active_subdomain_indices,
                        nsubdomains_1d, orders, aa, bb, pleft, pright))
            self.psols = self.pressure_solver.solve()
            self.subdomain_vels = self._get_velocities(self.psols)[0]
        elif flow_type == "stokes" or flow_type == "navier_stokes":
            nrefine = 3 #5
            mesh = _fem_gappy_mesh(nrefine, intervals)
            element = {'u': ElementVector(_get_element(mesh, 2)),
                       'p': _get_element(mesh, 1)}
            basis = {variable: Basis(mesh, e, intorder=4)
                     for variable, e in element.items()}
            bndry_conds = _fem_gappy_stokes_bndry_conds(
                mesh.boundaries.keys())
            Re = 700
            self.vel_solver = FEMSteadyStatePDE(
                Stokes(mesh, element, basis, bndry_conds,
                       flow_type=="navier_stokes",
                       partial(_vector_forcing_full, 0),
                       partial(_forcing_full, 0), viscosity=L/Re))
            sol = self.vel_solver.solve()
            D_vals,  D_dofs = self.vel_solver.physics.assemble(sol)[2:]
            # print(D_vals)
            # print(D_dofs)
            # mesh = basis['u'].mesh
            # midp = mesh.p[:, mesh.facets].mean(axis=1)
            # for name, test in _gappy_bndry_tests(intervals).items():
            #     bndry_facets = np.nonzero(test(midp))[0]
            #     print(name, bndry_facets)
            #     plt.plot(*midp, 'o')
            #     plt.plot(*midp[:, bndry_facets], 'x')
            #     plt.show()
            # assert False
            vel, pres = np.split(
                sol, [sol.shape[0]-basis['p'].zeros().shape[0]])
            # self._get_velocities_fem(vel)
            # self._fem_plot_solution(sol, basis, intervals)
            # plt.show()
           
        else:
            raise ValueError(f"flow_type {flow_type} not supported")
        assert False

        tracer_domain_decomp = GappyRectangularDomainDecomposition(
            self.domain_bounds, nsubdomains_1d, orders[0]-1,
            missing_subdomain_indices, 12, intervals)
        self.tracer_solver = self._set_tracer_solver(
            tracer_domain_decomp, active_subdomain_indices, nsubdomains_1d,
            orders, amp, loc, scale)

    def _get_velocities_fem(self):
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

    def _get_meshgrid(self, intervals, npts_1d):
        from pyapprox.util.visualization import get_meshgrid_samples
        X, Y, pts = get_meshgrid_samples(self.domain_bounds, npts_1d)
        bt = _gappy_bndry_tests(intervals)
        II = np.ones(pts.shape[1], dtype=bool)
        II[bt["obs0"](pts) | bt["obs1"](pts) | bt["obs2"](pts)] = False
        return X, Y, pts, II

    def _get_fem_velocities_subset(self, vel, basis, pts, II):
        Z1, Z2 = np.zeros((pts.shape[1])), np.zeros((pts.shape[1]))
        Z1[II] = basis['u'].split_bases()[0].interpolator(
            vel[::2])(pts[:, II])
        Z2[II] = basis['u'].split_bases()[1].interpolator(
            vel[1::2])(pts[:, II])
        return Z1, Z2

    def _fem_plot_solution(self, sol, basis, intervals):
         # draw(mesh)
        vel, pres = np.split(
            sol, [sol.shape[0]-basis['p'].zeros().shape[0]])
        axs = plt.subplots(1, 5, figsize=(5*8, 6))[1]
        X, Y, pts, II = self._get_meshgrid(intervals, 51)
        Z1, Z2 = self._get_fem_velocities_subset(vel, basis, pts, II)
        axs[4].streamplot(X, Y, Z1.reshape(X.shape), Z2.reshape(X.shape),
                          color='k')
        im = axs[4].contourf(
            X, Y, np.sqrt(Z1**2+Z2**2).reshape(X.shape), cmap="coolwarm",
            levels=21)
        plt.colorbar(im, ax=axs[4])
        # use small number of samples so not too many arrows
        X, Y, pts, II = self._get_meshgrid(intervals, 21)
        Z1, Z2 = self._get_fem_velocities_subset(vel, basis, pts, II)
        axs[0].quiver(pts[0, II], pts[1, II], Z1[II], Z2[II],
                      scale_units='xy', angles='xy')
        plot(basis['u'].split_bases()[0], vel[::2], ax=axs[1],
             colorbar=True)
        plot(basis['u'].split_bases()[1], vel[1::2], ax=axs[2],
             colorbar=True)
        plot(basis['p'], pres, ax=axs[3], colorbar=True)

        # could use the following instead of meshgrid on velocities
        # but it is hard to control arrow size
        # plot(basis['u'], vel, ax=axs[0])
        # below replaces plot(basis['u'], vel, ax=axs[0])
        # which does not allow thining of field. Keep stride a multiple
        # of 3
        # m, z = basis['u'].refinterp(vel, nrefs=1)
        # stride = 3*(nrefine+1)
        # axs[0].quiver(*m.p[:, ::stride], *z.reshape(2, -1)[:, ::stride],
        #               angles='xy', scale_units="xy")

        

    def _set_tracer_solver(self, tracer_domain_decomp,
                           active_subdomain_indices, nsubdomains_1d,
                           orders, amp, loc, scale):
        tracer_solver = SteadyStateDomainDecompositionSolver(
            tracer_domain_decomp)
        tracer_solver._decomp.init_subdomains(
            partial(
                init_obstructed_tracer_flow_subdomain_model,
                tracer_domain_decomp, active_subdomain_indices,
                nsubdomains_1d, self.subdomain_vels, None, orders,
                self.nominal_concentration, amp, loc, scale))
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
