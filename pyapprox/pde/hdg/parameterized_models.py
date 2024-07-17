import os
import pickle
import numpy as np
from functools import partial
import torch

from pyapprox.util.utilities import common_matrix_rows
from pyapprox.interface.wrappers import evaluate_1darray_function_on_2d_array
from pyapprox.pde.autopde.mesh import TransformedCollocationMesh
from pyapprox.pde.autopde.solvers import SteadyStatePDE, TransientPDE
from pyapprox.pde.autopde.physics import AdvectionDiffusionReaction
from pyapprox.pde.hdg.pde_coupling import (
    TransientDomainDecompositionSolver,
    SteadyStateDomainDecompositionSolver, GappyRectangularDomainDecomposition,
    get_active_subdomain_indices, TurbineDomainDecomposition)
from skfem.element import ElementVector
from pyapprox.pde.galerkin.util import _get_element
from pyapprox.pde.galerkin.physics import Stokes, Basis
from pyapprox.pde.galerkin.physics import (
    AdvectionDiffusionReaction as AdvectionDiffusionReactionFEM)
from pyapprox.pde.galerkin.solvers import SteadyStatePDE as FEMSteadyStatePDE
from skfem.visuals.matplotlib import plot, plt
from skfem import MeshQuad, Functional
from pyapprox.pde.galerkin.meshes import init_gappy
from pyapprox.pde.kle.torchkle import TorchMeshKLE


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
    if x.ndim == 2:
        loc = loc[:, None]
        return amp*np.exp(-np.sum((x-loc)**2/scale**2, axis=0))[:, None]
    else:
        loc = loc[:, None, None]
        vals = amp*np.exp(-np.sum((x-loc)**2/scale**2, axis=0))
        return vals


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
        subdomain_vels, deltat, orders, nominal_concentration, robin_alpha,
        amp, loc, scale, mesh_transform, subdomain_id):

    # top and bottom boundaries are neumann
    bndry_conds = [
        [partial(full_fun_axis_1, 0., oned=False), "N"],
        [partial(full_fun_axis_1, 0., oned=False), "N"],
        [partial(full_fun_axis_1, 0., oned=False), "N"],
        [partial(full_fun_axis_1, 0., oned=False), "N"]]
    bndry_conds = update_obstructed_tracer_flow_boundary_conds(
        robin_alpha, nominal_concentration, subdomain_id, nsubdomains_1d,
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
    # assert vel_vals.shape[0] == xx.shape[1]
    return vel_vals


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
            (x[1] <= (intervals[1][3]+e)))}


def _fem_gappy_mesh(nrefine, intervals):
    MeshQuad.init_gappy = init_gappy
    mesh = (
        MeshQuad.init_gappy(*intervals).refined(nrefine).with_boundaries(
            _gappy_bndry_tests(intervals)))
    return mesh


def _stokes_no_slip_bndry_fun(x):
    vals = np.stack([0 * x[1], np.zeros_like(x[1])])
    return vals


def _fem_gappy_stokes_inlet_bndry_fun(x):
    """return the plane Poiseuille parabolic inlet profile"""
    vals = np.stack([4 * x[1] * (1. - x[1]), np.zeros_like(x[1])])
    # vals = np.stack([1/.08 * x[1] * (1. - x[1])**4, np.zeros_like(x[1])])
    return vals


def _fem_dirichlet_scalar_bndry_fun(val, x):
    vals = 0 * x[1]+val
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


def _fem_gappy_advection_diffusion_reaction_bndry_conds(
        keys, alpha,  nominal_concentration):
    R_bndry_conds = dict()
    for key in keys:
        if key == "left":
            R_bndry_conds[key] = [
                partial(_fem_dirichlet_scalar_bndry_fun,
                        alpha*nominal_concentration), alpha]
        if key == "right":
            R_bndry_conds[key] = [
                partial(_fem_dirichlet_scalar_bndry_fun,
                        alpha*nominal_concentration), alpha]
        # else:
        #   apply zero neumann on right boundary, i.e. do nothing
    # return [D_bndry_conds, {}, {}]
    return [{}, {}, R_bndry_conds]


def _vector_forcing_full(fill_value, xx):
    # return shape (xx.shape[1], xx.shape[0], xx.shape[2])
    return np.hstack([xx[0][:, None]*0+fill_value for x in xx])


def _forcing_full(fill_value, xx):
    shape = (xx.shape[1], xx.shape[0], xx.shape[2])
    return np.full((shape), fill_value)


class FEMIntegrateRectangularSubdomain(Functional):
    def __init__(self, bounds):
        super().__init__(self._form)
        self._bounds = bounds

    def indicator(self, samples):
        indices = np.where(
            (samples[0, :] >= self._bounds[0]) &
            (samples[0, :] <= self._bounds[1]) &
            (samples[1, :] >= self._bounds[2]) &
            (samples[1, :] <= self._bounds[3]))[0]
        vals = np.zeros(samples.shape[1:])
        vals[indices] = 1.0
        return vals

    def _form(self, w):
        return w.y*self.indicator(w.x)


class SteadyObstructedFlowModel():
    def __init__(self, L, orders, bndry_info,
                 source_info, functionals=None, flow_type="navier_stokes",
                 vel_filename=None, reynolds_num=None,
                 tracer_solver_type="fem", nrefine=5, robin_alpha=0.1,
                 nominal_concentration=1.):
        self.domain_bounds = [0, L, 0, 1]
        self.orders = orders
        # aa, bb used to define beta PDF profile for right boundary when
        # solving darcy flow, pright scales the PDF profile
        aa, bb, pleft, pright = bndry_info
        self.nominal_concentration = nominal_concentration
        self.robin_alpha = robin_alpha
        amp, scale = source_info[:2]
        loc = np.array(source_info[2:])
        self._functionals = functionals
        self._flow_type = flow_type
        self._reynolds_num = reynolds_num
        self._tracer_solver_type = tracer_solver_type

        nsubdomains_1d = [5, 3]
        missing_subdomain_indices = [[1, 1], [3, 0], [3, 2]]
        self.intervals = [
            np.array([0, 2*L/7, 3*L/7, 4*L/7, 5*L/7, L]),
            np.linspace(*self.domain_bounds[2:], nsubdomains_1d[1]+1)]
        self.pressure_domain_decomp = GappyRectangularDomainDecomposition(
            self.domain_bounds, nsubdomains_1d, orders[0]-1,
            missing_subdomain_indices, 12, self.intervals)

        self.active_subdomain_indices = get_active_subdomain_indices(
            nsubdomains_1d, missing_subdomain_indices)

        self.tracer_domain_decomp = GappyRectangularDomainDecomposition(
            self.domain_bounds, nsubdomains_1d, orders[0]-1,
            missing_subdomain_indices, 12, self.intervals)

        self._init_fem_mesh_and_basis(nrefine)

        # set regardless of flow type so can access mesh to interpolate
        # velocities
        self.pressure_solver = SteadyStateDomainDecompositionSolver(
                self.pressure_domain_decomp)
        self.pressure_solver._decomp.init_subdomains(
            partial(init_steady_pressure_induced_flow_subdomain_model,
                    self.pressure_domain_decomp, self.active_subdomain_indices,
                    nsubdomains_1d, orders, aa, bb, pleft, pright))

        self.vel_filename = vel_filename
        self._compute_velocities()

        self.tracer_solver = self._set_tracer_solver(
            self.active_subdomain_indices, nsubdomains_1d,
            orders, amp, loc, scale)

    def _compute_darcy_velocities(self):
        assert self._reynolds_num is None
        self.psols = self.pressure_solver.solve()
        self.subdomain_vels = self._get_collocation_velocities(
            self.psols)

    def _compute_stokes_velocities(self):
        assert self._reynolds_num is not None
        if self.vel_filename is None or not os.path.exists(self.vel_filename):
            bndry_conds = _fem_gappy_stokes_bndry_conds(
                self._fem_mesh.boundaries.keys())
            L = self.intervals[0][-1]
            self.vel_solver = FEMSteadyStatePDE(
                Stokes(self._fem_mesh, self._stokes_fem_element,
                       self._stokes_fem_basis, bndry_conds,
                       self._flow_type == "navier_stokes",
                       partial(_vector_forcing_full, 0),
                       partial(_forcing_full, 0),
                       viscosity=L/self._reynolds_num))
            sol = self.vel_solver.solve()
            if self.vel_filename is not None:
                with open(self.vel_filename, 'wb') as file_object:
                    pickle.dump(sol, file_object)
        else:
            print("Loading velocity file", self.vel_filename)
            with open(self.vel_filename, 'rb') as file_object:
                sol = pickle.load(file_object)
        print(self._flow_type, "NDOF", sol.shape[0])
        self.vel, pres = np.split(
            sol, [sol.shape[0]-self._stokes_fem_basis['p'].zeros().shape[0]])
        self.subdomain_vels = self._get_velocities_fem(
            self.pressure_solver._decomp, self._stokes_fem_basis, self.vel)
        self.psols = self._get_presure_fem(
            self.pressure_solver._decomp, self._stokes_fem_basis,
            pres)

    def _compute_velocities(self):
        if self._flow_type == "darcy":
            self._compute_darcy_velocities()
        elif self._flow_type == "stokes" or self._flow_type == "navier_stokes":
            self._compute_stokes_velocities()
        else:
            raise ValueError(f"flow_type {self._flow_type} not supported")

    def _get_velocities_fem(self, decomp, basis, vels):
        subdomain_vels = []
        for jj, model in enumerate(decomp._subdomain_models):
            mesh = model.physics.mesh
            subdomain_vels.append(torch.hstack([
                torch.as_tensor(basis['u'].split_bases()[dd].interpolator(
                    vels[dd::2])(mesh.mesh_pts)[:, None])
                for dd in range(2)]))
        return subdomain_vels

    def _get_presure_fem(self, decomp, basis, pres):
        subdomain_pres = []
        for jj, model in enumerate(decomp._subdomain_models):
            mesh = model.physics.mesh
            subdomain_pres.append(
                torch.as_tensor(basis['p'].interpolator(pres)(
                    mesh.mesh_pts)[:, None]))
        return subdomain_pres

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
        vel, pres = np.split(
            sol, [sol.shape[0]-basis['p'].zeros().shape[0]])
        axs = plt.subplots(1, 5, figsize=(5*8, 6))[1]
        X, Y, pts, II = self._get_meshgrid(intervals, 101)
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

    def _init_fem_mesh_and_basis(self, nrefine):
        self._fem_mesh = _fem_gappy_mesh(nrefine, self.intervals)
        self._tracer_fem_element = _get_element(self._fem_mesh, 1)
        self._tracer_fem_basis = Basis(
            self._fem_mesh, self._tracer_fem_element, intorder=4)
        self._stokes_fem_element = {
            'u': ElementVector(_get_element(self._fem_mesh, 2)),
            'p': _get_element(self._fem_mesh, 1)}
        self._stokes_fem_basis = {
            variable: Basis(self._fem_mesh, e, intorder=4)
            for variable, e in self._stokes_fem_element.items()}

    def _nl_diff_fun(self, diff_fun, x, sol):
        return diff_fun(x)

    def _nl_diff_jac(self, x, sol):
        return x[0]*0

    def _set_fem_tracer_solver(self, amp, loc, scale):
        forc_fun = partial(obstructed_flow_forcing_fun, amp, loc, scale)
        diff_fun = partial(full_fun_axis_1, .1, oned=False)
        vel_fun = partial(
            fixed_vel_fun,
            self._stokes_fem_basis['u'].interpolate(self.vel))
        bndry_conds = _fem_gappy_advection_diffusion_reaction_bndry_conds(
            self._fem_mesh.boundaries.keys(), self.robin_alpha,
            self.nominal_concentration)
        nl_diff_funs = [partial(self._nl_diff_fun, diff_fun),
                        self._nl_diff_jac]
        tracer_solver = FEMSteadyStatePDE(
            AdvectionDiffusionReactionFEM(
                self._fem_mesh, self._tracer_fem_element,
                self._tracer_fem_basis, bndry_conds,
                diff_fun, forc_fun, vel_fun, nl_diff_funs))
        return tracer_solver

    def _set_hdg_tracer_solver(self, active_subdomain_indices, nsubdomains_1d,
                               orders, amp, loc, scale):
        tracer_solver = SteadyStateDomainDecompositionSolver(
            self.tracer_domain_decomp)
        tracer_solver._decomp.init_subdomains(
            partial(
                init_obstructed_tracer_flow_subdomain_model,
                self.tracer_domain_decomp, active_subdomain_indices,
                nsubdomains_1d, self.subdomain_vels, None, orders,
                self.nominal_concentration, self.robin_alpha, amp, loc, scale))
        return tracer_solver

    def _set_tracer_solver(self, active_subdomain_indices, nsubdomains_1d,
                           orders, amp, loc, scale):
        if self._tracer_solver_type == "fem":
            # boundary conditions are not consistent with hdg
            # need to add general test to test_finite_elements to
            # check interpolating velocity field into function
            # passed to AdvectionDiffusionReaction
            # raise NotImplementedError("Need to complete")
            return self._set_fem_tracer_solver(amp, loc, scale)
        return self._set_hdg_tracer_solver(
            active_subdomain_indices, nsubdomains_1d,
            orders, amp, loc, scale)

    def _get_collocation_velocities(self, psols):
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
        return subdomain_vels

    def plot_velocities(self, ax, **kwargs):
        # use collocation mesh to plot regardless of velocity solver
        # if want to use fem plot call self._fem_plot_solution
        return self._plot_collocation_velocities(
            self.subdomain_vels, ax, **kwargs)

    def _plot_collocation_velocities(
            self, subdomain_vels, ax, **kwargs):

        X, Y, pts, II = self._get_meshgrid(self.intervals, 101)
        masks = self.pressure_solver._decomp._in_subdomains(pts)
        Z1 = np.full((pts.shape[1], 1), np.nan)
        Z2 = np.full((pts.shape[1], 1), np.nan)
        # Z3 = np.full((pts.shape[1], 1), np.nan)
        for ii, mask in enumerate(masks):
            model = self.pressure_solver._decomp._subdomain_models[ii]
            mesh = model.physics.mesh
            Z1[mask] = mesh.interpolate(
                self.subdomain_vels[ii][:, 0], pts[:, mask])
            Z2[mask] = mesh.interpolate(
                self.subdomain_vels[ii][:, 1], pts[:, mask])
        ax.streamplot(X, Y, Z1.reshape(X.shape), Z2.reshape(X.shape),
                      color='k', density=2)
        ax.contourf(
            X, Y, np.sqrt(Z1**2+Z2**2).reshape(X.shape), **kwargs)

    def _set_random_sample(self, sample):
        assert sample.ndim == 1
        assert sample.shape[0] == 5
        amp, scale = sample[:2]
        loc = sample[2:4]
        diff = np.exp(sample[4])
        forc_fun = partial(obstructed_flow_forcing_fun, amp, loc, scale)
        diff_fun = partial(full_fun_axis_1, diff, oned=False)
        if self._tracer_solver_type == "hdg":
            for model in self.tracer_solver._decomp._subdomain_models:
                model.physics._forc_fun = forc_fun
                model.physics._funs[-1] = forc_fun
                model.physics._diff_fun = diff_fun
                model.physics._funs[0] = diff_fun
            return

        self.tracer_solver.physics.diff_fun = diff_fun
        self.tracer_solver.physics.forc_fun = forc_fun
        self.tracer_solver.physics.nl_diff_funs = [
            partial(self._nl_diff_fun, diff_fun), self._nl_diff_jac]

        self.tracer_solver.physics.funs[0] = diff_fun
        self.tracer_solver.physics.funs[2] = forc_fun
        self.tracer_solver.physics.funs[3] = (
            self.tracer_solver.physics.nl_diff_funs)

    def _solve(self, sample):
        self._set_random_sample(sample)
        tracer_sols = self.tracer_solver.solve()
        return tracer_sols

    def _eval(self, sample):
        # import time
        # t0 = time.time()
        tracer_sols = self._solve(sample)
        # print("solve took", time.time()-t0)
        # t0 = time.time()
        qoi = []
        for ii in range(len(self._functionals)):
            qoi_ii = np.atleast_1d(self._functionals[ii](
                tracer_sols, sample.copy()))
            if qoi_ii.ndim == 2:
                assert qoi_ii.shape[1] == 1
                qoi_ii = qoi_ii[:, 0]
            qoi.append(qoi_ii)
        # print("qoi took", time.time()-t0)
        return np.hstack(qoi)

    def __call__(self, samples, return_grad=False):
        return evaluate_1darray_function_on_2d_array(
            self._eval, samples, return_grad=return_grad)


class TransientObstructedFlowModel(SteadyObstructedFlowModel):
    def __init__(self, deltat, final_time, L, orders, bndry_info,
                 source_info, functional=None):
        self.deltat = deltat
        self.final_time = final_time
        super().__init__(L, orders, bndry_info, source_info, functional)

    def _set_tracer_solver(self, active_subdomain_indices, nsubdomains_1d,
                           orders, amp, loc, scale):
        tracer_solver = TransientDomainDecompositionSolver(
            self.tracer_domain_decomp)
        tracer_solver._decomp.init_subdomains(
            partial(init_obstructed_tracer_flow_subdomain_model,
                    self.tracer_domain_decomp, active_subdomain_indices,
                    nsubdomains_1d, self.subdomain_vels,
                    self.deltat, orders, self.nominal_concentration,
                    self.robin_alpha, amp, loc, scale))
        return tracer_solver


def init_steady_turbine_subdomain_model(
        domain_decomp, orders, mesh_transform, subdomain_id):
    mesh = TransformedCollocationMesh(orders, mesh_transform)

    bndry_conds = [
        [partial(full_fun_axis_1, 0., oned=False), "D"],
        [partial(full_fun_axis_1, 0., oned=False), "D"],
        [partial(full_fun_axis_1, 0., oned=False), "D"],
        [partial(full_fun_axis_1, 0., oned=False), "D"]]

    solver = SteadyStatePDE(
        AdvectionDiffusionReaction(
            mesh, bndry_conds, partial(full_fun_axis_1, 1.0, oned=False),
            zero_vel_fun, None, partial(full_fun_axis_1, 0.0, oned=False),
            None))
    return solver


class TurbineBladeModel():
    def __init__(self, orders, functional=None, height_max=0.05, length=1.0,
                 kle_args=None):
        self.orders = orders
        self._functional = functional

        ninterface_dof = np.min(self.orders)-1
        self._decomp_solver = SteadyStateDomainDecompositionSolver(
            TurbineDomainDecomposition(ninterface_dof, height_max, length))
        self._decomp_solver._decomp.init_subdomains(
            partial(init_steady_turbine_subdomain_model,
                    self._decomp_solver._decomp, self.orders))
        self._subdomain_bndry_dict = self._get_subdomain_bndry_dict()
        self._bndry_funs = {
            "exterior": self._exterior_bndry_fun,
            "passage1": self._passage1_bndry_fun,
            "passage2": self._passage2_bndry_fun,
            "passage3": self._passage3_bndry_fun,
            "endbottom": self._endbottom_bndry_cond,
            "endtop": self._endtop_bndry_cond
        }
        self._kle, self._mesh_pts = self._init_kle(*kle_args)

    def _get_subdomain_bndry_dict(self):
        # return bndrys that are not interfaces

        # return (
        #     [{"exterior": [0, 1, 2, 3]}] *
        #     self._decomp_solver._decomp._nsubdomains)
        # when excluding first column
        return [
            {"exterior": 1, "passage1": 0},  # 0
            {"exterior": 1, "passage1": 0},  # 1
            {"exterior": 1, "passage1": 0},  # 3
            {"exterior": 1, "passage1": 0},  # 4
            {"exterior": 1, "passage1": 0},  # 5
            {"exterior": 3, "passage1": 2},  # 6
            {"exterior": 2, "passage1": 3},  # 7
            {"exterior": 3},  # 8
            {"passage1": 0, "passage3": 1},  # 9
            {"exterior": 2},  # 10
            {"exterior": 3, "passage3": 2, "endtop": 1},  # 11
            {"exterior": 2, "passage3": 3, "endbottom": 1},  # 12
        ]
        # when including first column
        # return [
        #     {"exterior": 1, "passage1": 0},  # 0
        #     {"exterior": 1},  # 1
        #     {"passage1": 0, "passage2": 1},  # 2
        #     {"exterior": 1},  # 3
        #     {"exterior": 1, "passage2": 0},  # 4
        #     {"exterior": 1, "passage2": 0},  # 5
        #     {"exterior": 3, "passage2": 2},  # 6
        #     {"exterior": 2, "passage2": 3},  # 7
        #     {"exterior": 3},  # 8
        #     {"passage2": 0, "passage3": 1},  # 9
        #     {"exterior": 2},  # 10
        #     {"exterior": 3, "passage3": 2, "endtop": 1},  # 11
        #     {"exterior": 2, "passage3": 3, "endbottom": 1},  # 12
        # ]

    def _exterior_bndry_fun(self, xx):
        zz = xx[0]
        length = self._decomp_solver._decomp._length
        slope = (self._h_te-self._h_le)/length
        vals = slope*zz+self._h_le
        # assert vals.max() <= self._h_le and vals.min() >= self._h_te
        return vals[:, None]

    def _passage1_bndry_fun(self, xx):
        return np.full((xx.shape[1], 1), self._t_c1)

    def _passage2_bndry_fun(self, xx):
        return np.full((xx.shape[1], 1), self._t_c2)

    def _passage3_bndry_fun(self, xx):
        return np.full((xx.shape[1], 1), self._t_c3)

    def _endtop_bndry_cond(self, xx):
        mesh = self._decomp_solver._decomp._subdomain_models[-2].physics.mesh
        zz1 = np.array([mesh._canonical_mesh_pts_1d[0][-1],
                        mesh._canonical_mesh_pts_1d[1][0]])[:, None]
        x_end, y_interior = mesh._map_samples_from_canonical_domain(zz1)[:, 0]
        zz2 = np.array([mesh._canonical_mesh_pts_1d[0][-1],
                       mesh._canonical_mesh_pts_1d[1][-1]])[:, None]
        y_exterior = mesh._map_samples_from_canonical_domain(zz2)[1, 0]
        pt = np.array([x_end, y_interior])[:, None]
        slope = (self._exterior_bndry_fun(pt)[0, 0] -
                 self._passage3_bndry_fun(pt)[0, 0])/(y_exterior-y_interior)
        vals = slope*(xx[1]-y_interior)+self._passage3_bndry_fun(pt)[0, 0]
        return vals[:, None]

    def _endbottom_bndry_cond(self, xx):
        zz = xx.copy()
        zz[1] *= -1
        return self._endtop_bndry_cond(zz)

    def _eval_kle_on_subdoman(self, kle_vals, xx):
        assert xx.shape[1] == kle_vals.shape[0]
        return torch.as_tensor(kle_vals)

    def _get_mesh_pts_splits(self):
        splits = [0]
        for m in self._decomp_solver._decomp._subdomain_models:
            ndof = m.physics.mesh.mesh_pts.shape[1]
            splits.append(ndof+splits[-1])
        splits = np.asarray(splits)[1:]
        return splits

    def _set_random_sample(self, sample):
        sample = sample.squeeze()
        assert sample.ndim == 1
        self._t_c1, self._t_c2, self._t_c3, self._h_le, self._h_te = sample[:5]
        assert self._h_le >= self._h_te, (self._h_le, self._h_te)
        self._thermal_conductivity = sample[5:]
        if self._kle is None:
            assert (sample.shape[0] == 5 +
                    self._decomp_solver._decomp._nsubdomains)
        else:
            assert sample.shape[0] == 5 + self._kle.nterms
            splits = self._get_mesh_pts_splits()
            kle_vals = np.split(
                self._eval_kle(self._thermal_conductivity[:, None]), splits)

        for subdomain_id, model in enumerate(
                self._decomp_solver._decomp._subdomain_models):
            if self._kle is None:
                diff_fun = partial(
                    full_fun_axis_1, self._thermal_conductivity[subdomain_id],
                    oned=False)
            else:
                diff_fun = partial(
                    self._eval_kle_on_subdoman, kle_vals[subdomain_id])
            model.physics._diff_fun = diff_fun
            model.physics._funs[0] = diff_fun

            for key, item in self._subdomain_bndry_dict[subdomain_id].items():
                for idx in np.atleast_1d(item):
                    model.physics._bndry_conds[idx] = (
                        [self._bndry_funs[key], "D"])
        # reset interface bndrys in case they were overwritten by the above
        # loop
        self._decomp_solver._decomp._set_subdomain_interface_boundary_conditions()

    def _solve(self, sample):
        macro_newton_kwargs = {"maxiters": 2, "verbosity": 2, "rtol": 1e-7}
        subdomain_newton_kwargs = {}  # {"maxiters": 1, "verbosity": 2}
        self._set_random_sample(sample)
        sol = self._decomp_solver.solve(
            macro_newton_kwargs=macro_newton_kwargs,
            subdomain_newton_kwargs=subdomain_newton_kwargs)
        return sol

    def _eval(self, sample):
        sol = self._solve(sample)
        qoi = self._functional(sol, sample.copy)
        if isinstance(qoi, np.ndarray):
            if qoi.ndim == 1:
                return qoi
            assert qoi.shape[1] == 1
            return qoi[:, 0]
        return np.asarray([qoi])

    def __call__(self, samples, return_grad=False):
        return evaluate_1darray_function_on_2d_array(
            self._eval, samples, return_grad=return_grad)

    def _get_mesh(self):
        mesh_grid = []
        for model in self._decomp_solver._decomp._subdomain_models:
            mesh_grid.append(model.physics.mesh.mesh_pts)
        mesh_grid = np.hstack(mesh_grid)
        return mesh_grid

    def _expand_unique_mesh_values(self, restricted_vals):
        common_samples_dict = common_matrix_rows(self._mesh_pts.T)
        expanded_vals = np.empty(
            (self._mesh_pts.shape[1], restricted_vals.shape[1]))
        kk = 0
        for key, item in common_samples_dict.items():
            expanded_vals[item] = restricted_vals[kk]
            kk += 1
        return expanded_vals

    def _init_kle(self, *args):
        mesh_pts = self._get_mesh()
        if len(args) == 0:
            return None, mesh_pts

        length_scale, sigma, nterms = args
        self._common_mesh_pts_dict = common_matrix_rows(mesh_pts.T)
        unique_indices = np.array(
            [item[0] for key, item in self._common_mesh_pts_dict.items()])
        kle = TorchMeshKLE(mesh_pts[:, unique_indices], use_log=True)
        kle.compute_basis(length_scale, sigma, nterms)
        return kle, mesh_pts

    def _eval_kle(self, sample):
        return self._expand_unique_mesh_values(
            self._kle(sample))
