import torch
from abc import ABC, abstractmethod
import numpy as np
from functools import partial

from pyapprox.util.utilities import cartesian_product
from pyapprox.variables.transforms import _map_hypercube_samples
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d, barycentric_interpolation_1d,
    multivariate_barycentric_lagrange_interpolation
)
from pyapprox.pde.spectralcollocation.spectral_collocation import (
    chebyshev_derivative_matrix
)
from pyapprox.util.visualization import get_meshgrid_function_data, plt
from pyapprox.pde.autopde.util import newton_solve
from pyapprox.pde.autopde.time_integration import ImplicitRungeKutta


class IntervalMeshBoundary():
    def __init__(self, bndry_name, inactive_coord, tol=1e-15):
        self._bndry_index = {"left": 0, "right": 1}[bndry_name]
        self._inactive_coord = inactive_coord
        self._tol = tol

    def bndry_normals(self, samples):
        normals = np.array([[-1], [1]])
        return np.tile(normals[self._bndry_index, (1, samples.shape[1])]).T

    def quadrature_rule(self):
        return np.ones((1, 1)), np.ones((1, 1))

    def samples_on_boundary(self, samples):
        return np.where(
            np.absolute(self._inactive_coord-samples[0, :]) < self._tol)[0]


class RectangularMeshBoundary():
    def __init__(self, bndry_name, inactive_coord, order, active_bounds,
                 tol=1e-15):
        if len(active_bounds) != 2:
            msg = "Bounds must be specfied for the dimension with the "
            msg += "varying coordinates"
            raise ValueError(msg)

        self._bndry_index = {"left": 0, "right": 1, "bottom": 2, "top": 3}[
            bndry_name]
        self._order = order
        self._active_bounds = active_bounds
        self._inactive_coord = inactive_coord
        self._tol = tol

    def bndry_normals(self, samples):
        normals = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        return np.tile(normals[self._bndry_index, (1, samples.shape[1])]).T

    def quadrature_rule(self):
        nsamples = self._order+3
        xquad_1d, wquad_1d = gauss_jacobi_pts_wts_1D(nsamples, 0, 0)
        xlist = [None, None]
        xlist[int(self._bndry_index < 2)] = xquad_1d
        xlist[int(self._bndry_index >= 2)] = self._inactive_coord
        xquad = cartesian_product(xlist)
        wquad = wquad_1d[:, None]*np.prod(self.bounds[1]-self._bounds[0])
        return xquad, wquad

    def samples_on_boundary(self, samples):
        dd = int(self._bndry_index >= 2)
        indices = np.where(
            np.absolute(self._inactive_coord-samples[dd, :]) < self._tol)[0]
        return indices


class CartesianProductCollocationMesh():
    def __init__(self, domain_bounds, orders, bndry_conds):

        if len(orders) != len(domain_bounds)//2:
            raise ValueError("Order must be specified for each dimension")
        if len(orders) > 2:
            raise ValueError("Only 1D and 2D meshes supported")

        super().__init__()
        self._domain_bounds = np.asarray(domain_bounds)
        self._orders = orders
        self.nphys_vars = len(self._domain_bounds)//2

        self._canonical_domain_bounds = np.ones(2*self.nphys_vars)
        self._canonical_domain_bounds[::2] = -1.

        (self._canonical_mesh_pts_1d, self._canonical_derivative_matrices_1d,
         self._canonical_mesh_pts_1d_baryc_weights, self.mesh_pts,
         self._derivative_matrices) = (
             self._form_derivative_matrices())

        self._bndrys = self._form_boundaries()
        self._bndry_indices = self._determine_boundary_indices()
        if len(self._bndrys) != len(bndry_conds):
            raise ValueError(
                "Incorrect number of boundary conditions provided")
        self._bndry_conds = bndry_conds

    def _form_boundaries(self):
        if self.nphys_vars == 1:
            return [IntervalMeshBoundary(name, inactive_coord)
                    for name, inactive_coord in zip(
                            ["left", "right"], self._domain_bounds)]
        return [
            RectangularMeshBoundary(
                name, self._domain_bounds[ii], self._orders[int(ii < 2)],
                self._domain_bounds[2*int(ii < 2): 2*int(ii < 2)+2])
            for ii, name in enumerate(["left", "right", "bottom", "top"])]

    def _determine_boundary_indices(self):
        bndry_indices = [[] for ii in range(2*self.nphys_vars)]
        for ii in range(2*self.nphys_vars):
            bndry_indices[ii] = self._bndrys[ii].samples_on_boundary(
                self.mesh_pts)
        return bndry_indices

    def _map_samples_from_canonical_domain(self, canonical_samples):
        return _map_hypercube_samples(
            canonical_samples, self._canonical_domain_bounds,
            self._domain_bounds)

    def _map_samples_to_canonical_domain(self, samples):
        return _map_hypercube_samples(
            samples, self._domain_bounds, self._canonical_domain_bounds)

    @staticmethod
    def _form_1d_derivative_matrices(order):
        return chebyshev_derivative_matrix(order)

    def _form_derivative_matrices(self):
        canonical_mesh_pts_1d, canonical_derivative_matrices_1d = [], []
        for ii in range(self.nphys_vars):
            mpts, der_mat = self._form_1d_derivative_matrices(self._orders[ii])
            canonical_mesh_pts_1d.append(mpts)
            canonical_derivative_matrices_1d.append(der_mat)

        canonical_mesh_pts_1d_baryc_weights = [
            compute_barycentric_weights_1d(xx) for xx in canonical_mesh_pts_1d]

        mesh_pts = self._map_samples_from_canonical_domain(
            cartesian_product(canonical_mesh_pts_1d))

        if self.nphys_vars == 1:
            derivative_matrices = [
                canonical_derivative_matrices_1d[0]*2./(
                    self._domain_bounds[1]-self._domain_bounds[0])]
        else:
            ident_mats = [np.eye(o+1) for o in self._orders]
            # assumes that 2d-mesh_pts varies in x1 faster than x2,
            # e.g. points are
            # [[x11,x21],[x12,x21],[x13,x12],[x11,x22],[x12,x22],...]
            derivative_matrices = [
                np.kron(np.eye(self._orders[1]+1),
                        canonical_derivative_matrices_1d[0]*2./(
                            self._domain_bounds[1]-self._domain_bounds[0])),
                np.kron(canonical_derivative_matrices_1d[1]*2./(
                    self._domain_bounds[3]-self._domain_bounds[2]),
                        np.eye(self._orders[0]+1))]
        derivative_matrices = [torch.tensor(mat) for mat in derivative_matrices]

        return (canonical_mesh_pts_1d, canonical_derivative_matrices_1d,
                canonical_mesh_pts_1d_baryc_weights,
                mesh_pts, derivative_matrices)

    def _interpolate(self, canonical_abscissa_1d,
                     canonical_barycentric_weights_1d,
                     values, eval_samples):
        if eval_samples.ndim == 1:
            eval_samples = eval_samples[None, :]
            assert eval_samples.shape[0] == self.mesh_pts.shape[0]
        if values.ndim == 1:
            values = values[:, None]
            assert values.ndim == 2
        canonical_eval_samples = self._map_samples_to_canonical_domain(
            eval_samples)
        interp_vals = multivariate_barycentric_lagrange_interpolation(
            canonical_eval_samples, canonical_abscissa_1d,
            canonical_barycentric_weights_1d, values,
            np.arange(self.nphys_vars))
        return interp_vals

    def interpolate(self, mesh_values, eval_samples):
        canonical_abscissa_1d = self._canonical_mesh_pts_1d
        return self._interpolate(
            canonical_abscissa_1d, self._canonical_mesh_pts_1d_baryc_weights,
            mesh_values, eval_samples)

    def _plot_2d(self, mesh_values, num_pts_1d=100, ncontour_levels=20,
                 ax=None):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        # interpolate values onto plot points

        def fun(x):
            return self.interpolate(mesh_values, x)

        X, Y, Z = get_meshgrid_function_data(
            fun, self._domain_bounds, num_pts_1d, qoi=0)
        return ax.contourf(
            X, Y, Z, levels=np.linspace(Z.min(), Z.max(), ncontour_levels))

    def _plot_1d(self, mesh_values, nplot_pts_1d=None, ax=None,
                 **kwargs):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 6))[1]

        if nplot_pts_1d is not None:
            # interpolate values onto plot points
            plot_mesh = np.linspace(
                self._domain_bounds[0], self._domain_bounds[1], nplot_pts_1d)
            interp_vals = self.interpolate(mesh_values, plot_mesh)
            return ax.plot(plot_mesh, interp_vals, **kwargs)
        # just plot values on mesh points
        return ax.plot(self.mesh_pts[0, :], mesh_values, **kwargs)

    def plot(self, mesh_values, nplot_pts_1d=None, ax=None,
             **kwargs):
        if self.nphys_vars == 1:
            return self._plot_1d(
                mesh_values, nplot_pts_1d, ax, **kwargs)
        return self._plot_2d(
            mesh_values, nplot_pts_1d, 30, ax=None)

    def _apply_dirichlet_boundary_conditions_to_residual(self, residual, sol):
        for ii, bndry_cond in enumerate(self._bndry_conds):
            if bndry_cond[1] == "D":
                idx = self._bndry_indices[ii]
                bndry_vals = bndry_cond[0](self.mesh_pts[:, idx])[:, 0]
                residual[idx] = sol[idx]-bndry_vals
        return residual

    def _apply_neumann_and_robin_boundary_conditions_to_residual(
            self, residual, sol):
        for ii, bndry_cond in enumerate(self._bndry_conds):
            if bndry_cond[1] == "N" or bndry_cond[1] == "R":
                idx = self._bndry_indices[ii]
                bndry_vals = bndry_cond[0](self.mesh_pts[:, idx])[:, 0]
                normal = (-1)**(ii+1)
                if ii < 2:
                    # warning flux is not dependent on diffusivity (
                    # diffusion equation not a standard boundary formulation)
                    flux = torch.linalg.multi_dot(
                        (self._derivative_matrices[0][idx, :], sol))
                else:
                    flux = torch.linalg.multi_dot(
                        (self._derivative_matrices[1][idx, :], sol))
                residual[idx] = normal*flux-bndry_vals
                if bndry_cond[1] == "R":
                    residual[idx] += bndry_cond[2]*sol[idx]
        return residual

    def _apply_boundary_conditions_to_residual(self, residual, sol):
        residual = self._apply_dirichlet_boundary_conditions_to_residual(
            residual, sol)
        residual = (
            self._apply_neumann_and_robin_boundary_conditions_to_residual(
                residual, sol))
        return residual


class AbstractFunction(ABC):
    def __init__(self, name, requires_grad=False):
        self._name = name
        self._requires_grad = requires_grad

    @abstractmethod
    def _eval(self, samples):
        raise NotImplementedError()

    def __call__(self, samples):
        vals = self._eval(samples)
        if vals.ndim != 2:
            raise ValueError("Function must return a 2D np.ndarray")
        return torch.tensor(vals, requires_grad=self._requires_grad)


class AbstractTransientFunction(AbstractFunction):
    @abstractmethod
    def set_time(self, time):
        raise NotImplementedError()


class Function(AbstractFunction):
    def __init__(self, fun, name='fun', requires_grad=False):
        super().__init__(name, requires_grad)
        self._fun = fun

    def _eval(self, samples):
        return self._fun(samples)


class TransientFunction(AbstractFunction):
    def __init__(self, fun, name='fun', requires_grad=False):
        super().__init__(name, requires_grad)
        self._fun = fun
        self._partial_fun = None
        self._time = None

    def _eval(self, samples):
        return self._partial_fun(samples)

    def set_time(self, time):
        self._partial_fun = partial(self._fun, time=time)


class AbstractSpectralCollocationResidual(ABC):
    def __init__(self, mesh):
        self.mesh = mesh
        self._funs = None

    @abstractmethod
    def _raw_residual(self, sol):
        raise NotImplementedError()

    def _residual(self, sol):
        # correct equations for boundary conditions
        raw_residual = self._raw_residual(sol)
        return self.mesh._apply_boundary_conditions_to_residual(
            raw_residual, sol)

    def _transient_residual(self, sol, time):
        # correct equations for boundary conditions
        for fun in self._funs:
            if hasattr(fun, "set_time"):
                fun.set_time(time)
        for bndry_cond in self.mesh._bndry_conds:
            if hasattr(bndry_cond[0], "set_time"):
                bndry_cond[0].set_time(time)
        return self._raw_residual(sol)


class SteadyStatePDE():
    def __init__(self, residual):
        self.residual = residual

    def solve(self, initial_guess, tol=1e-8, maxiters=10, verbosity=0):
        initial_guess = initial_guess.squeeze()
        if type(initial_guess) == np.ndarray:
            sol = torch.tensor(initial_guess.clone(), requires_grad=True)
        else:
            sol = initial_guess.clone().detach().requires_grad_(True)
        sol = newton_solve(self.residual._residual, sol, tol, maxiters, verbosity)
        return sol.detach().numpy()[:, None]


class SteadyStateLinearPDE(SteadyStatePDE):
    def solve(self):
        init_guess = torch.ones(
            (self.residual.mesh.mesh_pts.shape[1], 1), dtype=torch.double)
        return super().solve(init_guess)


class TransientPDE():
    def __init__(self, residual, deltat, tableau_name):
        self.residual = residual
        self.time_integrator = ImplicitRungeKutta(
            deltat, self.residual._transient_residual, tableau_name,
            constraints_fun=self._apply_boundary_conditions_to_residual)

    def _apply_boundary_conditions_to_residual(self, raw_residual, sol, time):
        for bndry_cond in self.residual.mesh._bndry_conds:
            if hasattr(bndry_cond[0], "set_time"):
                bndry_cond[0].set_time(time)
        return self.residual.mesh._apply_boundary_conditions_to_residual(
            raw_residual, sol)

    def solve(self, init_sol, init_time, final_time, verbosity=0):
        sols = self.time_integrator.integrate(
            init_sol, init_time, final_time, verbosity)
        return sols


class AdvectionDiffusionReaction(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, diff_fun, vel_fun, react_fun, forc_fun):
        super().__init__(mesh)

        self._diff_fun = diff_fun
        self._vel_fun = vel_fun
        self._react_fun = react_fun
        self._forc_fun = forc_fun

        self._funs = [
            self._diff_fun, self._vel_fun, self._react_fun, self._forc_fun]

    #     self._diff_vals, self._vel_vals, self._forc_vals = (
    #         self._precompute_data())

    # def _precompute_data(self):
    #     return (self._diff_fun(self.mesh.mesh_pts),
    #             self._vel_fun(self.mesh.mesh_pts),
    #             self._forc_fun(self.mesh.mesh_pts))

    def _raw_residual(self, sol):
        # torch requires 1d arrays but to multiply each row of derivative
        # matrix by diff_vals we must use [:, None] when computing residual
        diff_vals = self._diff_fun(self.mesh.mesh_pts)
        vel_vals = self._vel_fun(self.mesh.mesh_pts)
        residual = 0
        for dd in range(self.mesh.nphys_vars):
            residual -= torch.linalg.multi_dot(
                (self.mesh._derivative_matrices[dd],
                 diff_vals*self.mesh._derivative_matrices[dd],
                 sol))
            residual += torch.linalg.multi_dot(
                (vel_vals[:, dd:dd+1]*self.mesh._derivative_matrices[dd],
                 sol))
        residual += self._react_fun(sol)
        forc_vals = self._forc_fun(self.mesh.mesh_pts)[:, 0]
        print(diff_vals, vel_vals, forc_vals)
        residual -= forc_vals
        return -residual


class EulerBernoulliBeam(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, emod_fun, smom_fun, forc_fun):
        if mesh.nphys_vars > 1:
            raise ValueError("Only 1D meshes supported")

        super().__init__(mesh)

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
        # torch requires 1d arrays but to multiply each row of derivative
        # matrix by diff_vals we must use [:, None] when computing residual
        residual = 0
        dmat = self.mesh._derivative_matrices[0]
        residual = torch.linalg.multi_dot(
            (dmat, dmat, self._emod_vals*self._smom_vals*dmat, dmat, sol))
        residual -= self._forc_vals[:, 0]
        return residual

    def _residual(self, sol):
        # correct equations for boundary conditions
        raw_residual = self._raw_residual(sol)
        dmat = self.mesh._derivative_matrices[0]
        raw_residual[0] = sol[0]-0
        raw_residual[1] = torch.linalg.multi_dot((dmat[0, :], sol))-0
        raw_residual[-1] = torch.linalg.multi_dot((torch.linalg.multi_dot(
            (dmat, dmat))[-1, :], sol))-0
        raw_residual[-2] = torch.linalg.multi_dot((torch.linalg.multi_dot(
            (dmat, dmat, dmat))[-1, :], sol))-0
        return raw_residual


class Helmholtz(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, wnum_fun, forc_fun):
        super().__init__(mesh)

        self._wnum_fun = wnum_fun
        self._forc_fun = forc_fun
        self._wnum_vals, self._forc_vals = self._precompute_data()

    def _precompute_data(self):
        return (self._wnum_fun(self.mesh.mesh_pts),
                self._forc_fun(self.mesh.mesh_pts))

    def _raw_residual(self, sol):
        # torch requires 1d arrays but to multiply each row of derivative
        # matrix by diff_vals we must use [:, None] when computing residual
        residual = 0
        for dd in range(self.mesh.nphys_vars):
            dmat = self.mesh._derivative_matrices[dd]
            residual += torch.linalg.multi_dot((dmat, dmat, sol))
        residual += self._wnum_vals[:, 0]*sol
        residual -= self._forc_vals[:, 0]
        return residual
