import torch
from abc import ABC, abstractmethod
import numpy as np
from functools import partial

from pyapprox.util.utilities import cartesian_product, outer_product
from pyapprox.variables.transforms import _map_hypercube_samples
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d, barycentric_interpolation_1d,
    multivariate_barycentric_lagrange_interpolation
)
from pyapprox.pde.spectralcollocation.spectral_collocation import (
    chebyshev_derivative_matrix, lagrange_polynomial_derivative_matrix_2d,
    lagrange_polynomial_derivative_matrix_1d, fourier_derivative_matrix,
    fourier_basis
)
from pyapprox.util.visualization import (
    get_meshgrid_function_data, plt, get_meshgrid_samples
)
from pyapprox.pde.autopde.util import newton_solve
from pyapprox.pde.autopde.time_integration import ImplicitRungeKutta


class Canonical1DMeshBoundary():
    def __init__(self, bndry_name, inactive_coord, tol=1e-15):
        self._bndry_index = {"left": 0, "right": 1}[bndry_name]
        self._normal = torch.tensor([[-1], [1]])[self._bndry_index]
        self._inactive_coord = inactive_coord
        self._tol = tol

    def normals(self, samples):
        return torch.tile(self._normal, (1, samples.shape[1])).T

    def quadrature_rule(self):
        return np.ones((1, 1)), np.ones((1, 1))

    def samples_on_boundary(self, samples):
        return np.where(
            np.absolute(self._inactive_coord-samples[0, :]) < self._tol)[0]


class Canonical2DMeshBoundary():
    def __init__(self, bndry_name, inactive_coord, order, active_bounds,
                 tol=1e-15):
        if len(active_bounds) != 2:
            msg = "Bounds must be specfied for the dimension with the "
            msg += "varying coordinates"
            raise ValueError(msg)

        self._bndry_index = {"left": 0, "right": 1, "bottom": 2, "top": 3}[
            bndry_name]
        self._normal = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1]])[self._bndry_index]
        self._order = order
        self._active_bounds = active_bounds
        self._inactive_coord = inactive_coord
        self._tol = tol

    def normals(self, samples):
        return torch.tile(self._normal[:, None], (1, samples.shape[1])).T

    def quadrature_rule(self):
        nsamples = self._order+3
        xquad_1d, wquad_1d = gauss_jacobi_pts_wts_1D(nsamples, 0, 0)
        xlist = [None, None]
        xlist[int(self._bndry_index < 2)] = xquad_1d
        xlist[int(self._bndry_index >= 2)] = self._inactive_coord
        xquad = cartesian_product(xlist)
        wquad = wquad_1d[:, None]*np.prod(
            self._active_bounds[1::2]-self._active_bounds[::2])
        return xquad, wquad

    def samples_on_boundary(self, samples):
        dd = int(self._bndry_index >= 2)
        indices = np.where(
            np.absolute(self._inactive_coord-samples[dd, :]) < self._tol)[0]
        return indices


def partial_deriv(deriv_mats, quantity, dd, idx=None):
    if idx is None:
        return torch.linalg.multi_dot((deriv_mats[dd], quantity))
    return torch.linalg.multi_dot((deriv_mats[dd][idx], quantity))


def high_order_partial_deriv(order, deriv_mats, quantity, dd, idx=None):
    Dmat = torch.linalg.multi_dot([deriv_mats[dd] for ii in range(order)])
    if idx is None:
        return torch.linalg.multi_dot((Dmat, quantity))
    return torch.linalg.multi_dot((Dmat[idx], quantity))


def laplace(pderivs, quantity):
    vals = 0
    for dd in range(len(pderivs)):
        vals += pderivs[dd](pderivs[dd](quantity))
    return vals


def grad(pderivs, quantity, idx=None):
    vals = []
    for dd in range(len(pderivs)):
        vals.append(pderivs[dd](quantity, idx=idx)[:, None])
    return torch.hstack(vals)


def div(pderivs, quantities):
    vals = 0
    assert quantities.shape[1] == len(pderivs)
    for dd in range(len(pderivs)):
        vals += pderivs[dd](quantities[:, dd])
    return vals


def dot(quantities1, quantities2):
    vals = 0
    assert quantities1.shape[1] == quantities2.shape[1]
    vals = torch.sum(quantities1*quantities2, dim=1)
    return vals


class CanonicalCollocationMesh():
    def __init__(self, orders, bndry_conds, basis_types=None):
        if len(orders) > 2:
            raise ValueError("Only 1D and 2D meshes supported")
        self.nphys_vars = len(orders)
        self._basis_types = self._get_basis_types(self.nphys_vars, basis_types)
        self._orders = orders
        self._canonical_domain_bounds = self._get_canonical_domain_bounds(
            self.nphys_vars, self._basis_types)
        (self._canonical_mesh_pts_1d, self._canonical_deriv_mats_1d,
         self._canonical_mesh_pts_1d_baryc_weights, self._canonical_mesh_pts,
         self._canonical_deriv_mats) = self._form_derivative_matrices()

        self._bndrys = self._form_boundaries()
        self._bndry_indices = self._determine_boundary_indices()
        self.nunknowns = self._canonical_mesh_pts.shape[1]
        self._bndry_conds = None
        self._set_boundary_conditions(bndry_conds)
        self._partial_derivs = [partial(self.partial_deriv, dd=dd)
                                for dd in range(self.nphys_vars)]

    @staticmethod
    def _get_basis_types(nphys_vars, basis_types):
        if basis_types is None:
            basis_types = ["C"]*(nphys_vars)
        if len(basis_types) != nphys_vars:
            raise ValueError("Basis type must be specified for each dimension")
        return basis_types


    @staticmethod
    def _get_canonical_domain_bounds(nphys_vars, basis_types):
        canonical_domain_bounds = np.tile([-1, 1], nphys_vars)
        for ii in range(nphys_vars):
            if basis_types[ii] == "F":
                canonical_domain_bounds[2*ii:2*ii+2] = [0, 2*np.pi]
        return canonical_domain_bounds

    @staticmethod
    def _form_1d_derivative_matrices(order, basis_type):
        if basis_type == "C":
            return chebyshev_derivative_matrix(order)
        if basis_type == "F":
            return fourier_derivative_matrix(order)
        raise Exception(f"Basis type {basis_type} provided not supported")

    def _form_derivative_matrices(self):
        canonical_mesh_pts_1d, canonical_deriv_mats_1d = [], []
        for ii in range(self.nphys_vars):
            mpts, der_mat = self._form_1d_derivative_matrices(
                self._orders[ii], self._basis_types[ii])
            canonical_mesh_pts_1d.append(mpts)
            canonical_deriv_mats_1d.append(der_mat)

        canonical_mesh_pts_1d_baryc_weights = [
            compute_barycentric_weights_1d(xx) for xx in canonical_mesh_pts_1d]

        if self.nphys_vars == 1:
            canonical_deriv_mats = [canonical_deriv_mats_1d[0]]
        else:
            # assumes that 2d-mesh_pts varies in x1 faster than x2,
            # e.g. points are
            # [[x11,x21],[x12,x21],[x13,x12],[x11,x22],[x12,x22],...]
            canonical_deriv_mats = [
                np.kron(np.eye(self._orders[1]+1), canonical_deriv_mats_1d[0]),
                np.kron(canonical_deriv_mats_1d[1], np.eye(self._orders[0]+1))]
        canonical_deriv_mats = [torch.tensor(mat, dtype=torch.double)
                                for mat in canonical_deriv_mats]
        canonical_mesh_pts = cartesian_product(canonical_mesh_pts_1d)

        return (canonical_mesh_pts_1d, canonical_deriv_mats_1d,
                canonical_mesh_pts_1d_baryc_weights,
                canonical_mesh_pts, canonical_deriv_mats)

    def _form_boundaries(self):
        if self.nphys_vars == 1:
            return [Canonical1DMeshBoundary(name, inactive_coord)
                    for name, inactive_coord in zip(
                            ["left", "right"], self._canonical_domain_bounds)]
        return [
            Canonical2DMeshBoundary(
                name, self._canonical_domain_bounds[ii],
                self._orders[int(ii < 2)],
                self._canonical_domain_bounds[2*int(ii < 2): 2*int(ii < 2)+2])
            for ii, name in enumerate(["left", "right", "bottom", "top"])]

    def _determine_boundary_indices(self):
        bndry_indices = [[] for ii in range(2*self.nphys_vars)]
        for ii in range(2*self.nphys_vars):
            bndry_indices[ii] = self._bndrys[ii].samples_on_boundary(
                self._canonical_mesh_pts)
        return bndry_indices

    def _set_boundary_conditions(self, bndry_conds):
        if len(self._bndrys) != len(bndry_conds):
            raise ValueError(
                "Incorrect number of boundary conditions provided")
        for bndry_cond in bndry_conds:
            if bndry_cond[1] not in ["D", "R", "P", None]:
                raise ValueError(
                    "Boundary condition {bndry_cond[1} not supported")
            if (bndry_cond[1] not in [None, "P"] and
                not callable(bndry_cond[0])):
                raise ValueError("Boundary condition must be callable")
        self._bndry_conds = bndry_conds

    def interpolate(self, values, eval_samples):
        if eval_samples.ndim == 1:
            eval_samples = eval_samples[None, :]
            assert eval_samples.shape[0] == self.nunknowns
        if values.ndim == 1:
            values = values[:, None]
            assert values.ndim == 2
        return self._interpolate(values, eval_samples)

    def _interpolate(self, values, canonical_eval_samples):
        if np.all([t == "C" for t in self._basis_types]):
            return self._cheby_interpolate(
                self._canonical_mesh_pts_1d,
                self._canonical_mesh_pts_1d_baryc_weights, values,
                canonical_eval_samples)
        if np.all([t == "F" for t in self._basis_types]):
            return self._fourier_interpolate(values, canonical_eval_samples)
        raise ValueError("Mixed basis not currently supported")

    def _cheby_interpolate(self, canonical_abscissa_1d,
                           canonical_barycentric_weights_1d, values,
                           canonical_eval_samples):
        interp_vals = multivariate_barycentric_lagrange_interpolation(
            canonical_eval_samples, canonical_abscissa_1d,
            canonical_barycentric_weights_1d, values,
            np.arange(self.nphys_vars))
        return interp_vals

    def _fourier_interpolate(self, values, canonical_eval_samples):
        basis_vals = [
            fourier_basis(o, s)
            for o, s in zip(self._orders, canonical_eval_samples)]
        if self.nphys_vars == 1:
            return basis_vals[0].dot(values)
        return (basis_vals[0]*basis_vals[1]).dot(values)

    def _create_plot_mesh_1d(self, nplot_pts_1d):
        if nplot_pts_1d is None:
            return self._canonical_mesh_pts_1d[0]
        return np.linspace(
            self._canonical_domain_bounds[0],
            self._canonical_domain_bounds[1], nplot_pts_1d)

    def _plot_1d(self, mesh_values, nplot_pts_1d=None, ax=None,
                 **kwargs):
        plot_mesh = self._create_plot_mesh_1d(self, nplot_pts_1d)
        interp_vals = self.interpolate(mesh_values, plot_mesh)
        return ax.plot(plot_mesh, interp_vals, **kwargs)

    def _create_plot_mesh_2d(self, nplot_pts_1d):
        return get_meshgrid_samples(
            self._canonical_domain_bounds, nplot_pts_1d)

    def _plot_2d(self, mesh_values, nplot_pts_1d=100, ncontour_levels=20,
                 ax=None):
        X, Y, pts = self._create_plot_mesh_2d(nplot_pts_1d)
        Z = self._canonical_interpolate(mesh_values, pts).reshape(X.shape)
        return ax.tricontourf(
            X, Y, Z, levels=np.linspace(Z.min(), Z.max(), ncontour_levels))
        # return ax.contourf(
        #     X, Y, Z, levels=np.linspace(Z.min(), Z.max(), ncontour_levels))

    def plot(self, mesh_values, nplot_pts_1d=None, ax=None, **kwargs):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        if self.nphys_vars == 1:
            return self._plot_1d(
                mesh_values, nplot_pts_1d, ax, **kwargs)
        return self._plot_2d(
            mesh_values, nplot_pts_1d, 30, ax=ax)

    def _get_quadrature_rule(self):
        quad_rules = [
            gauss_jacobi_pts_wts_1D(o+2, 0, 0) for o in self._orders]
        canonical_xquad = cartesian_product([q[0] for q in quad_rules])
        canonical_wquad = outer_product([q[1] for q in quad_rules])
        return canonical_xquad, canonical_wquad

    def integrate(self, mesh_values):
        xquad, wquad = self._get_quadrature_rule()
        return self.interpolate(mesh_values, xquad)[:, 0].dot(wquad)

    def laplace(self, quantity):
        return laplace(self._partial_derivs, quantity)

    def partial_deriv(self, quantity, dd, idx=None):
        return partial_deriv(self._canonical_deriv_mats, quantity, dd, idx)

    def high_order_partial_deriv(self, orer, quantity, dd, idx=None):
        return high_order_partial_deriv(
            order, self._canonical_deriv_mats, quantity, dd, idx)

    def grad(self, quantity, idx=None):
        return grad(self._partial_derivs, quantity, idx)

    def div(self, quantities):
        return div(self._partial_derivs, quantities)

    def dot(self, quantities1, quantities2):
        return dot(quantities1, quantities2)

    # TODO remove self._bdnry_conds from mesh
    # and make property of residual or solver base class
    def _apply_custom_boundary_conditions_to_residual(self, residual, sol):
        for ii, bndry_cond in enumerate(self._bndry_conds):
            if bndry_cond[1] == "C":
                if self._basis_types[ii//2] == "F":
                    msg = "Cannot enforce non-periodic boundary conditions "
                    msg += "when using a Fourier basis"
                    raise ValueError(msg)
                idx = self._bndry_indices[ii]
                bndry_vals = bndry_cond[0](self.mesh_pts[:, idx])[:, 0]
                bndry_lhs = bndry_cond[2](sol, idx, self, ii)
                assert bndry_lhs.ndim == 1
                residual[idx] = (bndry_lhs-bndry_vals)
        return residual

    def _apply_dirichlet_boundary_conditions_to_residual(self, residual, sol):
        for ii, bndry_cond in enumerate(self._bndry_conds):
            if bndry_cond[1] == "D":
                if self._basis_types[ii//2] == "F":
                    msg = "Cannot enforce non-periodic boundary conditions "
                    msg += "when using a Fourier basis"
                    raise ValueError(msg)
                idx = self._bndry_indices[ii]
                bndry_vals = bndry_cond[0](self.mesh_pts[:, idx])[:, 0]
                residual[idx] = sol[idx]-bndry_vals
        return residual

    def _apply_periodic_boundary_conditions_to_residual(self, residual, sol):
        for ii in range(len(self._bndry_conds)//2):
            if (self._basis_types[ii] == "C" and
                self._bndry_conds[2*ii][1] == "P"):
                idx1 = self._bndry_indices[2*ii]
                idx2 = self._bndry_indices[2*ii+1]
                residual[idx1] = sol[idx1]-sol[idx2]
                residual[idx2] = (
                    self.partial_deriv(sol, ii//2, idx1) -
                    self.partial_deriv(sol, ii//2, idx2))
        return residual

    def _apply_neumann_and_robin_boundary_conditions_to_residual(
            self, residual, sol):
        for ii, bndry_cond in enumerate(self._bndry_conds):
            if bndry_cond[1] == "N" or bndry_cond[1] == "R":
                if self._basis_types[ii//2] == "F":
                    msg = "Cannot enforce non-periodic boundary conditions "
                    msg += "when using a Fourier basis"
                    raise ValueError(msg)
                idx = self._bndry_indices[ii]
                bndry_vals = bndry_cond[0](self.mesh_pts[:, idx])[:, 0]
                # TODO this will not work for non tensorial domains. The normals
                # must be mapped
                normal_vals = self._bndrys[ii].normals(self.mesh_pts[:, idx])
                # warning flux is not dependent on diffusivity (
                # diffusion equation not the usual boundary formulation used
                # for spectral Galerkin methods)
                flux_vals = self.grad(sol, idx)
                residual[idx] = self.dot(flux_vals, normal_vals)-bndry_vals
                if bndry_cond[1] == "R":
                    residual[idx] += bndry_cond[2]*sol[idx]
        return residual

    def _apply_boundary_conditions_to_residual(self, residual, sol):
        residual = self._apply_dirichlet_boundary_conditions_to_residual(
            residual, sol)
        residual = (
            self._apply_neumann_and_robin_boundary_conditions_to_residual(
                residual, sol))
        residual = (self._apply_periodic_boundary_conditions_to_residual(
            residual, sol))
        residual = (self._apply_custom_boundary_conditions_to_residual(
            residual, sol))
        return residual


class TransformedCollocationMesh(CanonicalCollocationMesh):
    def __init__(self, orders, bndry_conds, transform, transform_inv,
                 transform_inv_derivs, basis_types=None):

        super().__init__(orders, bndry_conds, basis_types)

        self._transform = transform
        self._transform_inv = transform_inv
        self._transform_inv_derivs = transform_inv_derivs

        self.mesh_pts = self._map_samples_from_canonical_domain(
            self._canonical_mesh_pts)

    def _map_samples_from_canonical_domain(self, canonical_samples):
        return self._transform(canonical_samples)

    def _map_samples_to_canonical_domain(self, samples):
        return self._transform_inv(samples)

    def _interpolate(self, values, eval_samples):
        canonical_eval_samples = self._map_samples_to_canonical_domain(
            eval_samples)
        return super()._interpolate(values, canonical_eval_samples)

    def partial_deriv(self, quantity, dd, idx=None):
        # dq/du = dq/dx * dx/du + dq/dy * dy/du
        vals = 0
        for ii in range(self.nphys_vars):
            if self._transform_inv_derivs[dd][ii] is not None:
                scale = self._transform_inv_derivs[dd][ii](
                    self._canonical_mesh_pts[:, idx])
                vals += scale*super().partial_deriv(quantity, ii, idx)
            # else: scale is zero
        return vals


def _derivatives_map_hypercube(current_range, new_range, samples):
    current_len = current_range[1]-current_range[0]
    new_len = new_range[1]-new_range[0]
    map_derivs = torch.full(
        (samples.shape[1], ), (new_len/current_len), dtype=torch.double)
    return map_derivs


class CartesianProductCollocationMesh(TransformedCollocationMesh):
    def __init__(self, domain_bounds, orders, bndry_conds, basis_types=None):
        nphys_vars = len(orders)
        self._domain_bounds = np.asarray(domain_bounds)
        basis_types = self._get_basis_types(nphys_vars, basis_types)
        canonical_domain_bounds = (
            CanonicalCollocationMesh._get_canonical_domain_bounds(
                nphys_vars, basis_types))
        transform = partial(
            _map_hypercube_samples,
            current_ranges=canonical_domain_bounds,
            new_ranges=self._domain_bounds)
        transform_inv = partial(
            _map_hypercube_samples,
            current_ranges=self._domain_bounds,
            new_ranges=canonical_domain_bounds)
        transform_inv_derivs = []
        for ii in range(nphys_vars):
            transform_inv_derivs.append([None for jj in range(nphys_vars)])
            transform_inv_derivs[ii][ii] = partial(
                _derivatives_map_hypercube,
                self._domain_bounds[2*ii:2*ii+2],
                canonical_domain_bounds[2*ii:2*ii+2])
        super().__init__(
            orders, bndry_conds, transform, transform_inv,
            transform_inv_derivs, basis_types=basis_types)

    def high_order_partial_deriv(self, order, quantity, dd, idx=None):
        # value of xx does not matter for cartesian_product meshes
        xx = np.zeros((1,1))
        deriv_mats = [tmp[0]*tmp[1][ii](xx)[0] for ii, tmp in enumerate(
            zip(self._canonical_deriv_mats, self._transform_inv_derivs))]
        return high_order_partial_deriv(
            order, self._canonical_deriv_mats, quantity, dd, idx)

    def _get_quadrature_rule(self):
        canonical_xquad, canonical_wquad = super()._get_quadrature_rule()
        xquad = self._map_samples_from_canonical_domain(canonical_xquad)
        wquad = canonical_wquad/np.prod(
            self._domain_bounds[1::2]-self._domain_bounds[::2])
        return xquad, wquad


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

    def solve(self, init_guess=None, **newton_kwargs):
        if init_guess is None:
            init_guess = torch.ones(
                (self.residual.mesh.nunknowns, 1), dtype=torch.double)
        init_guess = init_guess.squeeze()
        if type(init_guess) == np.ndarray:
            sol = torch.tensor(
                init_guess.clone(), requires_grad=True, dtype=torch.double)
        else:
            sol = init_guess.clone().detach().requires_grad_(True)
        sol = newton_solve(
            self.residual._residual, sol, **newton_kwargs)
        return sol.detach().numpy()[:, None]


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

    def solve(self, init_sol, init_time, final_time, verbosity=0,
              newton_opts={}):
        sols = self.time_integrator.integrate(
            init_sol, init_time, final_time, verbosity, newton_opts)
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
    def __init__(self, mesh, wnum_fun, forc_fun):
        super().__init__(mesh)

        self._wnum_fun = wnum_fun
        self._forc_fun = forc_fun

    def _raw_residual(self, sol):
        wnum_vals = self._wnum_fun(self.mesh.mesh_pts)
        forc_vals = self._forc_fun(self.mesh.mesh_pts)
        residual = (self.mesh.laplace(sol) + wnum_vals[:, 0]*sol -
                    forc_vals[:, 0])
        return residual


class VectorMesh():
    def __init__(self, meshes):
        self._meshes = meshes
        self.nunknowns = sum([m.mesh_pts.shape[1] for m in self._meshes])
        self.nphys_vars = self._meshes[0].nphys_vars
        self._bndry_conds = []
        for mesh in self._meshes:
            self._bndry_conds += mesh._bndry_conds

    def split_quantities(self, vector):
        cnt = 0
        split_vector = []
        for ii in range(len(self._meshes)):
            ndof = self._meshes[ii].mesh_pts.shape[1]
            split_vector.append(vector[cnt:cnt+ndof])
            cnt += ndof
        return split_vector

    def _apply_boundary_conditions_to_residual(self, residual, sol):
        split_sols = self.split_quantities(sol)
        split_residual = self.split_quantities(residual)
        for ii, mesh in enumerate(self._meshes):
            split_residual[ii] = mesh._apply_boundary_conditions_to_residual(
                split_residual[ii], split_sols[ii])
        return torch.cat(split_residual)

    def interpolate(self, sol_vals, xx):
        Z = []
        for ii in range(len(self._meshes)):
            Z.append(self._meshes[ii].interpolate(sol_vals[ii], xx))
        return Z

    def integrate(self, sol_vals):
        Z = []
        for ii in range(len(self._meshes)):
            Z.append(self._meshes[ii].integrate(sol_vals[ii]))
        return Z

    def plot(self, sol_vals, nplot_pts_1d=50, axs=None, **kwargs):
        if axs is None:
            fig, axs = plt.subplots(
                1, self.nphys_vars+1, figsize=(8*(self.nphys_vars+1), 6))
        if self._meshes[0].nphys_vars == 1:
            xx = np.linspace(
                *self._meshes[0]._domain_bounds, nplot_pts_1d)[None, :]
            Z =  self.interpolate(sol_vals, xx)
            objs = []
            for ii in range(2):
                obj = axs[ii].plot(xx[0, :], Z[ii], **kwargs)
                objs.append(obj)
            return objs
        X, Y, pts = get_meshgrid_samples(
            self._meshes[0]._domain_bounds, nplot_pts_1d)
        Z = self.interpolate(sol_vals, pts)
        objs = []
        for ii in range(3):
            obj = axs[ii].contourf(
                X, Y, Z[ii].reshape(X.shape),
                levels=np.linspace(Z[ii].min(), Z[ii].max(), 20))
            objs.append(obj)
        return objs


class CanonicalInteriorCollocationMesh(CanonicalCollocationMesh):
    def __init__(self, domain_bounds, orders):
        super().__init__(domain_bounds, orders, None)

        self._canonical_deriv_mats_alt = (
            self._form_derivative_matrices_alt())

    def _apply_boundary_conditions_to_residual(self, residual, sol):
        return residual

    def _form_canonical_deriv_matrices(self, canonical_mesh_pts_1d):
        eval_samples = cartesian_product(
            [-np.cos(np.linspace(0, np.pi, o+1)) for o in self._orders])
        if self.nphys_vars == 2:
            canonical_deriv_mats, __, canonical_mesh_pts = (
                lagrange_polynomial_derivative_matrix_2d(
                    eval_samples, canonical_mesh_pts_1d))
            return canonical_deriv_mats, canonical_mesh_pts

        return [lagrange_polynomial_derivative_matrix_1d(
            eval_samples[0], canonical_mesh_pts_1d[0])[0]], np.atleast_1d(
                canonical_mesh_pts_1d)

    def _form_derivative_matrices(self):
        # will work but divergence condition is only satisfied on interior
        # so if want to drive flow with only boundary conditions on velocity
        # it will not work
        canonical_mesh_pts_1d = [
            -np.cos(np.linspace(0, np.pi, o+1))[1:-1] for o in self._orders]
        canonical_mesh_pts_1d_baryc_weights = [
            compute_barycentric_weights_1d(xx) for xx in canonical_mesh_pts_1d]
        canonical_deriv_mats, canonical_mesh_pts = (
            self._form_canonical_deriv_matrices(canonical_mesh_pts_1d))
        canonical_deriv_mats = [
            torch.tensor(mat, dtype=torch.double) for mat in canonical_deriv_mats]
        return (canonical_mesh_pts_1d, None,
                canonical_mesh_pts_1d_baryc_weights,
                canonical_mesh_pts, canonical_deriv_mats)

    def _form_derivative_matrices_alt(self):
        canonical_mesh_pts_1d = [
            -np.cos(np.linspace(0, np.pi, o+1))[1:-1] for o in self._orders]
        if self.nphys_vars == 2:
            canonical_deriv_mats_alt = (
                lagrange_polynomial_derivative_matrix_2d(
                    cartesian_product(canonical_mesh_pts_1d),
                    [-np.cos(np.linspace(0, np.pi, o+1))
                     for o in self._orders])[0])
        else:
            canonical_deriv_mats_alt = [
                lagrange_polynomial_derivative_matrix_1d(
                    canonical_mesh_pts_1d[0],
                    -np.cos(np.linspace(0, np.pi, self._orders[0]+1)))[0]]
        canonical_deriv_mats_alt = [
            torch.tensor(mat, dtype=torch.double)
            for mat in canonical_deriv_mats_alt]
        return canonical_deriv_mats_alt

    def _get_canonical_deriv_mats(self, quantity):
        if quantity.shape[0] == self.nunknowns:
            return self._canonical_deriv_mats
        elif quantity.shape[0] == self._canonical_deriv_mats_alt[0].shape[1]:
            return self._canonical_deriv_mats_alt
        raise RuntimeError("quantity is the wrong shape")

    def _determine_boundary_indices(self):
        self._boundary_indices = None

    def partial_deriv(self, quantity, dd, idx=None):
        return partial_deriv(
            self._get_canonical_deriv_mats(quantity), quantity, dd, idx)


class TransformedInteriorCollocationMesh(CanonicalInteriorCollocationMesh):
    def __init__(self, orders, bndry_conds, transform, transform_inv,
                 transform_inv_derivs):

        super().__init__(orders, bndry_conds)

        self._transform = transform
        self._transform_inv = transform_inv
        self._transform_inv_derivs = transform_inv_derivs

        self.mesh_pts = self._map_samples_from_canonical_domain(
            self._canonical_mesh_pts)

    def _map_samples_from_canonical_domain(self, canonical_samples):
        return self._transform(canonical_samples)

    def _map_samples_to_canonical_domain(self, samples):
        return self._transform_inv(samples)

    def _interpolate(self, values, eval_samples):
        canonical_eval_samples = self._map_samples_to_canonical_domain(
            eval_samples)
        return super()._interpolate(values, canonical_eval_samples)

    def partial_deriv(self, quantity, dd, idx=None):
        # dq/du = dq/dx * dx/du + dq/dy * dy/du
        vals = 0
        for ii in range(self.nphys_vars):
            if self._transform_inv_derivs[dd][ii] is not None:
                scale = self._transform_inv_derivs[dd][ii](
                    self._canonical_mesh_pts[:, idx])
                vals += scale*super().partial_deriv(quantity, ii, idx)
            # else: scale is zero
        return vals


class InteriorCartesianProductCollocationMesh(TransformedInteriorCollocationMesh):
    def __init__(self, domain_bounds, orders):
        nphys_vars = len(orders)
        self._domain_bounds = np.asarray(domain_bounds)
        basis_types = self._get_basis_types(nphys_vars, None)
        canonical_domain_bounds = (
            CanonicalCollocationMesh._get_canonical_domain_bounds(
                nphys_vars, basis_types))
        transform = partial(
            _map_hypercube_samples,
            current_ranges=canonical_domain_bounds,
            new_ranges=self._domain_bounds)
        transform_inv = partial(
            _map_hypercube_samples,
            current_ranges=self._domain_bounds,
            new_ranges=canonical_domain_bounds)
        transform_inv_derivs = []
        for ii in range(nphys_vars):
            transform_inv_derivs.append([None for jj in range(nphys_vars)])
            transform_inv_derivs[ii][ii] = partial(
                _derivatives_map_hypercube,
                self._domain_bounds[2*ii:2*ii+2],
                canonical_domain_bounds[2*ii:2*ii+2])
        bndry_conds = [[None, None] for ii in range(nphys_vars*2)]
        super().__init__(
            orders, bndry_conds, transform, transform_inv,
            transform_inv_derivs)


class NavierStokes(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, vel_forc_fun, pres_forc_fun,
                 unique_pres_data=(0, 1)):
        super().__init__(mesh)

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
        nvel_unknowns = self.mesh._meshes[0].nunknowns
        residual[-1] = (
            self.mesh._meshes[-1].div(vel_sols) -
            self._pres_forc_fun(self.mesh._meshes[-1].mesh_pts)[:, 0])
        residual[-1][self._unique_pres_data[0]] = (
            split_sols[-1][self._unique_pres_data[0]]-self._unique_pres_data[1])
        return torch.cat(residual)


class LinearStokes(NavierStokes):
    def __init__(self, mesh, vel_forc_fun, pres_forc_fun,
                 unique_pres_data=(0, 1)):
        super().__init__(mesh, vel_forc_fun, pres_forc_fun,
                         unique_pres_data)
        self._navier_stokes = False


class ShallowWater(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, depth_forc_fun, vel_forc_fun, bed_fun):
        super().__init__(mesh)

        self._depth_forc_fun = depth_forc_fun
        self._vel_forc_fun = vel_forc_fun
        self._g = 9.81

        self._funs = [self._depth_forc_fun, self._vel_forc_fun]
        self._bed_vals = bed_fun(self.mesh._meshes[0].mesh_pts)

    def _raw_residual_1d(self, depth, vels, depth_forc_vals, vel_forc_vals):
        pderiv = self.mesh._meshes[0].partial_deriv
        residual = [0, 0]
        residual[0] = pderiv(depth*vels[:, 0], 0)-depth_forc_vals[:, 0]
        residual[1] = pderiv(depth*vels[:, 0]**2+self._g*depth**2/2, 0)
        residual[1] += self._g*depth*pderiv(self._bed_vals[:, 0], 0)
        residual[1] -= vel_forc_vals[:, 0]
        return torch.cat(residual)

    def _raw_residual_2d(self, depth, vels, depth_forc_vals, vel_forc_vals):
        pderiv = self.mesh._meshes[0].partial_deriv
        residual = [0, 0, 0]
         # depth equation (mass balance)
        for dd in range(self.mesh.nphys_vars):
            residual[0] += self.mesh._meshes[0].partial_deriv(
                depth*vels[:, dd], dd)
        residual[0] -= depth_forc_vals[:, 0]
        # velocity equations (momentum equations)
        residual[1] = pderiv(depth*vels[:, 0]**2+self._g*depth**2/2, 0)
        residual[1] += pderiv(depth*torch.prod(vels, dim=1), 1)
        residual[1] += self._g*depth*pderiv(self._bed_vals[:, 0], 0)
        residual[1] -= vel_forc_vals[:, 0]
        residual[2] = pderiv(depth*torch.prod(vels, dim=1), 0)
        residual[2] += pderiv(depth*vels[:, 1]**2+self._g*depth**2/2, 1)
        residual[2] += self._g*depth*pderiv(self._bed_vals[:, 0], 1)
        residual[2] -= vel_forc_vals[:, 1]
        return torch.cat(residual)

    def _raw_residual(self, sol):
        split_sols = self.mesh.split_quantities(sol)
        depth = split_sols[0]
        vels = torch.hstack([s[:, None] for s in split_sols[1:]])
        depth_forc_vals = self._depth_forc_fun(self.mesh._meshes[0].mesh_pts)
        vel_forc_vals = self._vel_forc_fun(self.mesh._meshes[1].mesh_pts)

        if self.mesh.nphys_vars == 1:
            return self._raw_residual_1d(
                depth, vels, depth_forc_vals, vel_forc_vals)

        return self._raw_residual_2d(
                depth, vels, depth_forc_vals, vel_forc_vals)

        residual = [0 for ii in range(len(split_sols))]
        # depth equation (mass balance)
        for dd in range(self.mesh.nphys_vars):
            # split_sols = [q1, q2] = [h, u, v]
            residual[0] += self.mesh._meshes[0].partial_deriv(
                depth*vels[:, dd], dd)
        residual[0] -= depth_forc_vals[:, 0]
        # velocity equations (momentum equations)
        for dd in range(self.mesh.nphys_vars):
            # split_sols = [q1, q2] = [h, u, v]
            residual[dd+1] += self.mesh._meshes[dd].partial_deriv(
                depth*vels[:, dd]**2+self._g*depth**2/2, dd)
            # split_sols = [q1, q2] = [h, uh, vh]
            # residual[dd+1] += self.mesh._meshes[dd].partial_deriv(
            #     vels[:, dd]**2/depth+self._g*depth**2/2, dd)
            residual[dd+1] += self._g*depth*self.mesh._meshes[dd].partial_deriv(
                self._bed_vals[:, 0], dd)
            residual[dd+1] -= vel_forc_vals[:, dd]
        if self.mesh.nphys_vars > 1:
            residual[1] += self.mesh._meshes[1].partial_deriv(
                depth*torch.prod(vels, dim=1), 1)
            residual[2] += self.mesh._meshes[2].partial_deriv(
                depth*torch.prod(vels, dim=1), 0)
        return torch.cat(residual)


class ShallowShelfVelocities(AbstractSpectralCollocationResidual):
    def __init__(self, mesh, forc_fun, bed_fun, beta_fun,
                 depth_fun, A, rho, homotopy_val=0):
        super().__init__(mesh)

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
    def __init__(self, mesh, forc_fun, bed_fun, beta_fun,
                 depth_forc_fun, A, rho, homotopy_val=0):
        if len(mesh._meshes) != mesh._meshes[0].nphys_vars+1:
            raise ValueError("Incorrect number of meshes provided")

        super().__init__(mesh, forc_fun, bed_fun, beta_fun,
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
    def __init__(self, mesh, forc_fun, lambda_fun, mu_fun, rho):
        super().__init__(mesh)

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
    def __init__(self, mesh, forc_fun, bed_fun, beta_fun,
                 depth_fun, A, rho, homotopy_val=0):
        super().__init__(mesh)

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
            # plt.plot(mesh_pts[0], dudx_ij[0][0][idx], '--o')
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
        multi_dot = torch.linalg.multi_dot
        if bndry_index < 2:
            return self._vecs[0][idx, 0]*(-1)**((bndry_index+1) % 2)

        # derivative of surface
        dsdx = multi_dot(
            (mesh._deriv_mats[0][idx, :], self._surface_vals))
        normals = torch.vstack((dsdx.T, torch.zeros((1, idx.shape[0]))))
        normals /= torch.sqrt(torch.sum(normals**2, dim=0))
        vals = torch.sum(self._vecs[0][idx, :]*normals.T, dim=1)
        if bndry_index == 3:
            return vals

        # negative sign on vals because normals are reversed
        return -vals + self._beta_vals[idx, 0]*sol[idx]
