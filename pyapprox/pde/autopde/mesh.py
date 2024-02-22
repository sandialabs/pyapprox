import torch
from functools import partial
import matplotlib.tri as tri
import numpy as np

from pyapprox.util.utilities import cartesian_product, outer_product
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.variables.transforms import _map_hypercube_samples
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d, barycentric_interpolation_1d
)
from pyapprox.util.visualization import plt, get_meshgrid_samples
from pyapprox.pde.autopde.mesh_transforms import ScaleAndTranslationTransform


def full_fun_axis_1(fill_val, xx, oned=True):
    vals = torch.full((xx.shape[1], ), fill_val, dtype=torch.double)
    if oned:
        return vals
    else:
        return vals[:, None]


def chebyshev_derivative_matrix(order):
    if order == 0:
        pts = np.array([1], float)
        derivative_matrix = np.array([0], float)
    else:
        # this is reverse order used by matlab cheb function
        pts = -np.cos(np.linspace(0., np.pi, order+1))
        scalars = np.ones((order+1), float)
        scalars[0] = 2.
        scalars[order] = 2.
        scalars[1:order+1:2] *= -1
        derivative_matrix = np.empty((order+1, order+1), float)
        for ii in range(order+1):
            row_sum = 0.
            for jj in range(order+1):
                if (ii == jj):
                    denominator = 1.
                else:
                    denominator = pts[ii]-pts[jj]
                numerator = scalars[ii] / scalars[jj]
                derivative_matrix[ii, jj] = numerator / denominator
                row_sum += derivative_matrix[ii, jj]
            derivative_matrix[ii, ii] -= row_sum

    # I return points and calculate derivatives using reverse order of points
    # compared to what is used by Matlab cheb function thus the
    # derivative matrix I return will be the negative of the matlab version
    return pts, derivative_matrix


def _chebyshev_second_derivative_matrix_entry(degree, pts, ii, jj):
    if (ii == 0 and jj == 0) or (ii == degree and jj == degree):
        return (degree**4-1)/15

    if (ii == jj and ((ii > 0) and (ii < degree))):
        return -((degree**2-1)*(1-pts[ii]**2)+3)/(
            3*(1-pts[ii]**2)**2)

    if (ii != jj and (ii > 0 and ii < degree)):
        deriv = (-1)**(ii+jj)*(
            pts[ii]**2+pts[ii]*pts[jj]-2)/(
                (1-pts[ii]**2)*(pts[ii]-pts[jj])**2)
        if jj == 0 or jj == degree:
            deriv /= 2
        return deriv

    # because I define pts from left to right instead of right to left
    # the next two formulas are different to those in the book
    # Roger Peyret. Spectral Methods for Incompressible Viscous Flow
    # I.e. pts  = -x
    if (ii == 0 and jj > 0):
        deriv = 2/3*(-1)**jj*(
            (2*degree**2+1)*(1+pts[jj])-6)/(1+pts[jj])**2
        if jj == degree:
            deriv /= 2
        return deriv

    if ii == degree and jj < degree:
        deriv = 2/3*(-1)**(jj+degree)*(
            (2*degree**2+1)*(1-pts[jj])-6)/(1-pts[jj])**2
        if jj == 0:
            deriv /= 2
        return deriv

    raise RuntimeError("Will not happen")


def fourier_derivative_matrix(order):
    assert order % 2 == 1
    npts = (order+1)
    h = 2*np.pi/npts
    indices = np.arange(1, npts)
    col = np.hstack([0, .5*(-1)**indices*(1/np.tan(indices*h/2))])
    row = col[np.hstack([0, indices[::-1]])]
    pts = h*np.arange(1, npts+1)
    return pts, toeplitz(col, row)


def fourier_second_order_derivative_matrix(order):
    assert order % 2 == 1
    npts = (order+1)
    h = 2*np.pi/npts
    indices = np.arange(1, npts)
    col = np.hstack(
        [-np.pi**2/(3*h**2)-1/6, -.5*(-1)**indices/(np.sin(indices*h/2)**2)])
    pts = h*np.arange(1, npts+1)
    return pts, toeplitz(col)


def fourier_basis(order, samples):
    npts = (order+1)
    h = 2*np.pi/npts
    pts = h*np.arange(1, npts+1)
    II = np.where(samples==2*np.pi)[0]
    samples[II] = 0
    xx = samples[:, None]-pts[None, :]
    vals = np.sin(np.pi*xx/h)/(2*np.pi/h*np.tan(xx/2))
    return vals


def chebyshev_second_derivative_matrix(degree):
    # this is reverse order used in book
    pts = -np.cos(np.linspace(0., np.pi, degree+1))
    derivative_matrix = np.empty((degree+1, degree+1))
    for ii in range(degree+1):
        for jj in range(degree+1):
            derivative_matrix[ii, jj] = \
                _chebyshev_second_derivative_matrix_entry(degree, pts, ii, jj)
    return pts, derivative_matrix


def lagrange_polynomial_derivative_matrix_1d(eval_samples, abscissa):
    nabscissa = abscissa.shape[0]
    samples_diff = eval_samples[:, None]-abscissa[None, :]
    abscissa_diff = abscissa[:, None]-abscissa[None, :]
    basis_vals = np.ones((eval_samples.shape[0], nabscissa))
    deriv_mat = np.empty_like(basis_vals)
    for ii in range(nabscissa):
        indices = np.delete(np.arange(nabscissa), ii)
        numer = samples_diff[:, indices].prod(axis=1)
        denom = abscissa_diff[ii, indices].prod(axis=0)
        basis_vals[:, ii] = numer/denom
        numer_deriv = 0
        for jj in range(nabscissa):
            if ii != jj:
                numer_deriv += np.delete(
                    samples_diff, (ii, jj), axis=1).prod(axis=1)
        deriv_mat[:, ii] = numer_deriv/denom
    return deriv_mat, basis_vals


def lagrange_polynomial_basis_matrix_2d(eval_samples, abscissa_1d):
    nabscissa_1d = [a.shape[0] for a in abscissa_1d]
    basis_vals = np.ones((eval_samples.shape[1], np.prod(nabscissa_1d)))
    numer = [[], []]
    denom = [[], []]
    samples_diff = [None, None]
    for dd in range(2):
        samples_diff[dd] = eval_samples[dd][:, None]-abscissa_1d[dd][None, :]
        abscissa_diff = abscissa_1d[dd][:, None]-abscissa_1d[dd][None, :]
        for jj in range(nabscissa_1d[dd]):
            indices = np.delete(np.arange(nabscissa_1d[dd]), jj)
            numer[dd].append(samples_diff[dd][:, indices].prod(axis=1))
            denom[dd].append(abscissa_diff[jj, indices].prod(axis=0))
        numer[dd] = np.asarray(numer[dd])
        denom[dd] = np.asarray(denom[dd])
    cnt = 0
    for jj in range(nabscissa_1d[1]):
        basis_vals[:, cnt:cnt+nabscissa_1d[0]] = (
            numer[0][:]/denom[0][:, None]*numer[1][jj]/denom[1][jj]).T
        cnt += nabscissa_1d[0]
    return basis_vals


def lagrange_polynomial_derivative_matrix_2d(eval_samples, abscissa_1d):
    nabscissa_1d = [a.shape[0] for a in abscissa_1d]
    basis_vals = np.ones((eval_samples.shape[1], np.prod(nabscissa_1d)))
    deriv_mat = np.empty((2, eval_samples.shape[1], np.prod(nabscissa_1d)))
    numer = [[], []]
    denom = [[], []]
    samples_diff = [None, None]
    for dd in range(2):
        samples_diff[dd] = eval_samples[dd][:, None]-abscissa_1d[dd][None, :]
        abscissa_diff = abscissa_1d[dd][:, None]-abscissa_1d[dd][None, :]
        for jj in range(nabscissa_1d[dd]):
            indices = np.delete(np.arange(nabscissa_1d[dd]), jj)
            numer[dd].append(samples_diff[dd][:, indices].prod(axis=1))
            denom[dd].append(abscissa_diff[jj, indices].prod(axis=0))

    cnt = 0
    for jj in range(nabscissa_1d[1]):
        for ii in range(nabscissa_1d[0]):
            basis_vals[:, cnt] = (
                numer[0][ii]/denom[0][ii]*numer[1][jj]/denom[1][jj])
            numer_deriv_0 = 0
            for kk in range(nabscissa_1d[0]):
                if ii != kk:
                    numer_deriv_0 += np.delete(
                        samples_diff[0], (ii, kk), axis=1).prod(axis=1)
            deriv_mat[0, :, cnt] = (
                numer_deriv_0/denom[0][ii]*numer[1][jj]/denom[1][jj])

            numer_deriv_1 = 0
            for kk in range(nabscissa_1d[1]):
                if jj != kk:
                    numer_deriv_1 += np.delete(
                        samples_diff[1], (jj, kk), axis=1).prod(axis=1)
            deriv_mat[1, :, cnt] = (
                numer[0][ii]/denom[0][ii]*numer_deriv_1/denom[1][jj])
            cnt += 1
    abscissa = cartesian_product(abscissa_1d)
    return deriv_mat, basis_vals, abscissa


def kronecker_product_2d(matrix1, matrix2):
    """
    TODO: I can store kroneker as a sparse matrix see ( scipy.kron )
    """
    assert matrix1.shape == matrix2.shape
    assert matrix1.ndim == 2
    block_num_rows = matrix1.shape[0]
    matrix_num_rows = block_num_rows**2
    matrix = np.empty((matrix_num_rows, matrix_num_rows), float)

    # loop through blocks
    start_col = 0
    for jj in range(block_num_rows):
        start_row = 0
        for ii in range(block_num_rows):
            matrix[start_row:start_row+block_num_rows,
                   start_col:start_col+block_num_rows] = \
                matrix2*matrix1[ii, jj]
            start_row += block_num_rows
        start_col += block_num_rows
    return matrix


class Canonical1DMeshBoundary():
    def __init__(self, bndry_name, tol=1e-15):
        self._bndry_index = {"left": 0, "right": 1}[bndry_name]
        self._normal = torch.as_tensor(
            [[-1], [1]], dtype=torch.double)[self._bndry_index]
        self._inactive_coord = {"left": -1, "right": 1}[bndry_name]
        self._tol = tol

    def normals(self, samples):
        return torch.tile(self._normal, (1, samples.shape[1])).T

    def quadrature_rule(self):
        return np.ones((1, 1)), np.ones((1, 1))

    def samples_on_boundary(self, samples):
        return np.where(
            np.absolute(self._inactive_coord-samples[0, :]) < self._tol)[0]


class Canonical2DMeshBoundary():
    def __init__(self, bndry_name, order, tol=1e-15):
        active_bounds = np.array([-1, 1])
        if len(active_bounds) != 2:
            msg = "Bounds must be specfied for the dimension with the "
            msg += "varying coordinates"
            raise ValueError(msg)

        self._bndry_index = {
            "left": 0, "right": 1, "bottom": 2, "top": 3}[bndry_name]
        self._normal = torch.as_tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=torch.double)[self._bndry_index]
        self._order = order
        self._active_bounds = active_bounds
        self._inactive_coord = {
            "left": -1, "right": 1, "bottom": -1, "top": 1}[bndry_name]
        self._tol = tol

    def normals(self, samples):
        return torch.tile(self._normal[:, None], (1, samples.shape[1])).T

    def canonical_quadrature_rule(self):
        nsamples = self._order+3
        xquad_1d, wquad_1d = gauss_jacobi_pts_wts_1D(nsamples, 0, 0)
        xlist = [None, None]
        xlist[int(self._bndry_index < 2)] = xquad_1d
        xlist[int(self._bndry_index >= 2)] = np.array([self._inactive_coord])
        xquad = cartesian_product(xlist)
        wquad = wquad_1d[:, None]*np.prod(
            self._active_bounds[1::2]-self._active_bounds[::2])
        return xquad, wquad

    def samples_on_boundary(self, samples):
        dd = int(self._bndry_index >= 2)
        indices = np.where(
            np.absolute(self._inactive_coord-samples[dd, :]) < self._tol)[0]
        return indices


class Transformed2DMeshBoundary(Canonical2DMeshBoundary):
    # def __init__(self, bndry_name, order, bndry_deriv_vals, tol=1e-15):
    def __init__(self, bndry_name, order, normal_fun, tol=1e-15):
        super().__init__(bndry_name, order, tol)
        # self._bndary_deriv_vals = bndry_deriv_vals
        self._normal_fun = normal_fun
        self._active_var = int(self._bndry_index < 2)
        self._inactive_var = int(self._bndry_index >= 2)
        self._pts = -np.cos(np.linspace(0., np.pi, order+1))[None, :]
        self._bary_weights = compute_barycentric_weights_1d(self._pts[0, :])

    def normals(self, samples):
        if self._normal_fun is None:
            return super().normals(samples)
        normal_vals = torch.as_tensor(
            self._normal_fun(samples), dtype=torch.double)
        return normal_vals

    def _normals_from_derivs(self, canonical_samples):
        # compute normals numerically using mesh transforms
        # this will not give exact normals so we now pass in normal
        # function instead. This function is left incase it is needed in the
        # future
        surface_derivs = self._canonical_interpolate(
            self._bndary_deriv_vals, canonical_samples[self._active_var])
        normals = torch.empty((canonical_samples.shape[1], 2))
        normals[:, self._active_var] = -torch.as_tensor(
            surface_derivs, dtype=torch.double)
        normals[:, self._inactive_var] = 1
        factor = torch.sqrt(torch.sum(normals**2, axis=1))
        normals = 1/factor[:, None]*normals
        normals *= (-1)**((self._bndry_index+1) % 2)
        return normals

    def _canonical_interpolate(self, values, canonical_samples):
        return barycentric_interpolation_1d(
            self._pts[0, :], self._bary_weights, values, canonical_samples)


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
    def __init__(self, orders, basis_types=None):
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
        self._partial_derivs = [partial(self.partial_deriv, dd=dd)
                                for dd in range(self.nphys_vars)]
        self._dmats = [None for dd in range(self.nphys_vars)]

        self._flux_islinear = False
        self._flux_normal_vals = [None for dd in range(2*self.nphys_vars)]
        self._normal_vals = [None for dd in range(2*self.nphys_vars)]

    def _clear_flux_normal_vals(self):
        self._flux_normal_vals = [None for dd in range(2*self.nphys_vars)]

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
        canonical_deriv_mats = [torch.as_tensor(mat, dtype=torch.double)
                                for mat in canonical_deriv_mats]
        canonical_mesh_pts = cartesian_product(canonical_mesh_pts_1d)

        return (canonical_mesh_pts_1d, canonical_deriv_mats_1d,
                canonical_mesh_pts_1d_baryc_weights,
                canonical_mesh_pts, canonical_deriv_mats)

    def _form_boundaries(self):
        if self.nphys_vars == 1:
            return [Canonical1DMeshBoundary(name) for name in ["left", "right"]]
        return [Canonical2DMeshBoundary(name, self._orders[int(ii < 2)])
            for ii, name in enumerate(["left", "right", "bottom", "top"])]

    def _determine_boundary_indices(self):
        bndry_indices = [[] for ii in range(2*self.nphys_vars)]
        for ii in range(2*self.nphys_vars):
            bndry_indices[ii] = self._bndrys[ii].samples_on_boundary(
                self._canonical_mesh_pts)
            if ii >= 2 and self.nphys_vars == 2:
                # corners appear twice. Remove second reference to corners
                # from top and bottom boundaries
                bndry_indices[ii] = bndry_indices[ii][1:-1]
        return bndry_indices

    def interpolate(self, values, canonical_eval_samples):
        if canonical_eval_samples.ndim == 1:
            canonical_eval_samples = canonical_eval_samples[None, :]
        if values.ndim == 1:
            values = values[:, None]
            assert values.ndim == 2
            assert values.shape[0] == self.nunknowns
        return self._interpolate(values, canonical_eval_samples)

    def _interpolate(self, values, canonical_eval_samples):
        if np.all([t == "C" for t in self._basis_types]):
            return self._cheby_interpolate(
                self._canonical_mesh_pts_1d,
                self._canonical_mesh_pts_1d_baryc_weights, values,
                canonical_eval_samples)
        if np.all([t == "F" for t in self._basis_types]):
            return self._fourier_interpolate(values, canonical_eval_samples)
        raise ValueError("Mixed basis not currently supported")

    def _get_lagrange_basis_mat(self, canonical_abscissa_1d,
                                canonical_eval_samples):
        if self.nphys_vars == 1:
            return torch.as_tensor(lagrange_polynomial_derivative_matrix_1d(
                canonical_eval_samples[0, :], canonical_abscissa_1d[0])[1],
                                   dtype=torch.double)
        return torch.as_tensor(lagrange_polynomial_basis_matrix_2d(
            canonical_eval_samples, canonical_abscissa_1d), dtype=torch.double)

    def _cheby_interpolate(self, canonical_abscissa_1d,
                           canonical_barycentric_weights_1d, values,
                           canonical_eval_samples):
        # if type(values) != np.ndarray:
        #     values = values.detach().numpy()
        # interp_vals = multivariate_barycentric_lagrange_interpolation(
        #     canonical_eval_samples, canonical_abscissa_1d,
        #     canonical_barycentric_weights_1d, values,
        #     np.arange(self.nphys_vars))
        values = torch.as_tensor(values, dtype=torch.double)
        basis_mat = self._get_lagrange_basis_mat(
            canonical_abscissa_1d, canonical_eval_samples)
        interp_vals = torch.linalg.multi_dot((basis_mat, values))
        return interp_vals

    def _fourier_interpolate(self, values, canonical_eval_samples):
        if type(values) != np.ndarray:
            values = values.detach().numpy()
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

    def _plot_data_1d(self, mesh_values, nplot_pts_1d=None):
        plot_mesh = self._create_plot_mesh_1d(nplot_pts_1d)
        interp_vals = self.interpolate(mesh_values, plot_mesh[None, :])
        return interp_vals, plot_mesh

    # def _plot_1d(self, mesh_values, nplot_pts_1d=None, ax=None,
    #              **kwargs):
    #     interp_vals, plot_mesh = self._plot_data_1d(mesh_values, nplot_pts_1d)
    #     return self.plot_from_data_1d(interp_vals, plot_mesh)

    def _plot_from_data_1d(self, interp_vals, plot_mesh, ax, **kwargs):
        return ax.plot(plot_mesh, interp_vals, **kwargs)

    def _create_plot_mesh_2d(self, nplot_pts_1d):
        return get_meshgrid_samples(
            self._canonical_domain_bounds, nplot_pts_1d)

    def _plot_data_2d(self, mesh_values, nplot_pts_1d=100):
        X, Y, pts = self._create_plot_mesh_2d(nplot_pts_1d)
        Z = self._interpolate(mesh_values, pts)
        return Z, X, Y, pts

    def _plot_from_data_2d(self, Z, X, Y, pts, ax, levels=20,
                           cmap="coolwarm"):
        triang = tri.Triangulation(pts[0], pts[1])
        x = pts[0, triang.triangles].mean(axis=1)
        y = pts[1, triang.triangles].mean(axis=1)
        can_pts = self._map_samples_to_canonical_domain(
            np.vstack((x[None, :], y[None, :])))
        mask = np.where((can_pts[0] >= -1) & (can_pts[0] <= 1) &
                        (can_pts[1] >= -1) & (can_pts[1] <= 1), 0, 1)
        triang.set_mask(mask)
        if isinstance(levels, int):
            levels = np.linspace(Z.min(), Z.max(), levels)
        else:
            levels = levels
        return ax.tricontourf(triang, Z[:, 0], levels=levels, cmap=cmap)

    # def _plot_2d(self, mesh_values, nplot_pts_1d=100, levels=20,
    #              ax=None, cmap="coolwarm"):
    #     Z, X, Y, pts = self._plot_data_2d(mesh_values, nplot_pts_1d)
    #     return self._plot_2d_from_data(Z, X, Y, pts, levels, ax, cmap)

    def _plot_data(self, mesh_values, nplot_pts_1d=100):
        if self.nphys_vars == 1:
            return self._plot_data_1d(mesh_values, nplot_pts_1d)
        if nplot_pts_1d is None:
            raise ValueError("nplot_pts_1d must be not None for 2D plot")
        return self._plot_data_2d(mesh_values, nplot_pts_1d)

    def _plot_from_data(self, plot_data, ax, **kwargs):
        if self.nphys_vars == 1:
            return self._plot_from_data_1d(*plot_data, ax, **kwargs)
        return self._plot_from_data_2d(*plot_data, ax, **kwargs)

    def plot(self, mesh_values, nplot_pts_1d=100, ax=None, **kwargs):
        plot_data = self._plot_data(mesh_values, nplot_pts_1d)
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        return self._plot_from_data(plot_data, ax, **kwargs)

    def _get_quadrature_rule(self):
        quad_rules = [
            gauss_jacobi_pts_wts_1D(o+2, 0, 0) for o in self._orders]
        canonical_xquad = cartesian_product([q[0] for q in quad_rules])
        canonical_wquad = outer_product([q[1]*2 for q in quad_rules])
        return canonical_xquad, canonical_wquad

    def integrate(self, mesh_values):
        xquad, wquad = self._get_quadrature_rule()
        vals = self.interpolate(mesh_values, xquad)[:, 0]
        return vals.dot(torch.as_tensor(wquad, dtype=torch.double))

    def laplace(self, quantity):
        return laplace(self._partial_derivs, quantity)

    def partial_deriv(self, quantity, dd, idx=None):
        return partial_deriv(self._canonical_deriv_mats, quantity, dd, idx)

    def high_order_partial_deriv(self, order, quantity, dd, idx=None):
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
    def _apply_custom_boundary_conditions_to_residual(
            self, bndry_conds, residual, sol):
        for ii, bndry_cond in enumerate(bndry_conds):
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

    def _apply_dirichlet_boundary_conditions_to_residual(
            self, bndry_conds, residual, sol):
        for ii, bndry_cond in enumerate(bndry_conds):
            if bndry_cond[1] == "D":
                if self._basis_types[ii//2] == "F":
                    msg = "Cannot enforce non-periodic boundary conditions "
                    msg += "when using a Fourier basis"
                    raise ValueError(msg)
                idx = self._bndry_indices[ii]
                bndry_vals = bndry_cond[0](self.mesh_pts[:, idx])[:, 0]
                residual[idx] = sol[idx]-bndry_vals
        return residual

    def _apply_periodic_boundary_conditions_to_residual(
            self, bndry_conds, residual, sol):
        for ii in range(len(bndry_conds)//2):
            if (self._basis_types[ii] == "C" and bndry_conds[2*ii][1] == "P"):
                idx1 = self._bndry_indices[2*ii]
                idx2 = self._bndry_indices[2*ii+1]
                residual[idx1] = sol[idx1]-sol[idx2]
                residual[idx2] = (
                    self.partial_deriv(sol, ii//2, idx1) -
                    self.partial_deriv(sol, ii//2, idx2))
        return residual

    def _apply_neumann_and_robin_boundary_conditions_to_residual(
            self, bndry_conds, residual, sol, flux_jac):
        for ii, bndry_cond in enumerate(bndry_conds):
            if bndry_cond[1] == "N" or bndry_cond[1] == "R":
                if self._basis_types[ii//2] == "F":
                    msg = "Cannot enforce non-periodic boundary conditions "
                    msg += "when using a Fourier basis"
                    raise ValueError(msg)
                idx = self._bndry_indices[ii]
                bndry_vals = bndry_cond[0](self.mesh_pts[:, idx])[:, 0]
                normal_vals = self._bndrys[ii].normals(self.mesh_pts[:, idx])
                flux_jac_vals = flux_jac(idx)
                flux_vals = torch.hstack([
                    torch.linalg.multi_dot((flux_jac_vals[dd], sol))[:, None]
                    for dd in range(len(flux_jac_vals))])
                residual[idx] = self.dot(flux_vals, normal_vals)-bndry_vals
                if bndry_cond[1] == "R":
                    residual[idx] += bndry_cond[2]*sol[idx]
        return residual

    def _apply_boundary_conditions_to_residual(
            self, bndry_conds, residual, sol, flux_jac):
        residual = self._apply_dirichlet_boundary_conditions_to_residual(
            bndry_conds, residual, sol)
        residual = (
            self._apply_neumann_and_robin_boundary_conditions_to_residual(
                bndry_conds, residual, sol, flux_jac))
        residual = (self._apply_periodic_boundary_conditions_to_residual(
            bndry_conds, residual, sol))
        residual = (self._apply_custom_boundary_conditions_to_residual(
            bndry_conds, residual, sol))
        return residual

    def _dmat(self, dd):
        if self._dmats[dd] is not None:
            return self._dmats[dd]
        basis = torch.as_tensor(
            self._transform.curvelinear_basis(self._canonical_mesh_pts),
            dtype=torch.double)
        dmat = 0
        for ii in range(self.nphys_vars):
            dmat += basis[:, dd, ii:ii+1]*self._canonical_deriv_mats[ii]
            self._dmats[dd] = dmat
        return dmat

    def _apply_dirichlet_boundary_conditions_special_indexing(
            self, bndry_conds, residual, jac, sol):
        # special indexing copies data which slows down function

        # needs to have indices as argument so this fucntion can be used
        # when setting boundary conditions for forward and adjoint solves

        # it is slightly faster to set jac entries outside loop
        idx = [self._bndry_indices[ii]
               for ii in range(len(bndry_conds)) if bndry_conds[ii][1] == "D"]
        if len(idx) == 0:
            return residual, jac

        idx = np.hstack(idx)
        jac[idx, :] = 0.
        jac[idx, idx] = 1.
        for ii, bndry_cond in enumerate(bndry_conds):
            if bndry_cond[1] != "D":
                continue
            idx = self._bndry_indices[ii]
            # jac[idx, :] = 0
            # jac[idx, idx] = 1
            bndry_vals = bndry_cond[0](self.mesh_pts[:, idx])[:, 0]
            residual[idx] = sol[idx]-bndry_vals
        return residual, jac

    @staticmethod
    def _bndry_slice(vec, idx, axis):
        # avoid copying data
        if len(idx) == 1:
            if axis == 0:
                return vec[idx]
            return vec[:, idx]
        stride = idx[1]-idx[0]
        if axis == 0:
            return vec[idx[0]:idx[-1]+1:stride]
        return vec[:, idx[0]:idx[-1]+1:stride]

    def _apply_dirichlet_boundary_conditions_slicing(
            self, bndry_conds, residual, jac, sol):
        for ii, bndry_cond in enumerate(bndry_conds):
            if bndry_cond[1] != "D":
                continue
            idx = self._bndry_indices[ii]
            jac[idx, ] = 0.
            jac[idx, idx] = 1.
            bndry_vals = bndry_cond[0](
                self._bndry_slice(self.mesh_pts, idx, 1))
            assert bndry_vals.ndim == 2
            residual[idx] = self._bndry_slice(sol, idx, 0)-bndry_vals[:, 0]
        return residual, jac

    def _apply_dirichlet_boundary_conditions(
            self, bndry_conds, residual, jac, sol):
        # return self._apply_dirichlet_boundary_conditions_special_indexing(
        #     bndry_conds, residual, jac, sol)
        return self._apply_dirichlet_boundary_conditions_slicing(
            bndry_conds, residual, jac, sol)

    def _apply_neumann_and_robin_boundary_conditions(
            self, bndry_conds, residual, jac, sol, flux_jac):
        for ii, bndry_cond in enumerate(bndry_conds):
            if bndry_cond[1] != "N" and bndry_cond[1] != "R":
                continue
            idx = self._bndry_indices[ii]
            mesh_pts_idx = self._bndry_slice(self.mesh_pts, idx, 1)
            if self._normal_vals[ii] is None:
                self._normal_vals[ii] = self._bndrys[ii].normals(mesh_pts_idx)
            if not self._flux_islinear or self._flux_normal_vals[ii] is None:
                flux_jac_vals = flux_jac(idx)
                flux_normal_vals = [
                    self._normal_vals[ii][:, dd:dd+1]*flux_jac_vals[dd]
                    for dd in range(self.nphys_vars)]
                if self._flux_islinear:
                    self._flux_normal_vals[ii] = flux_normal_vals
            else:
                flux_normal_vals = self._flux_normal_vals[ii]
            # (D2*u)*n2+D2*u*n2
            jac[idx] = sum(flux_normal_vals)
            bndry_vals = bndry_cond[0](mesh_pts_idx)[:, 0]

            # residual[idx] = torch.linalg.multi_dot((jac[idx], sol))-bndry_vals
            residual[idx] = torch.linalg.multi_dot(
                (self._bndry_slice(jac, idx, 0), sol))-bndry_vals
            if bndry_cond[1] == "R":
                jac[idx, idx] += bndry_cond[2]
                # residual[idx] += bndry_cond[2]*sol[idx]
                residual[idx] += bndry_cond[2]*self._bndry_slice(sol, idx, 0)

        return residual, jac

    def _apply_periodic_boundary_conditions(
            self, bndry_conds, residual, jac, sol):
        for ii in range(len(bndry_conds)//2):
            if (self._basis_types[ii] == "C" and bndry_conds[2*ii][1] == "P"):
                idx1 = self._bndry_indices[2*ii]
                idx2 = self._bndry_indices[2*ii+1]
                jac[idx1, :] = 0
                jac[idx1, idx1] = 1
                jac[idx1, idx2] = -1
                jac[idx2] = self._dmat(ii//2)[idx1]-self._dmat(ii//2)[idx2]
                residual[idx1] = sol[idx1]-sol[idx2]
                residual[idx2] = (
                    torch.linalg.multi_dot((self._dmat(ii//2)[idx1], sol)) -
                    torch.linalg.multi_dot((self._dmat(ii//2)[idx2], sol)))
        return residual, jac

    def _apply_boundary_conditions(self, bndry_conds, residual, jac, sol,
                                   flux_jac):
        if jac is None:
            return self._apply_boundary_conditions_to_residual(
                bndry_conds, residual, sol, flux_jac), None
        residual, jac = self._apply_dirichlet_boundary_conditions(
            bndry_conds, residual, jac, sol)
        residual, jac = self._apply_periodic_boundary_conditions(
            bndry_conds, residual, jac, sol)
        residual, jac = self._apply_neumann_and_robin_boundary_conditions(
            bndry_conds, residual, jac, sol, flux_jac)
        return residual, jac

    def __repr__(self):
        return "{0}(orders={1})".format(self.__class__.__name__, self._orders)


class TransformedCollocationMesh(CanonicalCollocationMesh):
    # TODO need to changes weights of _get_quadrature_rule to account
    # for any scaling transformations

    def __init__(self, orders, transform, basis_types=None):

        super().__init__(orders, basis_types)

        self._transform = transform
        self.mesh_pts = self._map_samples_from_canonical_domain(
            self._canonical_mesh_pts)
        self._bndrys = self._transform_boundaries()
        self._dmats = [self._dmat(dd) for dd in range(self.nphys_vars)]

    def _transform_boundaries(self):
        if self.nphys_vars == 1:
            return self._bndrys
        for ii, name in enumerate(["left", "right", "bottom", "top"]):
            # active_var = int(ii > 2)
            # idx = self._bndry_indices[ii]
            self._bndrys[ii] = Transformed2DMeshBoundary(
                name, self._orders[int(ii < 2)],
                partial(self._transform.normal, ii),
                self._bndrys[ii]._tol)
        return self._bndrys

    def _map_samples_from_canonical_domain(self, canonical_samples):
        return self._transform.map_from_orthogonal(canonical_samples)

    def _map_samples_to_canonical_domain(self, samples):
        return self._transform.map_to_orthogonal(samples)

    def _interpolate(self, values, eval_samples):
        canonical_eval_samples = self._map_samples_to_canonical_domain(
            eval_samples)
        return super()._interpolate(values, canonical_eval_samples)

    def partial_deriv(self, quantity, dd, idx=None):
        assert quantity.ndim == 1
        if idx is None:
            return torch.linalg.multi_dot((self._dmat(dd), quantity))
        return torch.linalg.multi_dot((self._dmat(dd)[idx], quantity))

    def high_order_partial_deriv(self, order, quantity, dd, idx=None):
        if idx is None:
            return torch.linalg.multi_dot([self._dmat(dd)]*order+[quantity])
        return torch.linalg.multi_dot(
            (torch.linalg.multi_dot([self._dmat(dd)]*order)[idx],
             quantity))

    def _create_plot_mesh_1d(self, nplot_pts_1d):
        if nplot_pts_1d is None:
            return self.mesh_pts[0, :]
        return np.linspace(
            *self._transform._ranges, nplot_pts_1d)

    def _create_plot_mesh_2d(self, nplot_pts_1d):
        X, Y, pts = super()._create_plot_mesh_2d(nplot_pts_1d)
        pts = self._map_samples_from_canonical_domain(pts)
        return X, Y, pts

    def _get_quadrature_rule(self):
        canonical_xquad, canonical_wquad = super()._get_quadrature_rule()
        xquad = self._map_samples_from_canonical_domain(canonical_xquad)
        wquad = self._transform.modify_quadrature_weights(
            canonical_xquad, canonical_wquad)
        return xquad, wquad


class CartesianProductCollocationMesh(TransformedCollocationMesh):
    def __init__(self, domain_bounds, orders, basis_types=None):
        nphys_vars = len(orders)
        self._domain_bounds = np.asarray(domain_bounds)
        basis_types = self._get_basis_types(nphys_vars, basis_types)
        canonical_domain_bounds = (
            CanonicalCollocationMesh._get_canonical_domain_bounds(
                nphys_vars, basis_types))
        transform = ScaleAndTranslationTransform(
            canonical_domain_bounds, self._domain_bounds)
        super().__init__(
            orders, transform, basis_types=basis_types)


class VectorMesh():
    def __init__(self, meshes):
        self._meshes = meshes
        self.nunknowns = sum([m.mesh_pts.shape[1] for m in self._meshes])
        self.nphys_vars = self._meshes[0].nphys_vars

    def split_quantities(self, vector):
        cnt = 0
        split_vector = []
        for ii in range(len(self._meshes)):
            ndof = self._meshes[ii].mesh_pts.shape[1]
            split_vector.append(vector[cnt:cnt+ndof])
            cnt += ndof
        return split_vector

    def _zero_boundary_equations(self, mesh, bndry_conds, jac):
        for ii in range(len(bndry_conds)):
            if bndry_conds[ii][1] is not None:
                jac[mesh._bndry_indices[ii], :] = 0
        return jac

    def _apply_boundary_conditions_to_residual(
            self, bndry_conds, residual, sol, flux_jac):
        split_sols = self.split_quantities(sol)
        split_residual = self.split_quantities(residual)
        for ii, mesh in enumerate(self._meshes):
            split_residual[ii] = (
                mesh._apply_boundary_conditions(
                    bndry_conds[ii], split_residual[ii], None,
                    split_sols[ii], flux_jac[ii]))[0]
        return torch.cat(split_residual), None

    def _apply_boundary_conditions(
            self, bndry_conds, residual, jac, sol, flux_jac):
        if jac is None:
            return self._apply_boundary_conditions_to_residual(
                bndry_conds, residual, sol, flux_jac)

        split_sols = self.split_quantities(sol)
        split_residual = self.split_quantities(residual)
        split_jac = self.split_quantities(jac)
        for ii, mesh in enumerate(self._meshes):
            split_jac[ii] = self._zero_boundary_equations(
                mesh, bndry_conds[ii], split_jac[ii])
            ssjac = self.split_quantities(split_jac[ii].T)
            split_residual[ii], tmp = (
                mesh._apply_boundary_conditions(
                    bndry_conds[ii], split_residual[ii], ssjac[ii].T,
                    split_sols[ii], flux_jac[ii]))
            ssjac = [s.T for s in ssjac]
            ssjac[ii] = tmp
            split_jac[ii] = torch.hstack(ssjac)
        return torch.cat(split_residual), torch.vstack(split_jac)

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
                1, self.nphys_vars+1, figsize=(8*(len(sol_vals)), 6))
        if self._meshes[0].nphys_vars == 1:
            xx = np.linspace(
                *self._meshes[0]._domain_bounds, nplot_pts_1d)[None, :]
            Z = self.interpolate(sol_vals, xx)
            objs = []
            for ii in range(2):
                obj, = axs[ii].plot(xx[0, :], Z[ii], **kwargs)
                objs.append(obj)
            return objs
        X, Y, pts = get_meshgrid_samples(
            self._meshes[0]._domain_bounds, nplot_pts_1d)
        Z = self.interpolate(sol_vals, pts)
        objs = []
        for ii in range(len(Z)):
            if abs(Z[ii].min()-Z[ii].max()) <= 1e-12:
                levels = np.linspace(Z[ii].min(), Z[ii].max(), 2)
                Z[ii][0, 0] += 1e-12
                obj = axs[ii].contourf(
                    X, Y, Z[ii].reshape(X.shape), levels=levels)
            else:
                levels = np.linspace(Z[ii].min(), Z[ii].max(), 20)
                obj = axs[ii].contourf(
                    X, Y, Z[ii].reshape(X.shape), levels=levels)
            objs += (obj.collections)
        return objs


class CanonicalInteriorCollocationMesh(CanonicalCollocationMesh):
    def __init__(self, orders):
        super().__init__(orders, None)

        self._canonical_deriv_mats_alt = (
            self._form_derivative_matrices_alt())
        self._dmats_alt = [None for dd in range(self.nphys_vars)]

    def _apply_boundary_conditions_to_residual(self, bndry_conds, residual,
                                               sol):
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
            torch.as_tensor(mat, dtype=torch.double)
            for mat in canonical_deriv_mats]
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
            torch.as_tensor(mat, dtype=torch.double)
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


class TransformedInteriorCollocationMesh(CanonicalInteriorCollocationMesh):
    def __init__(self, orders, transform):

        super().__init__(orders)

        self._transform = transform
        self.mesh_pts = self._map_samples_from_canonical_domain(
            self._canonical_mesh_pts)
        self._canonical_mesh_pts_alt = cartesian_product(
                [-np.cos(np.linspace(0, np.pi, o+1)) for o in self._orders])
        self.mesh_pts_alt = self._map_samples_from_canonical_domain(
            self._canonical_mesh_pts_alt)

    def _map_samples_from_canonical_domain(self, canonical_samples):
        return self._transform.map_from_orthogonal(canonical_samples)

    def _map_samples_to_canonical_domain(self, samples):
        return self._transform.map_to_orthogonal(samples)

    def _interpolate(self, values, eval_samples):
        canonical_eval_samples = self._map_samples_to_canonical_domain(
            eval_samples)
        return super()._interpolate(values, canonical_eval_samples)

    def partial_deriv(self, quantity, dd, idx=None):
        assert quantity.ndim == 1
        if idx is None:
            return torch.linalg.multi_dot((self._dmat(quantity, dd), quantity))
        return torch.linalg.multi_dot(
            (self._dmat(quantity, dd)[idx], quantity))

    def _create_dmats(self, canonical_mesh_pts, canonical_deriv_mats):
        basis = torch.as_tensor(
            self._transform.curvelinear_basis(canonical_mesh_pts),
            dtype=torch.double)
        dmats = []
        for dd in range(self.nphys_vars):
            dmat = 0
            for ii in range(self.nphys_vars):
                dmat += basis[:, dd, ii:ii+1]*canonical_deriv_mats[ii]
            dmats.append(dmat)
        return dmats

    def _dmat_alt(self, dd):
        if self._dmats_alt[dd] is None:
            self._dmats_alt = self._create_dmats(
                self._canonical_mesh_pts_alt, self._canonical_deriv_mats)
        return self._dmats_alt[dd]

    def _dmat_full(self, dd):
        if self._dmats[dd] is None:
            self._dmats = self._create_dmats(
                self._canonical_mesh_pts, self._canonical_deriv_mats_alt)
        return self._dmats[dd]

    def _dmat(self, quantity, dd):
        if quantity.shape[0] == self.mesh_pts.shape[1]:
            alt = True
        elif quantity.shape[0] == self.mesh_pts_alt.shape[1]:
            alt = False
        else:
            RuntimeError()
        if alt:
            return self._dmat_alt(dd)
        return self._dmat_full(dd)


class InteriorCartesianProductCollocationMesh(
        TransformedInteriorCollocationMesh):
    def __init__(self, domain_bounds, orders):
        nphys_vars = len(orders)
        self._domain_bounds = np.asarray(domain_bounds)
        basis_types = self._get_basis_types(nphys_vars, None)
        canonical_domain_bounds = (
            CanonicalCollocationMesh._get_canonical_domain_bounds(
                nphys_vars, basis_types))
        transform = ScaleAndTranslationTransform(
            canonical_domain_bounds, self._domain_bounds)
        super().__init__(orders, transform)


def subdomain_integral_functional(subdomain_bounds, mesh, sol, params):
    xx_quad, ww_quad = mesh._get_quadrature_rule()
    domain_lens = (mesh._domain_bounds[1::2]-mesh._domain_bounds[0::2])
    subdomain_lens = (subdomain_bounds[1::2]-subdomain_bounds[0::2])
    xx_quad = (
        subdomain_bounds[::2, None] +
        (-mesh._domain_bounds[::2, None]+xx_quad)*(
            subdomain_lens/domain_lens)[:, None])
    ww_quad = ww_quad*np.prod(subdomain_lens/domain_lens, axis=0)
    vals = mesh.interpolate(sol, xx_quad)[:, 0]
    return vals.dot(torch.as_tensor(ww_quad, dtype=torch.double))


def final_time_functional(functional, mesh, sol, params):
    # times = np.arange(sol.shape[1])
    # generate_animation(mesh, sol, times, filename=None, maxn_frames=100,
    #                    duration=2)
    # plt.show()
    # for s in sol.T:
    #     print(s.min(), s.max())
    #     print(functional(mesh, s, params))
    # assert False
    return functional(mesh, sol[:, -1], params)


def cartesian_mesh_solution_functional(
        xx, mesh, sols, params, tt=None):
    for ii in range(mesh.nphys_vars):
        assert xx[ii, :].min() >= mesh._domain_bounds[2*ii]
        assert xx[ii, :].max() <= mesh._domain_bounds[2*ii+1]
    if tt is None:
        assert sols.ndim == 1
        return mesh.interpolate(sols, xx).flatten()
    # tt is normalized time assuming [T0, TN] -> [0, 1]
    assert tt.min() >= 0 and tt.max() <= 1
    vals0 = mesh.interpolate(sols, xx)
    from pyapprox.surrogates.interp.tensorprod import (
        piecewise_quadratic_interpolation)
    time_mesh = np.linspace(0, 1, vals0.shape[1])
    vals1 = piecewise_quadratic_interpolation(tt, time_mesh, vals0.T, [0, 1])
    return vals1.flatten()  # append each row to last


def generate_animation(mesh, sols, times, filename=None, maxn_frames=100,
                       duration=2, **kwargs):
    # duration: in seconds
    # filename:
    # osx use .mp4 extension
    # linux use.avi
    ims = []
    if isinstance(mesh, VectorMesh):
        nsols = len(mesh._meshes)
    else:
        nsols = 1
    fig, axs = plt.subplots(1, nsols, figsize=(nsols*8, 6))
    nframes = min(maxn_frames, len(times))
    stride = len(times)//nframes

    if isinstance(mesh, VectorMesh):
        raise NotImplementedError
    plot_data = []
    for tt in range(0, len(times), stride):
        sol = sols[:, tt]
        plot_data.append(mesh._plot_data(sol, nplot_pts_1d=100))
    Z_min = np.min(np.hstack([d[0] for d in plot_data]))
    Z_max = np.max(np.hstack([d[0] for d in plot_data]))
    levels = np.linspace(Z_min, Z_max, 101)
    for tt in range(0, len(times), stride):
        sol = sols[:, tt]
        im = mesh.plot(sol[:, None], ax=axs, levels=levels, **kwargs)
        ims.append(im.collections)

    # for tt in range(0, len(times), stride):
    #     if isinstance(mesh, VectorMesh):
    #         sol = mesh.split_quantities(sols[:, tt])
    #         im = mesh.plot(sol, axs=axs, **kwargs)
    #         ims.append(im)
    #     else:
    #         sol = sols[:, tt]
    #         im = mesh.plot(sol[:, None], ax=axs, **kwargs)
    #         ims.append(im.collections)

    import matplotlib.animation as animation
    nframes = len(ims)
    fps = nframes/duration  # for saving animation
    # interval in milliseconds
    interval = duration/nframes*1000  # for displaying animation
    ani = animation.ArtistAnimation(
        fig, ims, interval=interval, blit=True, repeat_delay=1000)
    plt.show()

    if filename is None:
        return ani
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(filename, writer=writervideo)
