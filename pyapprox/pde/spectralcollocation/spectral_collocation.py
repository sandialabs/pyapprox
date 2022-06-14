from abc import ABC, abstractmethod
import numpy as np
from functools import partial
from scipy.linalg import toeplitz

from pyapprox.util.utilities import (
    cartesian_product, outer_product, get_tensor_product_quadrature_rule
)
from pyapprox.util.sys_utilities import get_num_args
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d, barycentric_interpolation_1d,
    multivariate_barycentric_lagrange_interpolation
)
from pyapprox.variables.transforms import _map_hypercube_samples


from pyapprox.util.sys_utilities import package_available
# def package_available(name): # hack to turn off torch use
#     return False

if package_available("torch"):
    import torch as pkg
else:
    import numpy as pkg


def np_to_pkg_format(array):
    if package_available("torch"):
        return pkg.tensor(array)
    return array


def zeros_fun_axis_1(x):
    # axis_1 used when x is mesh points
    return np.zeros((x.shape[1], 1))


def ones_fun_axis_0(x):
    # axis_1 used when x is solution like quantity
    return np.ones((x.shape[0], 1))


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


def integrate_subdomain(mesh_values, subdomain, order, interpolate,
                        quad_rule=None):
    # Keep separate from model class so that pre-computed so
    # it can also be used by QoI functions
    if quad_rule is None:
        quad_pts, quad_wts = get_2d_sub_domain_quadrature_rule(
            subdomain, order)
    else:
        quad_pts, quad_wts = quad_rule
    quad_vals = interpolate(mesh_values, quad_pts)
    # Compute and return integral
    return np.dot(quad_vals[:, 0], quad_wts)


def get_2d_sub_domain_quadrature_rule(subdomain, order):
    pts_1d, wts_1d = [], []
    for ii in range(2):
        gl_pts, gl_wts = gauss_jacobi_pts_wts_1D(order[ii], 0, 0)
        # Scale points from [-1,1] to to physical domain
        x_range = subdomain[2*ii+1]-subdomain[2*ii]
        # Remove factor of 0.5 from weights and shift to [a,b]
        wts_1d.append(gl_wts*x_range)
        pts_1d.append(x_range*(gl_pts+1.)/2.+subdomain[2*ii])
    # Interpolate mesh values onto quadrature nodes
    pts = cartesian_product(pts_1d)
    wts = outer_product(wts_1d)
    return pts, wts


class AbstractCartesianProductCollocationMesh(ABC):
    def __init__(self, domain, order):
        self.domain = None
        self.nphys_vars = None
        self.order = None
        self.canonical_domain = None
        self.quad_pts = None
        self.quad_wts = None
        self.mesh_pts_1d = None
        self.derivative_matrices_1d = None
        self.mesh_pts = None
        self.bndry_conds = None
        self.adjoint_bndry_conds = None
        self.boundary_indices = None
        self.derivative_matrices = []
        self._mesh_pts_1d_barycentric_weights = None
        self.set_domain(domain, order)
        self._arange_nphys_vars = np.arange(self.nphys_vars)
        # Note all variables set by member function
        # defined in this abstract class must be defined here
        # or sometimes when a derived class is created these variables
        # will not be populated when scope returned to outside class

    def _expand_order(self, order):
        order = np.atleast_1d(order).astype(int)
        assert order.ndim == 1
        if order.shape[0] == 1 and self.nphys_vars > 1:
            order = np.array([order[0]]*self.nphys_vars, dtype=int)
        if order.shape[0] != self.nphys_vars:
            msg = "order must be a scalar or an array_like with an entry for"
            msg += " each physical dimension"
            raise ValueError(msg)
        return order

    def set_domain(self, domain, order):
        self.domain = np.asarray(domain)
        self.nphys_vars = self.domain.shape[0]//2
        self.order = self._expand_order(order)
        self.canonical_domain = np.ones(2*self.nphys_vars)
        self.canonical_domain[::2] = -1.
        self.form_derivative_matrices()
        self.quad_pts, self.quad_wts = self.form_quadrature_rule(
            self.order, self.domain)
        self._determine_boundary_indices()

    def map_samples_from_canonical_domain(self, canonical_pts):
        pts = _map_hypercube_samples(
            canonical_pts, self.canonical_domain, self.domain)
        return pts

    def map_samples_to_canonical_domain(self, pts):
        return _map_hypercube_samples(
            pts, self.domain, self.canonical_domain)

    def form_quadrature_rule(self, order, domain):
        univariate_quad_rules = [
            partial(gauss_jacobi_pts_wts_1D, alpha_poly=0, beta_poly=0)
        ]*self.nphys_vars
        quad_pts, quad_wts = \
            get_tensor_product_quadrature_rule(
                order, self.nphys_vars, univariate_quad_rules,
                transform_samples=self.map_samples_from_canonical_domain)
        quad_wts *= np.prod(domain[1::2]-domain[::2])
        return quad_pts, quad_wts

    def _form_1d_derivative_matrices(self, order):
        return chebyshev_derivative_matrix(order)

    def form_derivative_matrices(self):
        self.mesh_pts_1d, self.derivative_matrices_1d = [], []
        for ii in range(self.nphys_vars):
            mpts, der_mat = self._form_1d_derivative_matrices(self.order[ii])
            self.mesh_pts_1d.append(mpts)
            self.derivative_matrices_1d.append(der_mat)

        self._mesh_pts_1d_barycentric_weights = [
            compute_barycentric_weights_1d(xx) for xx in self.mesh_pts_1d]

        self.mesh_pts = self.map_samples_from_canonical_domain(
            cartesian_product(self.mesh_pts_1d))
        self._form_derivative_matrices()

    @abstractmethod
    def _form_derivative_matrices(self):
        raise NotImplementedError()

    @abstractmethod
    def _determine_boundary_indices(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_bndry_normals(self, bndry_indices):
        raise NotImplementedError()

    def integrate(self, mesh_values, order=None, domain=None, qoi=None):
        if order is None:
            order = self.order
            quad_pts, quad_wts = self.quad_pts, self.quad_wts
        else:
            if domain is None:
                domain = self.domain
            assert np.all(domain[::2] >= self.domain[::2])
            assert np.all(domain[1::2] <= self.domain[1::2])
            order = self._expand_order(order)
            quad_pts, quad_wts = self.form_quadrature_rule(order, domain)
        if qoi is None:
            qoi = mesh_values.shape[1]-1

        quad_vals = self.interpolate(mesh_values[:, qoi:qoi+1], quad_pts)
        # Compute and return integral
        return np.dot(quad_vals[:, 0], quad_wts)

    def _interpolate(self, canonical_abscissa_1d,
                     canonical_barycentric_weights_1d,
                     values, eval_samples):
        if eval_samples.ndim == 1:
            eval_samples = eval_samples[None, :]
            assert eval_samples.shape[0] == self.mesh_pts.shape[0]
        if values.ndim == 1:
            values = values[:, None]
            assert values.ndim == 2
        eval_samples = _map_hypercube_samples(
            eval_samples, self.domain, self.canonical_domain)
        interp_vals = multivariate_barycentric_lagrange_interpolation(
            eval_samples,
            canonical_abscissa_1d,
            canonical_barycentric_weights_1d,
            values,
            self._arange_nphys_vars)
        return interp_vals

    def interpolate(self, mesh_values, eval_samples):
        canonical_abscissa_1d = self.mesh_pts_1d
        return self._interpolate(
            canonical_abscissa_1d, self._mesh_pts_1d_barycentric_weights,
            mesh_values, eval_samples)

    def set_boundary_conditions(self, bndry_conds):
        """
        Set time independent boundary conditions. If time dependent BCs
        are needed. Then this function must be called at every timestep
        """
        self.bndry_conds = bndry_conds

        if len(bndry_conds) != len(self.boundary_indices):
            msg = "Incorrect number of boundary conditions provided.\n"
            msg += f"\tNum boundary edges {len(self.boundary_indices)}\n"
            msg += f"\tNum boundary conditions provided {len(bndry_conds)}\n"
            raise ValueError(msg)

        # increase this if make boundary time dependent and/or
        # parameter dependent
        num_args = 1
        self.adjoint_bndry_conds = []
        for ii, bndry_cond in enumerate(bndry_conds):
            if bndry_cond[1] == "R":
                assert len(bndry_cond) == 3
            else:
                assert len(bndry_cond) == 2
            assert callable(bndry_cond[0])
            if get_num_args(bndry_cond[0]) != num_args:
                msg = f"Boundary condition function must have {num_args} "
                msg += "arguments"
                raise ValueError(msg)
            self.adjoint_bndry_conds.append([zeros_fun_axis_1, "D"])

    def _apply_dirichlet_boundary_conditions_to_matrix(
            self, matrix, bndry_conds):
        # needs to have indices as argument so this fucntion can be used
        # when setting boundary conditions for forward and adjoint solves
        for ii, bndry_cond in enumerate(bndry_conds):
            idx = self.boundary_indices[ii]
            matrix[idx, :] = 0
            matrix[idx, idx] = 1
        return matrix

    def _apply_neumann_and_robin_boundary_conditions_to_matrix(self, matrix):
        for ii, bndry_cond in enumerate(self.bndry_conds):
            if bndry_cond[1] == "N" or bndry_cond[1] == "R":
                idx = self.boundary_indices[ii]
                normal = (-1)**(ii+1)
                if ii < 2:
                    # warning flux is not dependent on diffusivity (
                    # diffusion equation not a standard boundary formulation)
                    matrix[idx, :] = normal*self.derivative_matrices[0][idx, :]
                else:
                    matrix[idx, :] = normal*self.derivative_matrices[1][idx, :]
                if bndry_cond[1] == "R":
                    matrix[idx, idx] += bndry_cond[2]
        return matrix

    def _apply_dirichlet_boundary_conditions_to_residual(self, residual, sol):
        for ii, bndry_cond in enumerate(self.pkg_bndry_conds):
            if bndry_cond[1] == "D":
                idx = self.boundary_indices[ii]
                residual[idx] = sol[idx]-bndry_cond[0](
                    self.mesh_pts[:, idx])
        return residual

    def _apply_neumann_and_robin_boundary_conditions_to_residual(
            self, residual, sol):
        for ii, bndry_cond in enumerate(self.pkg_bndry_conds):
            if bndry_cond[1] == "N" or bndry_cond[1] == "R":
                idx = self.boundary_indices[ii]
                bndry_vals = bndry_cond[0](self.mesh_pts[:, idx])
                normal = (-1)**(ii+1)
                if ii < 2:
                    # warning flux is not dependent on diffusivity (
                    # diffusion equation not a standard boundary formulation)
                    flux = pkg.linalg.multi_dot(
                        (self.pkg_derivative_matrices[0][idx, :], sol))
                else:
                    flux = pkg.linalg.multi_dot(
                        (self.pkg_derivative_matrices[1][idx, :], sol))
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

    def _apply_boundary_conditions_to_matrix(self, matrix):
        matrix = self._apply_dirichlet_boundary_conditions_to_matrix(
            matrix, self.bndry_conds)
        matrix = self._apply_neumann_and_robin_boundary_conditions_to_matrix(
            matrix)
        return matrix

    def _apply_boundary_conditions_to_rhs(self, rhs, bndry_conds):
        for ii, bndry_cond in enumerate(bndry_conds):
            idx = self.boundary_indices[ii]
            rhs[idx] = bndry_cond[0](self.mesh_pts[:, idx])
        return rhs


class OneDCollocationMesh(AbstractCartesianProductCollocationMesh):
    def _form_derivative_matrices(self):
        if self.domain.shape[0] != 2:
            raise ValueError("Domain must be 1D")
        self.derivative_matrices = [
            self.derivative_matrices_1d[0]*2./(self.domain[1]-self.domain[0])]

    def _determine_boundary_indices(self):
        self.boundary_indices = [
            np.array([0]), np.array(
                [self.derivative_matrices[0].shape[0]-1])]

    def _get_bndry_normal(self, bndry_index):
        if bndry_index == 0:
            return -1.
        return 1.

    def _get_bndry_normals(self, bndry_indices):
        normals = np.empty((len(bndry_indices), self.nphys_vars))
        for ii, bndry_index in enumerate(bndry_indices):
            normals[ii] = self._get_bndry_normal(bndry_index)
        return normals

    def plot(self, mesh_values, num_plot_pts_1d=None, ax=None, 
             **kwargs):
        import pylab as plt
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if num_plot_pts_1d is not None:
            # interpolate values onto plot points
            plot_mesh = np.linspace(
                self.domain[0], self.domain[1], num_plot_pts_1d)
            interp_vals = self.interpolate(mesh_values, plot_mesh)
            ax.plot(plot_mesh, interp_vals, **kwargs)
        else:
            # just plot values on mesh points
            ax.plot(self.mesh_pts[0, :], mesh_values, **kwargs)


class RectangularCollocationMesh(AbstractCartesianProductCollocationMesh):
    def __init__(self, domain, order):
        # because __init__ calls function that sets self.boundary_indices
        # first define default value for all new attributes then call
        # super
        self.boundary_indices_to_edges_map = None
        self.lr_neumann_bndry_indices = None
        self.bt_neumann_bndry_indices = None
        self.lr_robin_bndry_indices = None
        self.bt_robin_bndry_indices = None
        self._bndry_segment_canonical_abscissa = None
        self._bndry_segment_canonical_bary_weights = None
        super().__init__(domain, order)
        self._bndry_normals = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self._set_bndry_segment_interpolation_data()

    def _form_derivative_matrices(self):
        if self.domain.shape[0] != 4:
            raise ValueError("Domain must be 2D")
        ident1 = np.eye(self.order[1]+1)
        ident2 = np.eye(self.order[0]+1)
        self.derivative_matrices.append(np.kron(
            ident1,
            self.derivative_matrices_1d[0]*2./(self.domain[1]-self.domain[0])))
        # form derivative (in x2-direction) matrix of a 2d polynomial
        # this assumes that 2d-mesh_pts varies in x1 faster than x2,
        # e.g. points are
        # [[x11,x21],[x12,x21],[x13,x12],[x11,x22],[x12,x22],...]
        self.derivative_matrices.append(np.kron(
            self.derivative_matrices_1d[1],
            ident2*2./(self.domain[3]-self.domain[2])))

    def _determine_boundary_indices(self):
        # boundary edges are stored with the following order,
        # left, right, bottom, top
        self.boundary_indices = [[] for ii in range(4)]
        self.boundary_indices_to_edges_map = -np.ones(self.mesh_pts.shape[1])
        # Todo avoid double counting the bottom and upper boundaries
        tol = 1e-15
        for ii in range(self.mesh_pts.shape[1]):
            if np.allclose(self.mesh_pts[0, ii], self.domain[0], atol=tol):
                self.boundary_indices[0].append(ii)
                self.boundary_indices_to_edges_map[ii] = 0
            if np.allclose(self.mesh_pts[0, ii], self.domain[1], atol=tol):
                self.boundary_indices[1].append(ii)
                self.boundary_indices_to_edges_map[ii] = 1
            if (np.allclose(self.mesh_pts[1, ii], self.domain[2], atol=tol) and
                not np.allclose(self.mesh_pts[0, ii], self.domain[0], atol=tol) and
                not np.allclose(self.mesh_pts[0, ii], self.domain[1], atol=tol)):
                self.boundary_indices[2].append(ii)
                self.boundary_indices_to_edges_map[ii] = 2
            if (np.allclose(self.mesh_pts[1, ii], self.domain[3], atol=tol) and
                not np.allclose(self.mesh_pts[0, ii], self.domain[0], atol=tol) and
                not np.allclose(self.mesh_pts[0, ii], self.domain[1], atol=tol)):
                self.boundary_indices[3].append(ii)
                self.boundary_indices_to_edges_map[ii] = 3

        self.boundary_indices = [
            np.array(idx, dtype=int) for idx in self.boundary_indices]

        nbdry_dof = np.sum(
            [indices.shape[0] for indices in self.boundary_indices])
        if nbdry_dof != 2*(self.order[0]+1)+2*(self.order[1]+1)-4:
            raise RuntimeError("Ndof on boundary is incorrect")

    def _get_bndry_normals(self, bndry_indices):
        return self._bndry_normals[bndry_indices]
   
    def plot(self, mesh_values, num_pts_1d=100, ncontour_levels=20, ax=None):
        from pyapprox.util.visualization import get_meshgrid_function_data, plt
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # interpolate values onto plot points
        def fun(x): return self.interpolate(mesh_values, x)
        X, Y, Z = get_meshgrid_function_data(
            fun, self.domain, num_pts_1d, qoi=0)
        return ax.contourf(
            X, Y, Z, levels=np.linspace(Z.min(), Z.max(), ncontour_levels))

    def _bndry_segment_mesh(self, segment_id):
        return self.mesh_pts[:, self.boundary_indices[segment_id]]

    def _set_bndry_segment_interpolation_data(self):
        self._bndry_segment_canonical_abscissa = []
        self._bndry_segment_canonical_bary_weights = []
        for segment_id in range(4):
            segment_mesh = self._bndry_segment_mesh(segment_id)
            canonical_segment_mesh = self.map_samples_to_canonical_domain(
                segment_mesh)
            if segment_id < 2:
                canonical_abscissa_1d = [canonical_segment_mesh[0, :1],
                                         canonical_segment_mesh[1, :]]
            else:
                canonical_abscissa_1d = [canonical_segment_mesh[0, :],
                                         canonical_segment_mesh[1, :1]]
            self._bndry_segment_canonical_abscissa.append(
                canonical_abscissa_1d)
            self._bndry_segment_canonical_bary_weights.append(
                [compute_barycentric_weights_1d(xx)
                 for xx in canonical_abscissa_1d])

    def _interpolate_bndry_segment(self, segment_id, values, samples):
        # if segment_id < 2:
        #     if not np.allclose(samples[0, :], segment_mesh[0, 0]):
        #         print(samples, segment_mesh[0, 0])
        #         raise RuntimeError()
        # else:
        #     if not np.allclose(samples[1, :], segment_mesh[1, 0]):
        #         raise RuntimeError()
        # interp_vals = self._interpolate(
        #     self._bndry_segment_canonical_abscissa[segment_id],
        #     self._bndry_segment_canonical_bary_weights[segment_id],
        #     values, samples)
        # return interp_vals
        assert values.ndim == 1
        dim = int(segment_id < 2)
        canonical_samples = _map_hypercube_samples(
            samples[dim:dim+1], self.domain, self.canonical_domain)
        approx_vals = barycentric_interpolation_1d(
            self._bndry_segment_canonical_abscissa[segment_id][dim],
            self._bndry_segment_canonical_bary_weights[segment_id][dim],
            values, canonical_samples[dim])[:, None]
        return approx_vals


class AbstractSpectralPDE(ABC):
    def __init__(self):
        self._qoi_functional = None
        self._qoi_functional_deriv = None
        self._mesh = None

    def _set_domain(self, domain, order):
        if len(domain) == 2:
            self._mesh = OneDCollocationMesh(domain, order)
        elif len(domain) == 4:
            self._mesh = RectangularCollocationMesh(domain, order)
        raise ValueError("Only 2D domains supported")

    @abstractmethod
    def _set_boundary_conditions(self, bndry_conds):
        raise NotImplementedError()

    def initialize(self, bndry_conds, order, domain):
        self._set_domain(domain, order)
        self._set_boundary_conditions(bndry_conds)

    @abstractmethod
    def _form_collocation_matrix(self, *args):
        raise NotImplementedError

    def set_qoi_functional(self, qoi_functional, qoi_functional_deriv=None):
        self._qoi_functional = qoi_functional
        self._qoi_functional_deriv = qoi_functional_deriv
