import numpy as np
from functools import partial
from abc import ABC, abstractmethod

from pyapprox.util.pya_numba import njit
from pyapprox.util.utilities import (
    cartesian_product, get_tensor_product_quadrature_rule
)
from pyapprox.surrogates.orthopoly.quadrature import (
    clenshaw_curtis_poly_indices_to_quad_rule_indices,
    clenshaw_curtis_rule_growth)
from pyapprox.surrogates.orthopoly.leja_quadrature import (
    get_univariate_leja_quadrature_rule)
from pyapprox.surrogates.interp.barycentric_interpolation import (
    univariate_lagrange_polynomial)


def piecewise_quadratic_interpolation(samples, mesh, mesh_vals, ranges):
    assert mesh.shape[0] == mesh_vals.shape[0]
    vals = 0
    samples = (samples-ranges[0])/(ranges[1]-ranges[0])
    mesh = (mesh-ranges[0])/(ranges[1]-ranges[0])
    assert mesh.shape[0] % 2 == 1
    for ii in range(0, mesh.shape[0]-2, 2):
        xl = mesh[ii]
        xr = mesh[ii+2]
        x = (samples-xl)/(xr-xl)
        interval_vals = canonical_piecewise_quadratic_interpolation(
            x, mesh_vals[ii:ii+3])
        # to avoid double counting we set left boundary of each interval to
        # zero except for first interval
        if ii == 0:
            interval_vals[(x < 0) | (x > 1), :] = 0.
        else:
            interval_vals[(x <= 0) | (x > 1), :] = 0.
        vals += interval_vals
    return vals


def canonical_piecewise_quadratic_interpolation(x, nodal_vals):
    r"""
    Piecewise quadratic interpolation of nodes at [0,0.5,1]
    Assumes all values are in [0,1].
    """
    assert x.ndim == 1
    assert nodal_vals.shape[0] == 3
    if nodal_vals.ndim == 1:
        nodal_vals = nodal_vals[:, None]
    x = x[:, None]
    vals = (nodal_vals[0]*(1.0-3.0*x+2.0*x**2) +
            nodal_vals[1]*(4.0*x-4.0*x**2) +
            nodal_vals[2]*(-x+2.0*x**2))
    return vals


def gradient_of_tensor_product_function(univariate_functions,
                                        univariate_derivatives, samples):
    num_samples = samples.shape[1]
    num_vars = len(univariate_functions)
    assert len(univariate_derivatives) == num_vars
    gradient = np.empty((num_vars, num_samples))
    # precompute data which is reused multiple times
    function_values = []
    for ii in range(num_vars):
        function_values.append(univariate_functions[ii](samples[ii, :]))

    for ii in range(num_vars):
        gradient[ii, :] = univariate_derivatives[ii](samples[ii, :])
        for jj in range(ii):
            gradient[ii, :] *= function_values[jj]
        for jj in range(ii+1, num_vars):
            gradient[ii, :] *= function_values[jj]
    return gradient


def evaluate_tensor_product_function(univariate_functions, samples):
    num_samples = samples.shape[1]
    num_vars = len(univariate_functions)
    values = np.ones((num_samples))
    for ii in range(num_vars):
        values *= univariate_functions[ii](samples[ii, :])
    return values


def piecewise_univariate_linear_quad_rule(range_1d, npoints):
    """
    Compute the points and weights of a piecewise-linear quadrature
    rule that can be used to compute a definite integral

    Parameters
    ----------
    range_1d : iterable (2)
       The lower and upper bound of the definite integral

    Returns
    -------
    xx : np.ndarray (npoints)
        The points of the quadrature rule

    ww : np.ndarray (npoints)
        The weights of the quadrature rule
    """
    if npoints == 1:
        return (np.array([(range_1d[1]+range_1d[0])/2]),
                np.array([float(range_1d[1]-range_1d[0])]))
    xx = np.linspace(range_1d[0], range_1d[1], npoints)
    ww = np.ones((npoints), dtype=float)/(npoints-1)*(range_1d[1]-range_1d[0])
    ww[0] *= 0.5
    ww[-1] *= 0.5
    return xx, ww


def piecewise_univariate_quadratic_quad_rule(range_1d, npoints):
    """
    Compute the points and weights of a piecewise-quadratic quadrature
    rule that can be used to compute a definite integral

    Parameters
    ----------
    range_1d : iterable (2)
       The lower and upper bound of the definite integral

    Returns
    -------
    xx : np.ndarray (npoints)
        The points of the quadrature rule

    ww : np.ndarray (npoints)
        The weights of the quadrature rule
    """
    if npoints == 1:
        return (np.array([(range_1d[1]+range_1d[0])/2]),
                np.array([(range_1d[1]-range_1d[0])]))
    xx = np.linspace(range_1d[0], range_1d[1], npoints)
    dx = 4/(3*(npoints-1))
    ww = dx*np.ones((npoints))*(range_1d[1]-range_1d[0])
    ww[0::2] *= 0.5
    ww[0] *= 0.5
    ww[-1] *= 0.5
    return xx, ww


def get_tensor_product_piecewise_polynomial_quadrature_rule(
        nsamples_1d, ranges, degree=1):
    """
    Compute the nodes and weights needed to integrate a 2D function using
    piecewise linear interpolation
    """
    nrandom_vars = len(ranges)//2
    if isinstance(nsamples_1d, int):
        nsamples_1d = np.array([nsamples_1d]*nrandom_vars)
    assert nrandom_vars == len(nsamples_1d)

    if degree == 1:
        piecewise_univariate_quad_rule = piecewise_univariate_linear_quad_rule
    elif degree == 2:
        piecewise_univariate_quad_rule = \
            piecewise_univariate_quadratic_quad_rule
    else:
        raise ValueError("degree must be 1 or 2")

    univariate_quad_rules = [
        partial(piecewise_univariate_quad_rule, ranges[2*ii:2*ii+2])
        for ii in range(nrandom_vars)]
    x_quad, w_quad = get_tensor_product_quadrature_rule(
        nsamples_1d, nrandom_vars,
        univariate_quad_rules)

    return x_quad, w_quad


@njit(cache=True)
def piecewise_quadratic_basis(level, xx):
    """
    Evaluate each piecewise quadratic basis on a dydatic grid of a specified
    level.

    Parameters
    ----------
    level : integer
        The level of the dydadic grid. The number of points in the grid is
        nbasis=2**level+1, except at level 0 when nbasis=1

    xx : np.ndarary (nsamples)
        The samples at which to evaluate the basis functions

    Returns
    -------
    vals : np.ndarary (nsamples, nbasis)
        Evaluations of each basis function
    """
    assert level > 0
    h = 1/float(1 << level)
    N = (1 << level)+1
    vals = np.zeros((xx.shape[0], N))
    for ii in range(N):
        xl = (ii-1.0)*h
        xr = xl+2.0*h
        if ii % 2 == 1:
            vals[:, ii] = np.maximum(-(xx-xl)*(xx-xr)/(h*h), 0.0)
            continue
        II = np.where((xx > xl-h) & (xx < xl+h))[0]
        xx_II = xx[II]
        vals[II, ii] = (xx_II**2-h*xx_II*(2*ii-3)+h*h*(ii-1)*(ii-2))/(2.*h*h)
        JJ = np.where((xx >= xl+h) & (xx < xr+h))[0]
        xx_JJ = xx[JJ]
        vals[JJ, ii] = (xx_JJ**2-h*xx_JJ*(2*ii+3)+h*h*(ii+1)*(ii+2))/(2.*h*h)
    return vals


@njit(cache=True)
def irregular_piecewise_left_constant_basis(nodes, xx):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    vals = np.zeros((xx.shape[0], nnodes-1))
    for ii in range(nnodes-1):
        xl = nodes[ii]
        xr = nodes[ii+1]
        II = np.where((xx >= xl) & (xx < xr))[0]
        vals[II, ii] = np.ones(II.shape[0], dtype=float)
    return vals


@njit(cache=True)
def irregular_piecewise_right_constant_basis(nodes, xx):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    vals = np.zeros((xx.shape[0], nnodes-1))
    for ii in range(1, nnodes):
        xr = nodes[ii]
        xl = nodes[ii-1]
        II = np.where((xx > xl) & (xx <= xr))[0]
        vals[II, ii] = np.ones(II.shape[0], dtype=float)
    return vals


@njit(cache=True)
def irregular_piecewise_midpoint_constant_basis(nodes, xx):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    vals = np.zeros((xx.shape[0], nnodes-1))
    for ii in range(nnodes-1):
        xl = nodes[ii]
        xr = nodes[ii+1]
        if ii < nnodes-1:
            II = np.where((xx >= xl) & (xx < xr))[0]
        else:
            II = np.where((xx >= xl) & (xx <= xr))[0]
        vals[II, ii] = np.ones(II.shape[0], dtype=float)
    return vals


@njit(cache=True)
def irregular_piecewise_linear_basis(nodes, xx):
    # abscissa are not equidistant
    assert xx.ndim == 1
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    vals = np.zeros((xx.shape[0], nnodes))
    for ii in range(nnodes):
        xm = nodes[ii]
        if ii > 0:
            xl = nodes[ii-1]
            II = np.where((xx >= xl) & (xx <= xm))[0]
            vals[II, ii] = (xx[II]-xl)/(xm-xl)
        if ii < nnodes-1:
            xr = nodes[ii+1]
            JJ = np.where((xx >= xm) & (xx <= xr))[0]
            vals[JJ, ii] = (xr-xx[JJ])/(xr-xm)
    return vals


def irregular_piecewise_linear_quadrature_weights(nodes):
    assert nodes.ndim == 1
    nnodes = nodes.shape[0]
    weights = np.zeros((nnodes,))
    for ii in range(nnodes):
        xm = nodes[ii]
        if ii > 0:
            xl = nodes[ii-1]
            weights[ii] += 0.5*(xm-xl)
        if ii < nnodes-1:
            xr = nodes[ii+1]
            weights[ii] += 0.5*(xr-xm)
    return weights


@njit(cache=True)
def irregular_piecewise_quadratic_basis(nodes, xx):
    # nodes are not equidistant
    assert xx.ndim == 1
    nnodes = nodes.shape[0]
    if nodes.ndim != 1 or nnodes % 2 != 1:
        raise ValueError("nodes has the wrong shape")
    vals = np.zeros((xx.shape[0], nnodes))
    for ii in range(nnodes):
        if ii % 2 == 1:
            xl, xm, xr = nodes[ii-1:ii+2]
            II = np.where((xx >= xl) & (xx <= xr))[0]
            vals[II, ii] = (xx[II]-xl)/(xm-xl)*(xx[II]-xr)/(xm-xr)
            continue
        if ii < nnodes-2:
            xl, xm, xr = nodes[ii:ii+3]
            II = np.where((xx >= xl) & (xx <= xr))[0]
            vals[II, ii] = (xx[II]-xm)/(xl-xm)*(xx[II]-xr)/(xl-xr)
        if ii > 1:
            xl, xm, xr = nodes[ii-2:ii+1]
            II = np.where((xx >= xl) & (xx <= xr))[0]
            vals[II, ii] = (xx[II]-xl)/(xr-xl)*(xx[II]-xm)/(xr-xm)
    return vals


@njit(cache=True)
def irregular_piecewise_quadratic_quadrature_weights(nodes):
    # nodes are not equidistant
    nnodes = nodes.shape[0]
    if nodes.ndim != 1 or nnodes % 2 != 1:
        raise ValueError("nodes has the wrong shape")
    weights = np.zeros((nnodes,))
    for ii in range(nnodes):
        if ii % 2 == 1:
            xl, xm, xr = nodes[ii-1:ii+2]
            weights[ii] = (xl-xr)**3/(6*(xm-xl)*(xm-xr))
            continue
        if ii < nnodes-2:
            xl, xm, xr = nodes[ii:ii+3]
            weights[ii] += (((xr - xl)*(2*xl - 3*xm + xr))/(6*(xl - xm)))
        if ii > 1:
            xl, xm, xr = nodes[ii-2:ii+1]
            weights[ii] += ((xl - xr)*(xl - 3*xm + 2*xr))/(6*(xm - xr))
    return weights


@njit(cache=True)
def irregular_piecewise_cubic_basis(nodes, xx):
    # nodes are not equidistant
    assert xx.ndim == 1
    nnodes = nodes.shape[0]
    if (nodes.ndim != 1 or nnodes < 4 or (nnodes-4) % 3 != 0):
        raise ValueError("nodes has the wrong shape")
    vals = np.zeros((xx.shape[0], nnodes))
    for ii in range(nnodes):
        if ii % 3 == 1:
            x1, x2, x3, x4 = nodes[ii-1:ii+3]
            II = np.where((xx >= x1) & (xx <= x4))[0]
            vals[II, ii] = ((xx[II]-x1)/(x2-x1)*(xx[II]-x3)/(x2-x3) *
                            (xx[II]-x4)/(x2-x4))
            continue
        if ii % 3 == 2:
            x1, x2, x3, x4 = nodes[ii-2:ii+2]
            II = np.where((xx >= x1) & (xx <= x4))[0]
            vals[II, ii] = ((xx[II]-x1)/(x3-x1)*(xx[II]-x2)/(x3-x2) *
                            (xx[II]-x4)/(x3-x4))
            continue
        if ii % 3 == 0 and ii < nnodes-3:
            x1, x2, x3, x4 = nodes[ii:ii+4]
            II = np.where((xx >= x1) & (xx <= x4))[0]
            vals[II, ii] = ((xx[II]-x2)/(x1-x2)*(xx[II]-x3)/(x1-x3) *
                            (xx[II]-x4)/(x1-x4))
        if ii % 3 == 0 and ii >= 3:
            x1, x2, x3, x4 = nodes[ii-3:ii+1]
            II = np.where((xx >= x1) & (xx <= x4))[0]
            vals[II, ii] = ((xx[II]-x1)/(x4-x1)*(xx[II]-x2)/(x4-x2) *
                            (xx[II]-x3)/(x4-x3))
    return vals


def irregular_piecewise_cubic_quadrature_weights(nodes):
    # An interpolating quadrature with n + 1 nodes that are symmetrically
    # placed around the center of the interval will integrate polynomials up to
    # degree n exactly when n is odd, and up to degree n + 1 exactly when
    # n is even.
    nnodes = nodes.shape[0]
    if (nodes.ndim != 1 or nnodes < 4 or (nnodes-4) % 3 != 0):
        raise ValueError(f"nodes has the wrong shape {nnodes}")
    weights = np.zeros((nnodes,))
    for ii in range(nnodes):
        if ii % 3 == 1:
            a, b, c, d = nodes[ii-1:ii+3]
            weights[ii] = ((a-d)**3*(a-2*c+d))/(12*(-a+b)*(b-c)*(b-d))
            continue
        if ii % 3 == 2:
            a, b, c, d = nodes[ii-2:ii+2]
            weights[ii] = ((a-d)**3*(a-2*b+d))/(12*(-a+c)*(-b+c)*(c - d))
            continue
        if ii % 3 == 0 and ii < nnodes-3:
            a, b, c, d = nodes[ii:ii+4]
            weights[ii] += (((d-a)*(3*a**2+6*b*c-2*(b+c)*d+d**2 +
                                    2*a*(-2*(b+c)+d)))/(12*(a-b)*(a-c)))
        if ii % 3 == 0 and ii >= 3:
            a, b, c, d = nodes[ii-3:ii+1]
            weights[ii] += ((a-d)*(a**2+6*b*c-2*a*(b+c-d)-4*b*d-4*c*d +
                                   3*d**2))/(12*(b-d)*(-c+d))
    return weights


class UnivariateInterpolatingBasis(ABC):
    @abstractmethod
    def __call__(self, nodes, samples):
        raise NotImplementedError

    def quadrature_weights(self, nodes):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)

    def integrate(self, nodes, vals):
        weights = self.quadrature_weights(nodes)
        return (weights[:, None]*vals).sum()


class UnivariatePiecewiseLeftConstantBasis(UnivariateInterpolatingBasis):
    @staticmethod
    def __call__(nodes, samples):
        return irregular_piecewise_left_constant_basis(nodes, samples)

    @staticmethod
    def quadrature_weights(nodes):
        # unlike other higherorder methods weights.shape[0] != nodes.shape[0]
        return np.diff(nodes)

    def integrate(self, nodes, vals):
        weights = self.quadrature_weights(nodes)
        return (weights[:, None]*vals[:-1]).sum()


class UnivariatePiecewiseRightConstantBasis(UnivariateInterpolatingBasis):
    @staticmethod
    def __call__(nodes, samples):
        return irregular_piecewise_right_constant_basis(nodes, samples)

    @staticmethod
    def quadrature_weights(nodes):
        # unlike other higherorder methods weights.shape[0] != nodes.shape[0]
        return np.diff(nodes)

    def integrate(self, nodes, vals):
        weights = self.quadrature_weights(nodes)
        return (weights[:, None]*vals[1:]).sum()


class UnivariatePiecewiseMidPointConstantBasis(UnivariateInterpolatingBasis):
    @staticmethod
    def __call__(nodes, samples):
        return irregular_piecewise_midpoint_constant_basis(nodes, samples)

    @staticmethod
    def quadrature_weights(nodes):
        # unlike other higherorder methods weights.shape[0] != nodes.shape[0]
        return np.diff(nodes)


class UnivariatePiecewiseLinearBasis(UnivariateInterpolatingBasis):
    @staticmethod
    def __call__(nodes, samples):
        return irregular_piecewise_linear_basis(nodes, samples)

    @staticmethod
    def quadrature_weights(nodes):
        return irregular_piecewise_linear_quadrature_weights(nodes)


class UnivariatePiecewiseQuadraticBasis(UnivariateInterpolatingBasis):
    @staticmethod
    def __call__(nodes, samples):
        return irregular_piecewise_quadratic_basis(nodes, samples)

    @staticmethod
    def quadrature_weights(nodes):
        return irregular_piecewise_quadratic_quadrature_weights(nodes)


class UnivariatePiecewiseCubicBasis(UnivariateInterpolatingBasis):
    @staticmethod
    def __call__(nodes, samples):
        return irregular_piecewise_cubic_basis(nodes, samples)

    @staticmethod
    def quadrature_weights(nodes):
        return irregular_piecewise_cubic_quadrature_weights(nodes)


class UnivariateLagrangeBasis(UnivariateInterpolatingBasis):
    @staticmethod
    def __call__(nodes, samples):
        return univariate_lagrange_polynomial(nodes, samples)


def get_univariate_interpolation_basis(basis_type):
    basis_dict = {"linear": UnivariatePiecewiseLinearBasis,
                  "quadratic": UnivariatePiecewiseQuadraticBasis,
                  "cubic": UnivariatePiecewiseCubicBasis,
                  "lagrange": UnivariateLagrangeBasis}
    if basis_type not in basis_dict:
        raise ValueError("basis_type {0} not supported must be in {1}".format(
            basis_type, list(basis_dict.keys())))
    return basis_dict[basis_type]()


class TensorProductBasis():
    def __init__(self, bases_1d):
        self._bases_1d = bases_1d

    def __call__(self, nodes_1d, samples):
        # assumes each array in nodes_1d is in ascending order
        nvars = len(nodes_1d)
        nnodes_1d = np.array([n.shape[0] for n in nodes_1d])
        active_vars = np.arange(nvars)[nnodes_1d > 1]
        nactive_vars = active_vars.shape[0]
        if nactive_vars == 0:
            return np.ones((samples.shape[1], 1))
        nsamples = samples.shape[1]
        nnodes_1d_active = nnodes_1d[active_vars]
        nnodes_1d_max = np.max(nnodes_1d_active)
        basis_vals_1d = np.empty(
            (nactive_vars, nnodes_1d_max, nsamples), dtype=np.float64)
        for dd in range(nactive_vars):
            idx = active_vars[dd]
            basis_vals_1d[dd, :nnodes_1d_active[dd], :] = self._bases_1d[idx](
                nodes_1d[idx], samples[idx, :]).T

        temp1 = basis_vals_1d.reshape(
            (nactive_vars*basis_vals_1d.shape[1], nsamples))
        indices = cartesian_product([np.arange(N) for N in nnodes_1d_active])
        nindices = indices.shape[1]
        temp2 = temp1[indices.ravel()+np.repeat(
            np.arange(nactive_vars)*basis_vals_1d.shape[1],
            nindices), :].reshape(nactive_vars, nindices, nsamples)
        basis_vals = np.prod(temp2, axis=0).T
        return basis_vals

    def _single_basis_fun(self, nodes_1d, idx, xx):
        return self(nodes_1d, xx)[:, idx]

    def plot_single_basis(self, ax, nodes_1d, ii, jj, nodes=None,
                          plot_limits=[-1, 1, -1, 1],
                          num_pts_1d=101, surface_cmap="coolwarm",
                          contour_cmap="gray"):
        from pyapprox.util.visualization import (
            get_meshgrid_function_data, plot_surface)
        idx = jj*nodes_1d[0].shape[0]+ii
        single_basis_fun = partial(self._single_basis_fun, nodes_1d, idx)
        X, Y, Z = get_meshgrid_function_data(
            single_basis_fun, plot_limits, num_pts_1d)
        if surface_cmap is not None:
            plot_surface(X, Y, Z, ax, axis_labels=None, limit_state=None,
                         alpha=0.3, cmap=surface_cmap, zorder=3, plot_axes=False)
        if contour_cmap is not None:
            num_contour_levels = 30
            offset = -(Z.max()-Z.min())/2
            ax.contourf(
                X, Y, Z, zdir='z', offset=offset,
                levels=np.linspace(Z.min(), Z.max(), num_contour_levels),
                cmap=contour_cmap, zorder=-1)

        if nodes is None:
            return
        ax.plot(nodes[0, :], nodes[1, :],
                offset*np.ones(nodes.shape[1]), 'o',
                zorder=100, color='b')

        x = np.linspace(-1, 1, 100)
        y = nodes[1, idx]*np.ones((x.shape[0]))
        z = single_basis_fun(np.vstack((x[None, :], y[None, :])))
        ax.plot(x, Y.max()*np.ones((x.shape[0])), z, '-r')
        ax.plot(nodes_1d[0], Y.max()*np.ones(
            (nodes_1d[0].shape[0])), np.zeros(nodes_1d[0].shape[0]), 'or')

        y = np.linspace(-1, 1, 100)
        x = nodes[0, idx]*np.ones((y.shape[0]))
        z = single_basis_fun(np.vstack((x[None, :], y[None, :])))
        ax.plot(X.min()*np.ones((x.shape[0])), y, z, '-r')
        ax.plot(X.min()*np.ones(
            (nodes_1d[1].shape[0])), nodes_1d[1],
                np.zeros(nodes_1d[1].shape[0]), 'or')


class TensorProductInterpolant():
    def __init__(self, bases_1d):
        self.basis = TensorProductBasis(bases_1d)

        self._nodes_1d = None
        self._nnodes_1d = None
        self._values = None

    def tensor_product_grid(self, nodes_1d):
        return cartesian_product(nodes_1d)

    def fit(self, nodes_1d, values):
        self._nodes_1d = nodes_1d
        self._nnodes = np.prod([n.shape[0] for n in nodes_1d])
        if values.shape[0] != self._nnodes:
            raise ValueError("nodes_1d and values are inconsistent")
        if values.ndim == 1:
            values = values[:, None]
        self._values = values

    def __call__(self, samples):
        basis_mat = self.basis(self._nodes_1d, samples)
        return basis_mat @ self._values

    def __repr__(self):
        return "{0}(bases={1})".format(
            self.__class__.__name__,
            "["+", ".join(map("{}".format, self._bases_1d)) + "]")


def piecewise_linear_basis(level, xx):
    """
    Evaluate each piecewise linear basis on a dydatic grid of a specified
    level.

    Parameters
    ----------
    level : integer
        The level of the dydadic grid. The number of points in the grid is
        nbasis=2**level+1, except at level 0 when nbasis=1

    xx : np.ndarary (nsamples)
        The samples at which to evaluate the basis functions

    Returns
    -------
    vals : np.ndarary (nsamples, nbasis)
        Evaluations of each basis function
    """
    assert level > 0
    N = (1 << level)+1
    vals = np.maximum(
        0, 1-np.absolute((1 << level)*xx[:, None]-np.arange(N)[None, :]))
    return vals


def nsamples_dydactic_grid_1d(level):
    """
    The number of points in a dydactic grid.

    Parameters
    ----------
    level : integer
        The level of the dydadic grid.

    Returns
    -------
    nsamples : integer
        The number of points in the grid
    """
    if level == 0:
        return 1
    return (1 << level)+1


def dydactic_grid_1d(level):
    """
    The points in a dydactic grid.

    Parameters
    ----------
    level : integer
        The level of the dydadic grid.

    Returns
    -------
    samples : np.ndarray(nbasis)
        The points in the grid
    """
    if level == 0:
        return np.array([0.0])
    return np.linspace(-1, 1, nsamples_dydactic_grid_1d(level))


def tensor_product_piecewise_polynomial_basis(
        levels, samples, basis_type="linear"):
    """
    Evaluate each piecewise polynomial basis on a tensor product dydactic grid

    Parameters
    ----------
    levels : array_like (nvars)
        The levels of each 1D dydadic grid.

    samples : np.ndarray (nvars, nsamples)
        The samples at which to evaluate the basis functions

    basis_type : string
        The type of piecewise polynomial basis, i.e. 'linear' or 'quadratic'

    Returns
    -------
    basis_vals : np.ndarray(nsamples, nbasis)
        Evaluations of each basis function
    """
    assert samples.min() >= -1 and samples.max() <= 1
    samples = (samples+1)/2
    assert samples.min() >= 0 and samples.max() <= 1
    nvars = samples.shape[0]
    levels = np.asarray(levels)
    if len(levels) != nvars:
        msg = "levels {0} and samples {1} are inconsistent".format(
            levels, nvars)
        raise ValueError(msg)

    basis_fun = {"linear": piecewise_linear_basis,
                 "quadratic": piecewise_quadratic_basis}[basis_type]

    active_vars = np.arange(nvars)[levels > 0]
    nactive_vars = active_vars.shape[0]
    if nactive_vars == 0:
        return np.ones((samples.shape[1], 1))
    nsamples = samples.shape[1]
    N_active = [nsamples_dydactic_grid_1d(ll) for ll in levels[active_vars]]
    N_max = np.max(N_active)
    basis_vals_1d = np.empty((nactive_vars, N_max, nsamples),
                             dtype=np.float64)
    for dd in range(nactive_vars):
        idx = active_vars[dd]
        basis_vals_1d[dd, :N_active[dd], :] = basis_fun(
            levels[idx], samples[idx, :]).T
        # assumes that basis functions are nested, i.e.
        # basis for x=0.5, 0, 1, 0.25, 0.75 and so on
        indices = clenshaw_curtis_poly_indices_to_quad_rule_indices(
            levels[idx])
        basis_vals_1d[dd, :N_active[dd], :] = (
            basis_vals_1d[dd, :N_active[dd], :][indices, :])

    temp1 = basis_vals_1d.reshape(
        (nactive_vars*basis_vals_1d.shape[1], nsamples))
    indices = cartesian_product([np.arange(N) for N in N_active])
    nindices = indices.shape[1]
    temp2 = temp1[indices.ravel()+np.repeat(
        np.arange(nactive_vars)*basis_vals_1d.shape[1], nindices), :].reshape(
            nactive_vars, nindices, nsamples)
    basis_vals = np.prod(temp2, axis=0).T
    return basis_vals


def tensor_product_piecewise_polynomial_interpolation_with_values(
        samples, fn_vals, levels, basis_type="linear"):
    """
    Use a piecewise polynomial basis to interpolate a function from values
    defined on a tensor product dydactic grid.

    Parameters
    ----------
    samples : np.ndarray (nvars, nsamples)
        The samples at which to evaluate the basis functions

    fn_vals : np.ndarary (nbasis, nqoi)
        The values of the function on the dydactic grid

    levels : array_like (nvars)
        The levels of each 1D dydadic grid.

    basis_type : string
        The type of piecewise polynomial basis, i.e. 'linear' or 'quadratic'

    Returns
    -------
    basis_vals : np.ndarray(nsamples, nqoi)
        Evaluations of the interpolant at the samples
    """
    basis_vals = tensor_product_piecewise_polynomial_basis(
        levels, samples, basis_type)
    if fn_vals.shape[0] != basis_vals.shape[1]:
        raise ValueError("The size of fn_vals is inconsistent with levels")
    # from matplotlib import pyplot as plt
    # print(fn_vals)
    # II = np.argsort(samples[0])
    # plt.plot(samples[0, II], basis_vals[II])
    # plt.plot(samples[0, II], basis_vals[II].dot(fn_vals))
    # plt.show()
    return basis_vals.dot(fn_vals)


def tensor_product_piecewise_polynomial_interpolation(
        samples, levels, fun, basis_type="linear", var_trans=None,
        return_all=False):
    """
    Use tensor-product piecewise polynomial basis to interpolate a function.

    Parameters
    ----------
    samples : np.ndarray (nvars, nsamples)
        The samples at which to evaluate the basis functions

    levels : array_like (nvars)
        The levels of each 1D dydadic grid.

    fun : callable
        Function with the signature

        `fun(samples) -> np.ndarray (nx, nqoi)`

        where samples is np.ndarray (nvars, nx)

    basis_type : string
        The type of piecewise polynomial basis, i.e. 'linear' or 'quadratic'

    Returns
    -------
    basis_vals : np.ndarray(nsamples, nqoi)
        Evaluations of the interpolant at the samples
    """
    samples_1d = [dydactic_grid_1d(ll)[
        clenshaw_curtis_poly_indices_to_quad_rule_indices(ll)]
                  for ll in levels]
    grid_samples = cartesian_product(samples_1d)
    if var_trans is not None:
        grid_samples = var_trans.map_from_canonical(grid_samples)
        samples = var_trans.map_to_canonical(samples)
    fn_vals = fun(grid_samples)
    approx_vals = (
        tensor_product_piecewise_polynomial_interpolation_with_values(
            samples, fn_vals, levels, basis_type))
    if not return_all:
        return approx_vals
    return approx_vals, grid_samples, fn_vals


def canonical_univariate_piecewise_polynomial_quad_rule(
        basis_type,  level, return_weights_for_all_levels=True):
    ordered_weights_1d = []
    for ll in range(level+1):
        npts = clenshaw_curtis_rule_growth(ll)
        if basis_type == "linear":
            x, w = piecewise_univariate_quadratic_quad_rule([-1, 1], npts)
        elif basis_type == "quadratic":
            x, w = piecewise_univariate_quadratic_quad_rule([-1, 1], npts)
        else:
            raise NotImplementedError(f"{basis_type} not supported")
        # piecewise rules are for lebesque integation
        # so must divide w/2 so that integation is respcet to
        # uniform measure
        w = w / 2
        quad_indices = clenshaw_curtis_poly_indices_to_quad_rule_indices(
            ll)
        ordered_weights_1d.append(w[quad_indices])
        # ordered samples for last x

    ordered_samples_1d = x[quad_indices]
    if return_weights_for_all_levels:
        return ordered_samples_1d, ordered_weights_1d

    return ordered_samples_1d, w[quad_indices]


def get_univariate_leja_quadrature_rules_from_variable(
        variable, growth_rules, levels, canonical=False, **kwargs):
    levels = np.atleast_1d(levels)
    if levels.shape[0] == 1:
        levels = np.ones(variable.num_vars(), dtype=int)*levels[0]
    if levels.shape[0] != variable.num_vars():
        raise ValueError(
            "levels must be an integer or specfied for each marginal")
    univariate_quad_rules = []
    for ii, marginal in enumerate(variable.marginals()):
        quad_rule = get_univariate_leja_quadrature_rule(
            marginal, growth_rules[ii], **kwargs)
        univariate_quad_rules.append(quad_rule)
    return univariate_quad_rules
