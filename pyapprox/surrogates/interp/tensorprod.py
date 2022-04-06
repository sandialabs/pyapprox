import numpy as np
from functools import partial

from pyapprox.util.pya_numba import njit
from pyapprox.util.utilities import cartesian_product, outer_product


def get_tensor_product_quadrature_rule(
        degrees, num_vars, univariate_quadrature_rules, transform_samples=None,
        density_function=None):
    r"""
    if get error about outer product failing it may be because
    univariate_quadrature rule is returning a weights array for every level,
    i.e. l=0,...level
    """
    degrees = np.atleast_1d(degrees)
    if degrees.shape[0] == 1 and num_vars > 1:
        degrees = np.array([degrees[0]]*num_vars, dtype=int)

    if callable(univariate_quadrature_rules):
        univariate_quadrature_rules = [univariate_quadrature_rules]*num_vars

    x_1d = []
    w_1d = []
    for ii in range(len(univariate_quadrature_rules)):
        x, w = univariate_quadrature_rules[ii](degrees[ii])
        x_1d.append(x)
        w_1d.append(w)
    samples = cartesian_product(x_1d, 1)
    weights = outer_product(w_1d)

    if density_function is not None:
        weights *= density_function(samples)
    if transform_samples is not None:
        samples = transform_samples(samples)
    return samples, weights


def piecewise_quadratic_interpolation(samples, mesh, mesh_vals, ranges):
    assert mesh.shape[0] == mesh_vals.shape[0]
    vals = np.zeros_like(samples)
    samples = (samples-ranges[0])/(ranges[1]-ranges[0])
    for ii in range(0, mesh.shape[0]-2, 2):
        xl = mesh[ii]
        xr = mesh[ii+2]
        x = (samples-xl)/(xr-xl)
        interval_vals = canonical_piecewise_quadratic_interpolation(
            x, mesh_vals[ii:ii+3])
        # to avoid double counting we set left boundary of each interval to
        # zero except for first interval
        if ii == 0:
            interval_vals[(x < 0) | (x > 1)] = 0.
        else:
            interval_vals[(x <= 0) | (x > 1)] = 0.
        vals += interval_vals
    return vals


def canonical_piecewise_quadratic_interpolation(x, nodal_vals):
    r"""
    Piecewise quadratic interpolation of nodes at [0,0.5,1]
    Assumes all values are in [0,1].
    """
    assert x.ndim == 1
    assert nodal_vals.shape[0] == 3
    vals = nodal_vals[0]*(1.0-3.0*x+2.0*x**2)+nodal_vals[1]*(4.0*x-4.0*x**2) +\
        nodal_vals[2]*(-x+2.0*x**2)
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
    xx = np.linspace(range_1d[0], range_1d[1], npoints)
    ww = np.ones((npoints))/(npoints-1)*(range_1d[1]-range_1d[0])
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
        return np.array([0.5])
    return np.linspace(0, 1, nsamples_dydactic_grid_1d(level))


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
    nvars = samples.shape[0]
    levels = np.asarray(levels)
    if len(levels) != nvars:
        raise ValueError("levels and samples are inconsistent")

    basis_fun = {"linear": piecewise_linear_basis,
                 "quadratic": piecewise_quadratic_basis}[basis_type]

    active_vars = np.arange(nvars)[levels > 0]
    nactive_vars = active_vars.shape[0]
    nsamples = samples.shape[1]
    N_active = [nsamples_dydactic_grid_1d(ll) for ll in levels[active_vars]]
    N_max = np.max(N_active)
    basis_vals_1d = np.empty((nactive_vars, N_max, nsamples),
                             dtype=np.float64)
    for dd in range(nactive_vars):
        idx = active_vars[dd]
        basis_vals_1d[dd, :N_active[dd], :] = basis_fun(
            levels[idx], samples[idx, :]).T
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
    return basis_vals.dot(fn_vals)


def tensor_product_piecewise_polynomial_interpolation(
        samples, levels, fun, basis_type="linear"):
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
    samples_1d = [dydactic_grid_1d(ll) for ll in levels]
    grid_samples = cartesian_product(samples_1d)
    fn_vals = fun(grid_samples)
    return tensor_product_piecewise_polynomial_interpolation_with_values(
        samples, fn_vals, levels, basis_type)
