import numpy as np
from functools import partial

from pyapprox.util.pya_numba import njit
from pyapprox.util.utilities import (
    cartesian_product, get_tensor_product_quadrature_rule
)
from pyapprox.surrogates.orthopoly.quadrature import (
    clenshaw_curtis_poly_indices_to_quad_rule_indices,
    clenshaw_curtis_rule_growth)
from pyapprox.surrogates.orthopoly.leja_quadrature import (
    get_univariate_leja_quadrature_rule)


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
