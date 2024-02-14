import warnings
import numpy as np
from scipy.special import comb as nchoosek

from pyapprox.util.utilities import cartesian_product
from pyapprox.util.pya_numba import njit
from pyapprox.util.sys_utilities import trace_error_with_msg
from pyapprox.util.visualization import (
    get_meshgrid_function_data, create_3d_axis, mpl, plot_surface)


def compute_barycentric_weights_1d(samples, interval_length=None,
                                   return_sequence=False,
                                   normalize_weights=False):
    """
    Return barycentric weights for a sequence of samples. e.g. of sequence
    x0,x1,x2 where order represents the order in which the samples are added
    to the interpolant.

    Parameters
    ----------
    return_sequence : boolean
        True - return [1],[1/(x0-x1),1/(x1-x0)],
                      [1/((x0-x2)(x0-x1)),1/((x1-x2)(x1-x0)),1/((x2-x1)(x2-x0))]
        False- return [1/((x0-x2)(x0-x1)),1/((x1-x2)(x1-x0)),1/((x2-x1)(x2-x0))]

    Note
    ----
    If length of interval [a,b]=4C then weights will grow or decay
    exponentially at C^{-n} where n is number of points causing overflow
    or underflow.

    To minimize this effect multiply each x_j-x_k by C^{-1}. This has effect
    of rescaling all weights by C^n. In rare situations where n is so large
    randomize or use Leja ordering of the samples before computing weights.
    See Barycentric Lagrange Interpolation by
    Jean-Paul Berrut and Lloyd N. Trefethen 2004
    """
    if interval_length is None:
        scaling_factor = 1.
    else:
        scaling_factor = interval_length/4.

    C_inv = 1/scaling_factor
    num_samples = samples.shape[0]

    try:
        from pyapprox.cython.barycentric_interpolation import \
            compute_barycentric_weights_1d_pyx
        weights = compute_barycentric_weights_1d_pyx(samples, C_inv)
    except (ImportError, ModuleNotFoundError) as e:
        msg = 'compute_barycentric_weights_1d extension failed'
        trace_error_with_msg(msg, e)

    weights = np.empty((num_samples, num_samples), dtype=float)
    weights[0, 0] = 1.
    for jj in range(1, num_samples):
        weights[jj, :jj] = C_inv * \
            (samples[:jj]-samples[jj])*weights[jj-1, :jj]
        weights[jj, jj] = np.prod(C_inv*(samples[jj]-samples[:jj]))
        weights[jj-1, :jj] = 1./weights[jj-1, :jj]

    weights[num_samples-1, :num_samples] =\
        1./weights[num_samples-1, :num_samples]

    if not return_sequence:
        result = weights[num_samples-1, :]
        # make sure magintude of weights is approximately O(1)
        # useful to sample sets like leja for gaussian variables
        # where interval [a,b] is not very useful
        # print('max_weights',result.min(),result.max())
        if normalize_weights:
            msg = 'I do not think I want to support this option'
            raise NotImplementedError(msg)
            result /= np.absolute(result).max()
            # result[I]=result

    else:
        result = weights

    assert np.all(np.isfinite(result)), (num_samples)
    return result


def _barycentric_interpolation_1d(abscissa, weights, vals, eval_samples):
    # mask = np.in1d(eval_samples, abscissa)
    # II = np.where(~mask)[0]
    # All eval_samples not corresponding to abscissa
    # using mask will avoid divide by zero but is much slower
    approx_vals = np.empty((eval_samples.shape[0]))
    diff = eval_samples[:, None]-abscissa[None, :]
    approx_vals = (
        weights*vals/diff).sum(axis=1)/(weights/diff).sum(axis=1)
    # set approx to vals at abscissa
    II = np.where(eval_samples == abscissa[:, None])
    approx_vals[II[1]] = vals[II[0]]
    return approx_vals


def barycentric_interpolation_1d(abscissa, weights, vals, eval_samples):
    with warnings.catch_warnings():
        # avoid division by zero warning
        warnings.simplefilter("ignore")
        return _barycentric_interpolation_1d(abscissa, weights, vals, eval_samples)


def barycentric_lagrange_interpolation_precompute(
        num_act_dims, abscissa_1d, barycentric_weights_1d,
        active_abscissa_indices_1d_list):
    num_abscissa_1d = np.empty((num_act_dims), dtype=np.int32)
    num_active_abscissa_1d = np.empty((num_act_dims), dtype=np.int32)
    shifts = np.empty((num_act_dims), dtype=np.int32)

    shifts[0] = 1
    num_abscissa_1d[0] = abscissa_1d[0].shape[0]
    num_active_abscissa_1d[0] = active_abscissa_indices_1d_list[0].shape[0]
    max_num_abscissa_1d = num_abscissa_1d[0]
    for act_dim_idx in range(1, num_act_dims):
        num_abscissa_1d[act_dim_idx] = abscissa_1d[act_dim_idx].shape[0]
        num_active_abscissa_1d[act_dim_idx] = \
            active_abscissa_indices_1d_list[act_dim_idx].shape[0]
        # multi-index needs only be defined over active_abscissa_1d
        shifts[act_dim_idx] = \
            shifts[act_dim_idx-1]*num_active_abscissa_1d[act_dim_idx-1]
        max_num_abscissa_1d = max(
            max_num_abscissa_1d, num_abscissa_1d[act_dim_idx])

    max_num_active_abscissa_1d = num_active_abscissa_1d.max()
    active_abscissa_indices_1d = np.empty(
        (num_act_dims, max_num_active_abscissa_1d), dtype=np.int32)
    for dd in range(num_act_dims):
        active_abscissa_indices_1d[dd, :num_active_abscissa_1d[dd]] = \
            active_abscissa_indices_1d_list[dd]

    # Create locality of data for increased preformance
    abscissa_and_weights = np.empty(
        (2*max_num_abscissa_1d, num_act_dims), dtype=np.float64)
    for dd in range(num_act_dims):
        for ii in range(num_abscissa_1d[dd]):
            abscissa_and_weights[2*ii, dd] = abscissa_1d[dd][ii]
            abscissa_and_weights[2*ii+1, dd] = barycentric_weights_1d[dd][ii]

    return (num_abscissa_1d, num_active_abscissa_1d, shifts,
            abscissa_and_weights, active_abscissa_indices_1d)


def multivariate_hierarchical_barycentric_lagrange_interpolation(
        x,
        abscissa_1d,
        barycentric_weights_1d,
        fn_vals,
        active_dims,
        active_abscissa_indices_1d):
    """
    Parameters
    ----------
    x : np.ndarray (num_vars, num_samples)
        The samples at which to evaluate the interpolant

    abscissa_1d : [np.ndarray]
        List of interpolation nodes in each active dimension. Each array
        has ndim==1

    barycentric_weights_1d : [np.ndarray]
        List of barycentric weights in each active dimension, corresponding to
        each of the interpolation nodes. Each array has ndim==1

    fn_vals : np.ndarray (num_samples, num_qoi)
        The function values at each of the interpolation nodes
        Each column is a flattened array that assumes the nodes
        were created with the same ordering as generated by
        the function cartesian_product.

        if active_abscissa_1d is not None the fn_vals must be same size as
        the tensor product of the active_abscissa_1d.

        Warning: Python code takes fn_vals as num_samples x num_qoi
        but c++ code takes num_qoi x num_samples. Todo change c++ code
        also look at c++ code to compute barycentric weights. min() on line 154
        seems to have no effect.

    active_dims : np.ndarray (num_active_dims)
        The dimensions which have more than one interpolation node. TODO
        check if this can be simply extracted in this function by looking
        at abscissa_1d.

    active_abscissa_indices_1d : [np.ndarray]
        The list (over each dimension) of indices for which we will compute
        barycentric basis functions. This is useful when used with
        heirarchical interpolation where the function values will be zero
        at some nodes and thus there is no need to compute associated basis
        functions

    Returns
    -------
    result : np.ndarray (num_samples,num_qoi)
        The values of the interpolant at the samples x
    """
    num_act_dims = active_dims.shape[0]
    (num_abscissa_1d, num_active_abscissa_1d, shifts, abscissa_and_weights,
     active_abscissa_indices_1d) = \
         barycentric_lagrange_interpolation_precompute(
             num_act_dims, abscissa_1d, barycentric_weights_1d,
             active_abscissa_indices_1d)

    if (np.prod(num_active_abscissa_1d) != fn_vals.shape[0]):
        print(np.prod(num_active_abscissa_1d), fn_vals.shape,
              num_active_abscissa_1d)
        msg = "The shapes of fn_vals and abscissa_1d are inconsistent"
        raise ValueError(msg)

    try:
        from pyapprox.cython.barycentric_interpolation import \
            multivariate_hierarchical_barycentric_lagrange_interpolation_pyx

        result = \
            multivariate_hierarchical_barycentric_lagrange_interpolation_pyx(
                x, fn_vals, active_dims,
                active_abscissa_indices_1d.astype(np.int_),
                num_abscissa_1d.astype(np.int_),
                num_active_abscissa_1d.astype(np.int_),
                shifts.astype(np.int_), abscissa_and_weights)
        if np.any(np.isnan(result)):
            raise ValueError('Error values not finite')

    except (ImportError, ModuleNotFoundError) as e:
        msg = "multivariate_hierarchical_barycentric_lagrange_interpolation"
        msg += " extension failed"
        trace_error_with_msg(msg, e)

        result = \
            __multivariate_hierarchical_barycentric_lagrange_interpolation(
                x, abscissa_1d, fn_vals, active_dims,
                active_abscissa_indices_1d, num_abscissa_1d,
                num_active_abscissa_1d, shifts, abscissa_and_weights)

    return result


@njit(cache=True)
def __multivariate_hierarchical_barycentric_lagrange_interpolation(
        x, abscissa_1d, fn_vals, active_dims, active_abscissa_indices_1d,
        num_abscissa_1d, num_active_abscissa_1d, shifts,
        abscissa_and_weights):

    eps = 2*np.finfo(np.double).eps
    num_pts = x.shape[1]
    num_act_dims = active_dims.shape[0]

    max_num_abscissa_1d = abscissa_and_weights.shape[0]//2
    multi_index = np.empty((num_act_dims), dtype=np.int64)

    num_qoi = fn_vals.shape[1]
    result = np.empty((num_pts, num_qoi), dtype=np.double)
    # Allocate persistent memory. Each point will fill in a varying amount
    # of entries. We use a view of this memory to stop reallocation for each
    # data point
    act_dims_pt_persistent = np.empty((num_act_dims), dtype=np.int64)
    act_dim_indices_pt_persistent = np.empty((num_act_dims), dtype=np.int64)
    c_persistent = np.empty((num_qoi, num_act_dims), dtype=np.double)
    bases = np.empty((max_num_abscissa_1d, num_act_dims), dtype=np.double)
    for kk in range(num_pts):
        # compute the active dimension of the kth point in x and the
        # set multi_index accordingly
        multi_index[:] = 0
        num_act_dims_pt = 0
        has_inactive_abscissa = False
        for act_dim_idx in range(num_act_dims):
            cnt = 0
            is_active_dim = True
            dim = active_dims[act_dim_idx]
            num_abscissa = num_abscissa_1d[act_dim_idx]
            x_dim_k = x[dim, kk]
            for ii in range(num_abscissa):
                if ((cnt < num_active_abscissa_1d[act_dim_idx]) and
                        (ii == active_abscissa_indices_1d[act_dim_idx][cnt])):
                    cnt += 1
                if (abs(x_dim_k - abscissa_1d[act_dim_idx][ii]) < eps):
                    is_active_dim = False
                    if ((cnt > 0) and
                            (active_abscissa_indices_1d[act_dim_idx][cnt-1] == ii)):
                        multi_index[act_dim_idx] = cnt-1
                    else:
                        has_inactive_abscissa = True
                    break

            if (is_active_dim):
                act_dims_pt_persistent[num_act_dims_pt] = dim
                act_dim_indices_pt_persistent[num_act_dims_pt] = act_dim_idx
                num_act_dims_pt += 1
        # end for act_dim_idx in range(num_act_dims):

        if (has_inactive_abscissa):
            result[kk, :] = 0.
        else:
            # compute barycentric basis functions
            denom = 1.
            for dd in range(num_act_dims_pt):
                dim = act_dims_pt_persistent[dd]
                act_dim_idx = act_dim_indices_pt_persistent[dd]
                num_abscissa = num_abscissa_1d[act_dim_idx]
                x_dim_k = x[dim, kk]
                bases[0, dd] = abscissa_and_weights[1, act_dim_idx] /\
                    (x_dim_k - abscissa_and_weights[0, act_dim_idx])
                denom_d = bases[0, dd]
                for ii in range(1, num_abscissa):
                    basis = abscissa_and_weights[2*ii+1, act_dim_idx] /\
                        (x_dim_k - abscissa_and_weights[2*ii, act_dim_idx])
                    bases[ii, dd] = basis
                    denom_d += basis

                denom *= denom_d

            if (num_act_dims_pt == 0):
                # if point is an abscissa return the fn value at that point
                fn_val_index = np.sum(multi_index*shifts)
                result[kk, :] = fn_vals[fn_val_index, :]
            else:
                # compute interpolant
                c_persistent[:, :] = 0.
                done = True
                if (num_act_dims_pt > 1):
                    done = False
                fn_val_index = np.sum(multi_index*shifts)
                while (True):
                    act_dim_idx = act_dim_indices_pt_persistent[0]
                    for ii in range(num_active_abscissa_1d[act_dim_idx]):
                        fn_val_index += shifts[act_dim_idx] * \
                            (ii-multi_index[act_dim_idx])
                        multi_index[act_dim_idx] = ii
                        basis = bases[active_abscissa_indices_1d[act_dim_idx][ii], 0]
                        c_persistent[:, 0] += basis * fn_vals[fn_val_index, :]

                    for dd in range(1, num_act_dims_pt):
                        act_dim_idx = act_dim_indices_pt_persistent[dd]
                        basis = bases[active_abscissa_indices_1d[act_dim_idx]
                                      [multi_index[act_dim_idx]], dd]
                        c_persistent[:, dd] += basis * c_persistent[:, dd-1]
                        c_persistent[:, dd-1] = 0.
                        if (multi_index[act_dim_idx] <
                                num_active_abscissa_1d[act_dim_idx]-1):
                            fn_val_index += shifts[act_dim_idx]
                            multi_index[act_dim_idx] += 1
                            break
                        elif (dd < num_act_dims_pt - 1):
                            fn_val_index -= shifts[act_dim_idx] * \
                                multi_index[act_dim_idx]
                            multi_index[act_dim_idx] = 0
                        else:
                            done = True
                    if (done):
                        break
                result[kk, :] = c_persistent[:, num_act_dims_pt-1] / denom
                if np.any(np.isnan(result[kk, :])):
                    # print (c_persistent [:,num_act_dims_pt-1])
                    # print (denom)
                    raise ValueError('Error values not finite')
    return result


def multivariate_barycentric_lagrange_interpolation(
        x, abscissa_1d, barycentric_weights_1d, fn_vals, active_dims):

    num_active_dims = active_dims.shape[0]
    active_abscissa_indices_1d = []
    for active_index in range(num_active_dims):
        active_abscissa_indices_1d.append(np.arange(
            abscissa_1d[active_index].shape[0]))

    return multivariate_hierarchical_barycentric_lagrange_interpolation(
        x, abscissa_1d, barycentric_weights_1d, fn_vals, active_dims,
        active_abscissa_indices_1d)


def clenshaw_curtis_barycentric_weights(level):
    if (level == 0):
        return np.array([0.5], float)
    else:
        mi = 2**(level) + 1
        w = np.ones(mi, np.double)
        w[0] = 0.5
        w[mi-1] = 0.5
        w[1::2] = -1.
        return w


def equidistant_barycentric_weights(n):
    w = np.zeros(n, np.double)
    for i in range(0, n - n % 2, 2):
        w[i] = 1. * nchoosek(n-1, i)
        w[i+1] = -1. * nchoosek(n-1, i+1)
    if (n % 2 == 1):
        w[n-1] = 1.
    return w


@njit(cache=True)
def univariate_lagrange_polynomial(abscissa, samples):
    assert abscissa.ndim == 1
    assert samples.ndim == 1
    nabscissa = abscissa.shape[0]
    values = np.ones((samples.shape[0], nabscissa), dtype=np.double)
    for ii in range(nabscissa):
        x_ii = abscissa[ii]
        for jj in range(nabscissa):
            if ii == jj:
                continue
            values[:, ii] *= (samples - abscissa[jj])/(x_ii-abscissa[jj])
    return values


def precompute_tensor_product_lagrange_polynomial_basis(
        samples, abscissa_1d, active_vars):

    nvars, nsamples = samples.shape
    nactive_vars = len(active_vars)
    nabscissa = np.empty(nactive_vars, dtype=np.int64)
    for dd in range(nactive_vars):
        nabscissa[dd] = abscissa_1d[dd].shape[0]
    max_nabscissa = nabscissa.max()

    basis_vals_1d = np.empty(
        (nactive_vars, max_nabscissa, nsamples), dtype=np.double)
    for dd in range(nactive_vars):
        basis_vals_1d[dd, :nabscissa[dd], :] = univariate_lagrange_polynomial(
            abscissa_1d[dd], samples[active_vars[dd]]).T
    return basis_vals_1d


def __tensor_product_lagrange_polynomial_basis(
        samples, basis_vals_1d, active_vars, values, active_indices):

    #try:
    from pyapprox.cython.barycentric_interpolation import \
        tensor_product_lagrange_interpolation_pyx
    approx_values = tensor_product_lagrange_interpolation_pyx(
        samples, values, basis_vals_1d, active_indices, active_vars)

    return approx_values

    # nvars, nsamples = samples.shape
    # nactive_vars = len(active_vars)

    # nindices = active_indices.shape[1]
    # temp1 = basis_vals_1d.reshape(
    #     (nactive_vars*basis_vals_1d.shape[1], nsamples))
    # temp2 = temp1[active_indices.ravel()+np.repeat(
    #     np.arange(nactive_vars)*basis_vals_1d.shape[1], nindices), :].reshape(
    #         nactive_vars, nindices, nsamples)
    # basis_matrix = np.prod(temp2, axis=0).T
    # approx_values = basis_matrix.dot(values)

    # prod with axis argument does not work with njit
    # approx_values = np.zeros((nsamples, values.shape[1]), dtype=np.double)
    # for jj in range(nindices):
    #     basis_vals = 1
    #     for dd in range(nactive_vars):
    #         basis_vals *= basis_vals_1d[dd, active_indices[dd, jj], :]
    #     approx_values += basis_vals[:, None]*values[jj, :]
    return approx_values


def tensor_product_lagrange_interpolation(
        samples, abscissa_1d, active_vars, values):
    assert len(abscissa_1d) == len(active_vars)
    active_indices = cartesian_product(
        [np.arange(x.shape[0]) for x in abscissa_1d])
    basis_vals_1d = precompute_tensor_product_lagrange_polynomial_basis(
        samples, abscissa_1d, active_vars)
    return __tensor_product_lagrange_polynomial_basis(
        samples, basis_vals_1d, active_vars, values, active_indices)


def tensor_product_barycentric_lagrange_interpolation(
        grid_samples_1d, fun, samples, return_all=False):
    """
    Use tensor-product Barycentric Lagrange interpolation to approximate a
    function.

    Parameters
    ----------
    grid_samples_1d : list (nvars)
        List containing 1D grid points defining the tensor product grid
        The ith entry is a np.ndarray (nsamples_ii)

    fun : callable
        Function with the signature

        `fun(samples) -> np.ndarray (nx, nqoi)`

        where samples is np.ndarray (nvars, nx)

    samples : np.ndarray (nvars, nsamples)
        The samples at which to evaluate the basis functions

    Returns
    -------
    interp_vals : np.ndarray (nsamples, nqoi)
        Evaluations of the interpolant at the samples

    grid_samples : np.ndarray (nvars, ngrid_samples)
        if return_all: The samples used to consruct the basis functions where
        ngrid_samples = prod([len(s) for s in grid_samples_1d])
    """
    barycentric_weights_1d = [
        compute_barycentric_weights_1d(ss) for ss in grid_samples_1d]

    grid_samples = cartesian_product(grid_samples_1d)
    fn_vals = fun(grid_samples)
    interp_vals = multivariate_barycentric_lagrange_interpolation(
        samples, grid_samples_1d, barycentric_weights_1d, fn_vals,
        np.arange(samples.shape[0]))
    if not return_all:
        return interp_vals
    return interp_vals, grid_samples
