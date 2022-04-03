from functools import partial

from warnings import warn

import numpy as np
from numpy.polynomial.legendre import leggauss

from scipy.linalg import solve_triangular
from scipy.linalg import lapack

from pyapprox.utilities.pya_numba import njit
from pyapprox.utilities.sys_utilities import hash_array


def sub2ind(sizes, multi_index):
    r"""
    Map a d-dimensional index to the scalar index of the equivalent flat
    1D array

    Examples
    --------

    .. math::

       \begin{bmatrix}
       0,0 & 0,1 & 0,2\\
       1,0 & 1,1 & 1,2\\
       2,0 & 2,1 & 2,2
       \end{bmatrix}
       \rightarrow
       \begin{bmatrix}
       0 & 3 & 6\\
       1 & 4 & 7\\
       2 & 5 & 8
       \end{bmatrix}

    >>> from pyapprox.utilities import sub2ind
    >>> sizes = [3,3]
    >>> ind = sub2ind(sizes,[1,0])
    >>> print(ind)
    1

    Parameters
    ----------
    sizes : integer
        The number of elems in each dimension. For a 2D index
        sizes = [numRows, numCols]

    multi_index : np.ndarray (len(sizes))
       The d-dimensional index

    Returns
    -------
    scalar_index : integer
        The scalar index

    See Also
    --------
    pyapprox.utilities.sub2ind
    """
    num_sets = len(sizes)
    scalar_index = 0
    shift = 1
    for ii in range(num_sets):
        scalar_index += shift * multi_index[ii]
        shift *= sizes[ii]
    return scalar_index


def ind2sub(sizes, scalar_index, num_elems):
    r"""
    Map a scalar index of a flat 1D array to the equivalent d-dimensional index

    Examples
    --------

    .. math::

        \begin{bmatrix}
        0 & 3 & 6\\
        1 & 4 & 7\\
        2 & 5 & 8
        \end{bmatrix}
        \rightarrow
        \begin{bmatrix}
        0,0 & 0,1 & 0,2\\
        1,0 & 1,1 & 1,2\\
        2,0 & 2,1 & 2,2
        \end{bmatrix}

    >>> from pyapprox.utilities import ind2sub
    >>> sizes = [3,3]
    >>> sub = ind2sub(sizes,1,9)
    >>> print(sub)
    [1 0]

    Parameters
    ----------
    sizes : integer
        The number of elems in each dimension. For a 2D index
        sizes = [numRows, numCols]

    scalar_index : integer
        The scalar index

    num_elems : integer
        The total number of elements in the d-dimensional matrix

    Returns
    -------
    multi_index : np.ndarray (len(sizes))
       The d-dimensional index

    See Also
    --------
    pyapprox.utilities.sub2ind
    """
    denom = num_elems
    num_sets = len(sizes)
    multi_index = np.empty((num_sets), dtype=int)
    for ii in range(num_sets-1, -1, -1):
        denom /= sizes[ii]
        multi_index[ii] = scalar_index / denom
        scalar_index = scalar_index % denom
    return multi_index


def cartesian_product(input_sets, elem_size=1):
    r"""
    Compute the cartesian product of an arbitray number of sets.

    The sets can consist of numbers or themselves be lists or vectors. All
    the lists or vectors of a given set must have the same number of entries
    (elem_size). However each set can have a different number of scalars,
    lists, or vectors.

    Parameters
    ----------
    input_sets
        The sets to be used in the cartesian product.

    elem_size : integer
        The size of the vectors within each set.

    Returns
    -------
    result : np.ndarray (num_sets*elem_size, num_elems)
        The cartesian product. num_elems = np.prod(sizes)/elem_size,
        where sizes[ii] = len(input_sets[ii]), ii=0,..,num_sets-1.
        result.dtype will be set to the first entry of the first input_set
    """
    import itertools
    out = []
    # ::-1 reverse order to be backwards compatiable with old
    # function below
    for r in itertools.product(*input_sets[::-1]):
        out.append(r)
    out = np.asarray(out).T[::-1, :]
    return out

    # try:
    #     from pyapprox.cython.utilities import cartesian_product_pyx
    #     # # fused type does not work for np.in32, np.float32, np.int64
    #     # # so envoke cython cast
    #     # if np.issubdtype(input_sets[0][0],np.signedinteger):
    #     #     return cartesian_product_pyx(input_sets,1,elem_size)
    #     # if np.issubdtype(input_sets[0][0],np.floating):
    #     #     return cartesian_product_pyx(input_sets,1.,elem_size)
    #     # else:
    #     #     return cartesian_product_pyx(
    #     #         input_sets,input_sets[0][0],elem_size)
    #     # always convert to float then cast back
    #     cast_input_sets = [np.asarray(s, dtype=float) for s in input_sets]
    #     out = cartesian_product_pyx(cast_input_sets, 1., elem_size)
    #     out = np.asarray(out, dtype=input_sets[0].dtype)
    #     return out
    # except:
    #     print('cartesian_product extension failed')

    # num_elems = 1
    # num_sets = len(input_sets)
    # sizes = np.empty((num_sets), dtype=int)
    # for ii in range(num_sets):
    #     sizes[ii] = input_sets[ii].shape[0]/elem_size
    #     num_elems *= sizes[ii]
    # # try:
    # #    from pyapprox.weave import c_cartesian_product
    # #    # note c_cartesian_product takes_num_elems as last arg and cython
    # #    # takes elem_size
    # #    return c_cartesian_product(input_sets, elem_size, sizes, num_elems)
    # # except:
    # #    print ('cartesian_product extension failed')

    # result = np.empty(
    #     (num_sets*elem_size, num_elems), dtype=type(input_sets[0][0]))
    # for ii in range(num_elems):
    #     multi_index = ind2sub(sizes, ii, num_elems)
    #     for jj in range(num_sets):
    #         for kk in range(elem_size):
    #             result[jj*elem_size+kk, ii] =\
    #                 input_sets[jj][multi_index[jj]*elem_size+kk]
    # return result


def outer_product(input_sets, axis=0):
    r"""
    Construct the outer product of an arbitary number of sets.

    Examples
    --------

    .. math::

        \{1,2\}\times\{3,4\}=\{1\times3, 2\times3, 1\times4, 2\times4\} =
        \{3, 6, 4, 8\}

    Parameters
    ----------
    input_sets
        The sets to be used in the outer product

    Returns
    -------
    result : np.ndarray(np.prod(sizes))
       The outer product of the sets.
       result.dtype will be set to the first entry of the first input_set
    """
    out = cartesian_product(input_sets)
    return np.prod(out, axis=axis)

    # try:
    #     from pyapprox.cython.utilities import outer_product_pyx
    #     # fused type does not work for np.in32, np.float32, np.int64
    #     # so envoke cython cast
    #     if np.issubdtype(input_sets[0][0], np.signedinteger):
    #         return outer_product_pyx(input_sets, 1)
    #     if np.issubdtype(input_sets[0][0], np.floating):
    #         return outer_product_pyx(input_sets, 1.)
    #     else:
    #         return outer_product_pyx(input_sets, input_sets[0][0])
    # except ImportError:
    #     print('outer_product extension failed')

    # num_elems = 1
    # num_sets = len(input_sets)
    # sizes = np.empty((num_sets), dtype=int)
    # for ii in range(num_sets):
    #     sizes[ii] = len(input_sets[ii])
    #     num_elems *= sizes[ii]

    # # try:
    # #     from pyapprox.weave import c_outer_product
    # #     return c_outer_product(input_sets)
    # # except:
    # #     print ('outer_product extension failed')

    # result = np.empty((num_elems), dtype=type(input_sets[0][0]))
    # for ii in range(num_elems):
    #     result[ii] = 1.0
    #     multi_index = ind2sub(sizes, ii, num_elems)
    #     for jj in range(num_sets):
    #         result[ii] *= input_sets[jj][multi_index[jj]]

    # return result


def unique_matrix_rows(matrix):
    unique_rows = []
    unique_rows_set = set()
    for ii in range(matrix.shape[0]):
        key = hash_array(matrix[ii, :])
        if key not in unique_rows_set:
            unique_rows_set.add(key)
            unique_rows.append(matrix[ii, :])
    return np.asarray(unique_rows)


def remove_common_rows(matrices):
    num_cols = matrices[0].shape[1]
    unique_rows_dict = dict()
    for ii in range(len(matrices)):
        matrix = matrices[ii]
        assert matrix.shape[1] == num_cols
        for jj in range(matrix.shape[0]):
            key = hash_array(matrix[jj, :])
            if key not in unique_rows_dict:
                unique_rows_dict[key] = (ii, jj)
            elif unique_rows_dict[key][0] != ii:
                del unique_rows_dict[key]
            # else:
            # entry is a duplicate entry in the current. Allow this to
            # occur but only add one of the duplicates to the unique rows dict

    unique_rows = []
    for key in list(unique_rows_dict.keys()):
        ii, jj = unique_rows_dict[key]
        unique_rows.append(matrices[ii][jj, :])

    return np.asarray(unique_rows)


def allclose_unsorted_matrix_rows(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        return False

    matrix1_dict = dict()
    for ii in range(matrix1.shape[0]):
        key = hash_array(matrix1[ii, :])
        # allow duplicates of rows
        if key not in matrix1_dict:
            matrix1_dict[key] = 0
        else:
            matrix1_dict[key] += 1

    matrix2_dict = dict()
    for ii in range(matrix2.shape[0]):
        key = hash_array(matrix2[ii, :])
        # allow duplicates of rows
        if key not in matrix2_dict:
            matrix2_dict[key] = 0
        else:
            matrix2_dict[key] += 1

    if len(list(matrix1_dict.keys())) != len(list(matrix2_dict.keys())):
        return False

    for key in list(matrix1_dict.keys()):
        if key not in matrix2_dict:
            return False
        if matrix2_dict[key] != matrix1_dict[key]:
            return False

    return True


def get_2d_cartesian_grid(num_pts_1d, ranges):
    r"""
    Get a 2d tensor grid with equidistant points.

    Parameters
    ----------
    num_pts_1d : integer
        The number of points in each dimension

    ranges : np.ndarray (4)
        The lower and upper bound of each dimension [lb_1,ub_1,lb_2,ub_2]

    Returns
    -------
    grid : np.ndarray (2,num_pts_1d**2)
        The points in the tensor product grid.
        [x1,x2,...x1,x2...]
        [y1,y1,...y2,y2...]
    """
    # from math_tools_cpp import cartesian_product_double as cartesian_product
    from PyDakota.math_tools import cartesian_product
    x1 = np.linspace(ranges[0], ranges[1], num_pts_1d)
    x2 = np.linspace(ranges[2], ranges[3], num_pts_1d)
    abscissa_1d = []
    abscissa_1d.append(x1)
    abscissa_1d.append(x2)
    grid = cartesian_product(abscissa_1d, 1)
    return grid


def nchoosek(nn, kk):
    try:  # SciPy >= 0.19
        from scipy.special import comb
    except ImportError:
        from scipy.misc import comb
    result = np.asarray(np.round(comb(nn, kk)), dtype=int)
    if np.isscalar(result):
        result = np.asscalar(result)
    return result


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

    # I = np.argsort(samples)
    # sorted_samples = samples[I]
    # idx2=0
    # for ii in range(0,mesh.shape[0]-2,2):
    #     xl=mesh[ii]; xr=mesh[ii+2]
    #     for jj in range(idx2,sorted_samples.shape[0]):
    #         if ii==0:
    #             if sorted_samples[jj]>=xl:
    #                 idx1=jj
    #                 break
    #         else:
    #             if sorted_samples[jj]>xl:
    #                 idx1=jj
    #                 break
    #     for jj in range(idx1,sorted_samples.shape[0]):
    #         if sorted_samples[jj]>xr:
    #             idx2=jj-1
    #             break
    #     if jj==sorted_samples.shape[0]-1:
    #         idx2=jj
    #     x=(sorted_samples[idx1:idx2+1]-xl)/(xr-xl)
    #     interval_vals = canonical_piecewise_quadratic_interpolation(
    #         x,mesh_vals[ii:ii+3])
    #     vals[idx1:idx2+1] += interval_vals
    # return vals[np.argsort(I)]


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


def lists_of_arrays_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    for ll in range(len(list1)):
        if not np.allclose(list1[ll], list2[ll]):
            return False
    return True


def lists_of_lists_of_arrays_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    for ll in range(len(list1)):
        for kk in range(len(list1[ll])):
            if not np.allclose(list1[ll][kk], list2[ll][kk]):
                return False
    return True


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


def get_all_primes_less_than_or_equal_to_n(n):
    primes = list()
    primes.append(2)
    for num in range(3, n+1, 2):
        if all(num % i != 0 for i in range(2, int(num**.5) + 1)):
            primes.append(num)
    return np.asarray(primes)


@njit(cache=True)
def get_first_n_primes(n):
    primes = list()
    primes.append(2)
    num = 3
    while len(primes) < n:
        # np.all does not work with numba
        # if np.all([num % i != 0 for i in range(2, int(num**.5) + 1)]):
        flag = True
        for i in range(2, int(num**.5) + 1):
            if (num % i == 0):
                flag = False
                break
        if flag is True:
            primes.append(num)
        num += 2
    return np.asarray(primes)


def approx_fprime(x, func, eps=np.sqrt(np.finfo(float).eps)):
    r"""Approx the gradient of a vector valued function at a single
    sample using finite_difference
    """
    assert x.shape[1] == 1
    nvars = x.shape[0]
    fprime = []
    func_at_x = func(x).squeeze()
    assert func_at_x.ndim == 1
    for ii in range(nvars):
        x_plus_eps = x.copy()
        x_plus_eps[ii] += eps
        fprime.append((func(x_plus_eps).squeeze()-func_at_x)/eps)
    return np.array(fprime)


def partial_functions_equal(func1, func2):
    if not (isinstance(func1, partial) and isinstance(func2, partial)):
        return False
    are_equal = all([getattr(func1, attr) == getattr(func2, attr)
                     for attr in ['func', 'args', 'keywords']])
    return are_equal


def get_all_sample_combinations(samples1, samples2):
    r"""
    For two sample sets of different random variables
    loop over all combinations

    samples1 vary slowest and samples2 vary fastest

    Let samples1 = [[1,2],[2,3]]
        samples2 = [[0, 0, 0],[0, 1, 2]]

    Then samples will be

    ([1, 2, 0, 0, 0])
    ([1, 2, 0, 1, 2])
    ([3, 4, 0, 0, 0])
    ([3, 4, 0, 1, 2])

    """
    import itertools
    samples = []
    for r in itertools.product(*[samples1.T, samples2.T]):
        samples.append(np.concatenate(r))
    return np.asarray(samples).T


def get_correlation_from_covariance(cov):
    r"""
    Compute the correlation matrix from a covariance matrix

    Parameters
    ----------
    cov : np.ndarray (nrows,nrows)
        The symetric covariance matrix

    Returns
    -------
    cor : np.ndarray (nrows,nrows)
        The symetric correlation matrix

    Examples
    --------
    >>> cov = np.asarray([[2,-1],[-1,2]])
    >>> get_correlation_from_covariance(cov)
    array([[ 1. , -0.5],
           [-0.5,  1. ]])
    """
    stdev_inv = 1/np.sqrt(np.diag(cov))
    cor = stdev_inv[np.newaxis, :]*cov*stdev_inv[:, np.newaxis]
    return cor


def evaluate_quadratic_form(matrix, samples):
    r"""
    Evaluate x.T.dot(A).dot(x) for several vectors x

    Parameters
    ----------
    num_samples : np.ndarray (nvars,nsamples)
        The vectors x

    matrix : np.ndarray(nvars,nvars)
        The matrix A

    Returns
    -------
    vals : np.ndarray (nsamples)
        Evaluations of the quadratic form for each vector x
    """
    return (samples.T.dot(matrix)*samples.T).sum(axis=1)


def split_dataset(samples, values, ndata1):
    """
    Split a data set into two sets.

    Parameters
    ----------
    samples : np.ndarray (nvars,nsamples)
        The samples to be split

    values : np.ndarray (nsamples,nqoi)
        Values of the data at ``samples``

    ndata1 : integer
        The number of samples allocated to the first split. All remaining
        samples will be added to the second split.

    Returns
    -------
    samples1 : np.ndarray (nvars,ndata1)
        The samples of the first split data set

    values1 : np.ndarray (nvars,ndata1)
        The values of the first split data set

    samples2 : np.ndarray (nvars,ndata1)
        The samples of the first split data set

    values2 : np.ndarray (nvars,ndata1)
        The values of the first split data set
    """
    assert ndata1 <= samples.shape[1]
    assert values.shape[0] == samples.shape[1]
    II = np.random.permutation(samples.shape[1])
    samples1 = samples[:, II[:ndata1]]
    samples2 = samples[:, II[ndata1:]]
    values1 = values[II[:ndata1], :]
    values2 = values[II[ndata1:], :]
    return samples1, samples2, values1, values2


def leave_one_out_lsq_cross_validation(basis_mat, values, alpha=0, coef=None):
    """
    let :math:`x_i` be the ith row of :math:`X` and let
    :math:`\beta=(X^\top X)^{-1}X^\top y` such that the residuals
    at the training samples satisfy

    .. math:: r_i = X\beta-y

    then the leave one out cross validation errors are given by

    .. math:: e_i = \frac{r_i}{1-h_i}

    where

    :math:`h_i = x_i^\top(X^\top X)^{-1}x_i`
    """
    assert values.ndim == 2
    assert basis_mat.shape[0] > basis_mat.shape[1]+2
    gram_mat = basis_mat.T.dot(basis_mat)
    gram_mat += alpha*np.eye(gram_mat.shape[0])
    H_mat = basis_mat.dot(np.linalg.inv(gram_mat).dot(basis_mat.T))
    H_diag = np.diag(H_mat)
    if coef is None:
        coef = np.linalg.lstsq(
            gram_mat, basis_mat.T.dot(values), rcond=None)[0]
    assert coef.ndim == 2
    residuals = basis_mat.dot(coef) - values
    cv_errors = residuals / (1-H_diag[:, None])
    cv_score = np.sqrt(np.sum(cv_errors**2, axis=0)/basis_mat.shape[0])
    return cv_errors, cv_score, coef


def leave_many_out_lsq_cross_validation(basis_mat, values, fold_sample_indices,
                                        alpha=0, coef=None):
    nfolds = len(fold_sample_indices)
    nsamples = basis_mat.shape[0]
    cv_errors = []
    cv_score = 0
    gram_mat = basis_mat.T.dot(basis_mat)
    gram_mat += alpha*np.eye(gram_mat.shape[0])
    if coef is None:
        coef = np.linalg.lstsq(
            gram_mat, basis_mat.T.dot(values), rcond=None)[0]
    residuals = basis_mat.dot(coef) - values
    gram_mat_inv = np.linalg.inv(gram_mat)
    for kk in range(nfolds):
        indices_kk = fold_sample_indices[kk]
        nvalidation_samples_kk = indices_kk.shape[0]
        assert nsamples - nvalidation_samples_kk >= basis_mat.shape[1]
        basis_mat_kk = basis_mat[indices_kk, :]
        residuals_kk = residuals[indices_kk, :]

        H_mat = np.eye(nvalidation_samples_kk) - basis_mat_kk.dot(
            gram_mat_inv.dot(basis_mat_kk.T))
        # print('gram_mat cond number', np.linalg.cond(gram_mat))
        # print('H_mat cond number', np.linalg.cond(H_mat))
        H_mat_inv = np.linalg.inv(H_mat)
        cv_errors.append(H_mat_inv.dot(residuals_kk))
        cv_score += np.sum(cv_errors[-1]**2, axis=0)
    return np.asarray(cv_errors), np.sqrt(cv_score/basis_mat.shape[0]), coef


def get_random_k_fold_sample_indices(nsamples, nfolds, random=True):
    sample_indices = np.arange(nsamples)
    if random is True:
        sample_indices = np.random.permutation(sample_indices)
    fold_sample_indices = [np.empty(0, dtype=int) for kk in range(nfolds)]
    nn = 0
    while nn < nsamples:
        for jj in range(nfolds):
            fold_sample_indices[jj] = np.append(
                fold_sample_indices[jj], sample_indices[nn])
            nn += 1
            if nn >= nsamples:
                break
    assert np.unique(np.hstack(fold_sample_indices)).shape[0] == nsamples
    return fold_sample_indices


def get_cross_validation_rsquared_coefficient_of_variation(
        cv_score, train_vals):
    r"""
    cv_score = :math:`N^{-1/2}\left(\sum_{n=1}^N e_n\right^{1/2}` where
    :math:`e_n` are the cross  validation residues at each test point and
    :math:`N` is the number of traing vals

    We define r_sq as

    .. math:: 1-\frac{N^{-1}\left(\sum_{n=1}^N e_n\right)}/mathbb{V}\left[Y\right] where Y is the vector of training vals
    """
    # total sum of squares (proportional to variance)
    denom = np.std(train_vals)
    # the factors of 1/N in numerator and denominator cancel out
    rsq = 1-(cv_score/denom)**2
    return rsq


def __integrate_using_univariate_gauss_legendre_quadrature_bounded(
        integrand, lb, ub, nquad_samples, rtol=1e-8, atol=1e-8,
        verbose=0, adaptive=True, tabulated_quad_rules=None):
    """
    tabulated_quad_rules : dictionary
        each entry is a tuple (x,w) of gauss legendre with weight
        function p(x)=1 defined on [-1,1]. The number of points in x is
        defined by the key.
        User must ensure that the dictionary contains any nquad_samples
        that may be requested
    """
    # Adaptive
    # nquad_samples = 10
    prev_res = np.inf
    it = 0
    while True:
        if (tabulated_quad_rules is None or
                nquad_samples not in tabulated_quad_rules):
            xx_canonical, ww_canonical = leggauss(nquad_samples)
        else:
            xx_canonical, ww_canonical = tabulated_quad_rules[nquad_samples]
        xx = (xx_canonical+1)/2*(ub-lb)+lb
        ww = ww_canonical*(ub-lb)/2
        res = integrand(xx).T.dot(ww).T
        diff = np.absolute(prev_res-res)
        if verbose > 1:
            print(it, nquad_samples, diff)
        if (np.all(np.absolute(prev_res-res) < rtol*np.absolute(res)+atol) or
                adaptive is False):
            break
        prev_res = res
        nquad_samples *= 2
        it += 1
    if verbose > 0:
        print(f'adaptive quadrature converged in {it} iterations')
    return res


def integrate_using_univariate_gauss_legendre_quadrature_unbounded(
        integrand, lb, ub, nquad_samples, atol=1e-8, rtol=1e-8,
        interval_size=2, max_steps=1000, verbose=0, adaptive=True,
        soft_error=False, tabulated_quad_rules=None):
    """
    Compute unbounded integrals by moving left and right from origin.
    Assume that integral decays towards +/- infinity. And that once integral
    over a sub interval drops below tolerance it will not increase again if
    we keep moving in same direction.
    """
    if interval_size <= 0:
        raise ValueError("Interval size must be positive")

    if np.isfinite(lb) and np.isfinite(ub):
        partial_lb, partial_ub = lb, ub
    elif np.isfinite(lb) and not np.isfinite(ub):
        partial_lb, partial_ub = lb, lb+interval_size
    elif not np.isfinite(lb) and np.isfinite(ub):
        partial_lb, partial_ub = ub-interval_size, ub
    else:
        partial_lb, partial_ub = -interval_size/2, interval_size/2

    result = __integrate_using_univariate_gauss_legendre_quadrature_bounded(
        integrand, partial_lb, partial_ub, nquad_samples, rtol,
        atol, verbose-1, adaptive, tabulated_quad_rules)

    step = 0
    partial_result = np.inf
    plb, pub = partial_lb-interval_size, partial_lb
    while (np.any(np.absolute(partial_result) >= rtol*np.absolute(result)+atol)
           and (plb >= lb) and step < max_steps):
        partial_result = \
            __integrate_using_univariate_gauss_legendre_quadrature_bounded(
                integrand, plb, pub, nquad_samples, rtol, atol,
                verbose-1, adaptive, tabulated_quad_rules)
        result += partial_result
        pub = plb
        plb -= interval_size
        step += 1
        if verbose > 1:
            print('Left', step, result, partial_result, plb, pub,
                  interval_size)
        if verbose > 0:
            if step >= max_steps:
                msg = "Early termination when computing left integral"
                msg += f"max_steps {max_steps} reached"
                if soft_error is True:
                    warn(msg, UserWarning)
                else:
                    raise RuntimeError(msg)
            if np.all(np.abs(partial_result) < rtol*np.absolute(result)+atol):
                msg = f'Tolerance {atol} {rtol} for left integral reached in '
                msg += f'{step} iterations'
                print(msg)

    step = 0
    partial_result = np.inf
    plb, pub = partial_ub, partial_ub+interval_size
    while (np.any(np.absolute(partial_result) >= rtol*np.absolute(result)+atol)
           and (pub <= ub) and step < max_steps):
        partial_result = \
            __integrate_using_univariate_gauss_legendre_quadrature_bounded(
                integrand, plb, pub, nquad_samples, rtol, atol,
                verbose-1, adaptive, tabulated_quad_rules)
        result += partial_result
        plb = pub
        pub += interval_size
        step += 1
        if verbose > 1:
            print('Right', step, result, partial_result, plb, pub,
                  interval_size)
        if verbose > 0:
            if step >= max_steps:
                msg = "Early termination when computing right integral. "
                msg += f"max_steps {max_steps} reached"
                if soft_error is True:
                    warn(msg, UserWarning)
                else:
                    raise RuntimeError(msg)
            if np.all(np.abs(partial_result) < rtol*np.absolute(result)+atol):
                msg = f'Tolerance {atol} {rtol} for right integral reached in '
                msg += f'{step} iterations'
                print(msg)
        # print(partial_result, plb, pub)

    return result


def qr_solve(Q, R, rhs):
    """
    Find the least squares solution Ax = rhs given a QR factorization of the
    matrix A

    Parameters
    ----------
    Q : np.ndarray (nrows, nrows)
        The unitary/upper triangular Q factor

    R : np.ndarray (nrows, ncols)
        The upper triangular R matrix

    rhs : np.ndarray (nrows, nqoi)
        The right hand side vectors

    Returns
    -------
    x : np.ndarray (nrows, nqoi)
        The solution
    """
    tmp = np.dot(Q.T, rhs)
    return solve_triangular(R, tmp, lower=False)


def equality_constrained_linear_least_squares(A, B, y, z):
    """
    Solve equality constrained least squares regression

    minimize || y - A*x ||_2   subject to   B*x = z

    It is assumed that

    Parameters
    ----------
    A : np.ndarray (M, N)
        P <= N <= M+P, and

    B : np.ndarray (N, P)
        P <= N <= M+P, and

    y : np.ndarray (M, 1)
        P <= N <= M+P, and

    z : np.ndarray (P, 1)
        P <= N <= M+P, and

    Returns
    -------
    x : np.ndarray (N, 1)
        The solution
    """
    return lapack.dgglse(A, B, y, z)[3]


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


def extract_sub_list(mylist, indices):
    """
    Extract a subset of items from a list

    Parameters
    ----------
    mylist : list(nitems)
        The list containing all items

    indices : iterable (nindices)
        The indices of the desired items

    Returns
    -------
    subset :  list (nindices)
        The extracted items
    """
    return [mylist[ii] for ii in indices]


def unique_elements_from_2D_list(list_2d):
    """
    Extract the unique elements from a list of lists

    Parameters
    ----------
    list_2d : list(list)
        The list of lists

    Returns
    -------
    unique_items :  list (nunique_items)
        The unique items
    """
    return list(set(flatten_2D_list(list_2d)))


def flatten_2D_list(list_2d):
    """
    Flatten a list of lists into a single list

    Parameters
    ----------
    list_2d : list(list)
        The list of lists

    Returns
    -------
    flattened_list :  list (nitems)
        The unique items
    """
    return [item for sub in list_2d for item in sub]


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
