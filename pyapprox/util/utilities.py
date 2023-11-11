from warnings import warn
from functools import partial

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_hermitenorm

from scipy.linalg import solve_triangular
from scipy.linalg import lapack

import matplotlib.pyplot as plt

from pyapprox.util.pya_numba import njit
from pyapprox.util.sys_utilities import hash_array
from pyapprox.util.sys_utilities import has_kwarg


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

    >>> from pyapprox.util.utilities import sub2ind
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
    pyapprox.util.sub2ind
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

    >>> from pyapprox.util.utilities import ind2sub
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
    pyapprox.util.sub2ind
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


def unique_matrix_row_indices(matrix):
    unique_row_indices = []
    unique_rows_set = set()
    for ii in range(matrix.shape[0]):
        key = hash_array(matrix[ii, :])
        if key not in unique_rows_set:
            unique_rows_set.add(key)
            unique_row_indices.append(ii)
    return np.array(unique_row_indices)


def unique_matrix_rows(matrix):
    return matrix[unique_matrix_row_indices(matrix)]
    # unique_rows = []
    # unique_rows_set = set()
    # for ii in range(matrix.shape[0]):
    #     key = hash_array(matrix[ii, :])
    #     if key not in unique_rows_set:
    #         unique_rows_set.add(key)
    #         unique_rows.append(matrix[ii, :])
    # return np.asarray(unique_rows)


def common_matrix_rows(matrix):
    unique_rows_dict = dict()
    for ii in range(matrix.shape[0]):
        key = hash_array(matrix[ii, :])
        if key not in unique_rows_dict:
            unique_rows_dict[key] = [ii]
        else:
            unique_rows_dict[key].append(ii)
    return unique_rows_dict


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
    if result.ndim == 0:
        result = result.item()
        # result = np.asscalar(result)
    return result


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
    # if func_at_x.ndim == 2:
    #     func_at_x = func_at_x[:, 0]
    assert func_at_x.ndim < 2
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
    return cv_errors, np.sqrt(cv_score/basis_mat.shape[0]), coef


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

    .. math:: 1-\frac{N^{-1}\left(\sum_{n=1}^N e_n\right)}/mathbb{V}\left[
              Y\right] where Y is the vector of training vals
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


def approx_jacobian(func, x, *args, epsilon=np.sqrt(np.finfo(float).eps)):
    x0 = np.asfarray(x)
    assert x0.ndim == 1 or x0.shape[1] == 1
    f0 = np.atleast_1d(func(*((x0,)+args)))
    if f0.ndim == 2:
        assert f0.shape[1] == 1
        f0 = f0[:, 0]
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(x0.shape)
    for i in range(len(x0)):
        dx[i] = epsilon
        f1 = func(*((x0+dx,)+args))
        if f1.ndim == 2:
            assert f1.shape[1] == 1
            f1 = f1[:, 0]
        jac[i] = (f1 - f0)/epsilon
        dx[i] = 0.0

    return jac.transpose()


def _check_gradients(fun, zz, direction, plot, disp, rel, fd_eps):
    function_val, directional_derivative = fun(zz, direction)
    if isinstance(function_val, np.ndarray):
        function_val = function_val.squeeze()

    if fd_eps is None:
        fd_eps = np.logspace(-13, 0, 14)[::-1]
    errors = []
    row_format = "{:<12} {:<25} {:<25} {:<25}"
    if disp:
        if rel:
            print(
                row_format.format(
                    "Eps", "norm(jv)", "norm(jv_fd)",
                    "Rel. Errors"))
        else:
            print(row_format.format(
                "Eps", "norm(jv)", "norm(jv_fd)",
                "Abs. Errors"))
    row_format = "{:<12.2e} {:<25} {:<25} {:<25}"
    for ii in range(fd_eps.shape[0]):
        zz_perturbed = zz.copy()+fd_eps[ii]*direction
        # perturbed_function_val = fun(zz_perturbed)
        # add jac=False so that exact gradient is not always computed
        perturbed_function_val = fun(zz_perturbed, direction=None)
        if isinstance(perturbed_function_val, np.ndarray):
            perturbed_function_val = perturbed_function_val.squeeze()
        # print(inspect.getfullargspec(fun).args)
        # print(perturbed_function_val, function_val, fd_eps[ii])
        fd_directional_derivative = (
            perturbed_function_val-function_val)/fd_eps[ii]
        # print(fd_directional_derivative)
        errors.append(np.linalg.norm(
            fd_directional_derivative.reshape(directional_derivative.shape) -
            directional_derivative))
        if rel:
            errors[-1] /= np.linalg.norm(directional_derivative)

        if disp:
            print(row_format.format(
                fd_eps[ii],
                np.linalg.norm(directional_derivative),
                np.linalg.norm(fd_directional_derivative),
                errors[ii]))

    if plot:
        plt.loglog(fd_eps, errors, 'o-')
        plt.ylabel(r'$\lvert\nabla_\epsilon f\cdot p-\nabla f\cdot p\rvert$')
        plt.xlabel(r'$\epsilon$')
        plt.show()

    return np.asarray(errors)


def _wrap_function_with_gradient(fun, return_grad):
    if ((return_grad is not None) and not callable(return_grad) and
            (return_grad != "return_gradp") and (return_grad != True)):
        raise ValueError("return_grad must be callable, 'jacp', or None")

    if callable(return_grad):
        def fun_wrapper(x, direction=None):
            if direction is None:
                return fun(x)
            return fun(x), return_grad(x).dot(direction)
        return fun_wrapper

    if return_grad == True and has_kwarg(fun, "return_grad"):
        # this is PyApprox's preferred convention
        def fun_wrapper(x, direction=None):
            if direction is None:
                val = fun(x, return_grad=False)
                return val
            vals, grad = fun(x, return_grad=True)
            return vals, grad.dot(direction)
        return fun_wrapper

    if return_grad == True:
        def fun_wrapper(x, direction=None):
            if direction is None:
                return fun(x)[0]
            vals, grad = fun(x)
            return vals, grad.dot(direction)
        return fun_wrapper

    if return_grad == "jacp":
        assert has_kwarg(fun, "return_grad")
        # this is PyApprox's other preferred convention
        def fun_wrapper(x, direction=None):
            if direction is None:
                return fun(x, return_grad=False)
            val, grad = fun(x, return_grad=True)
            return fun(x), grad.dot(direction)
        return fun_wrapper
    return fun


def check_gradients(fun, jac, zz, plot=False, disp=True, rel=True,
                    direction=None, fd_eps=None):
    """
    Compare a user specified jacobian with the jacobian computed with finite
    difference with multiple step sizes.

    Parameters
    ----------
    fun : callable

        A function with one of the following signatures

        ``fun(z) -> (vals)``

        or

        ``fun(z, jac) -> (vals, grad)``

        or

        ``fun(z, direction) -> (vals, directional_grad)``

        where ``z`` is a 2D np.ndarray with shape (nvars, 1) and the
        first output is a 2D np.ndarray with shape (nqoi, 1) and the second
        output is a gradient with shape (nqoi, nvars).
        jac is a flag that specifies if the function returns only
        the funciton value (False) or the function value and gradient (True)

    jac : callable or string
        If jac="jacp" then provided the jacobian of ``fun`` with signature

        ``jac(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars, 1) and the
        output is a 2D np.ndarray with shape (nqoi, nvars).
        This assumes that fun
        only returns a value (not gradient) and has signature

        ``fun(z) -> np.ndarray``


    zz : np.ndarray (nvars, 1)
        A sample of ``z`` at which to compute the gradient

    plot : boolean
        Plot the errors as a function of the finite difference step size

    disp : boolean
        True - print the errors
        False - do not print

    rel : boolean
        True - compute the relative error in the directional derivative,
        i.e. the absolute error divided by the directional derivative using
        ``jac``.
        False - compute the absolute error in the directional derivative

    direction : np.ndarray (nvars, 1)
        Direction to which Jacobian is applied. Default is None in which
        case random direction is chosen.

    fd_eps : np.ndarray (nstep_sizes)
        The finite difference step sizes used to compute the gradient.
        If None then fd_eps=np.logspace(-13, 0, 14)[::-1]

    Returns
    -------
    errors : np.ndarray (14, nqoi)
        The errors in the directional derivative of ``fun`` at 14 different
        values of finite difference tolerance for each quantity of interest
    """
    assert zz.ndim == 2
    assert zz.shape[1] == 1

    fun_wrapper = _wrap_function_with_gradient(fun, jac)

    if direction is None:
        direction = np.random.normal(0, 1, (zz.shape[0], 1))
        direction /= np.linalg.norm(direction)
    assert direction.ndim == 2 and direction.shape[1] == 1

    return _check_gradients(
        fun_wrapper, zz, direction, plot, disp, rel, fd_eps)


def check_hessian(jac, hessian_matvec, zz, plot=False, disp=True, rel=True,
                  direction=None, fd_eps=np.logspace(-13, 0, 14)[::-1]):
    """
    Compare a user specified Hessian matrix-vector product with the
    Hessian matrix vector produced computed with finite
    difference with multiple step sizes using a user specified jacobian.

    Parameters
    ---------
    jac : callable
        The jacobian  with signature

        ``jac(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,1) and the
        output is a 2D np.ndarray with shape (nqoi,nvars)

    hessian_matvec : callable
        A function implementing the hessian matrix-vector product with
        signature

        ``hessian_matvec(z,p) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,1), ``p`` is
        an arbitrary vector with shape (nvars,1) and the
        output is a 2D np.ndarray with shape (nqoi,nvars)

    zz : np.ndarray (nvars,1)
        A sample of ``z`` at which to compute the gradient

    plot : boolean
        Plot the errors as a function of the finite difference step size

    disp : boolean
        True - print the errors
        False - do not print

    rel : boolean
        True - compute the relative error in the directional derivative,
        i.e. the absolute error divided by the directional derivative using
        ``jac``.
        False - compute the absolute error in the directional derivative

    direction : np.ndarray (nvars, 1)
        Direction to which Hessian is applied. Default is None in which
        case random direction is chosen.

    Returns
    -------
    errors : np.ndarray (14, nqoi)
        The errors in the directional derivative of ``jac`` at 14 different
        values of finite difference tolerance for each quantity of interest
    """
    assert zz.ndim == 2
    assert zz.shape[1] == 1
    grad = jac(zz)
    if direction is None:
        direction = np.random.normal(0, 1, (zz.shape[0], 1))
        direction /= np.linalg.norm(direction)
    directional_derivative = hessian_matvec(zz, direction)
    errors = []
    row_format = "{:<12} {:<25} {:<25} {:<25}"
    if disp:
        if rel:
            print(
                row_format.format(
                    "Eps", "norm(jv)", "norm(jv_fd)",
                    "Rel. Errors"))
        else:
            print(row_format.format(
                "Eps", "norm(jv)", "norm(jv_fd)",
                "Abs. Errors"))
    for ii in range(fd_eps.shape[0]):
        zz_perturbed = zz.copy()+fd_eps[ii]*direction
        perturbed_grad = jac(zz_perturbed)
        fd_directional_derivative = (perturbed_grad-grad)/fd_eps[ii]
        # print(directional_derivative, fd_directional_derivative)
        errors.append(np.linalg.norm(
            fd_directional_derivative.reshape(directional_derivative.shape) -
            directional_derivative))
        if rel:
            errors[-1] /= np.linalg.norm(directional_derivative)
        if disp:
            print(row_format.format(fd_eps[ii],
                                    np.linalg.norm(directional_derivative),
                                    np.linalg.norm(fd_directional_derivative),
                                    errors[ii]))
            # print(fd_directional_derivative,directional_derivative)

    if plot:
        plt.loglog(fd_eps, errors, 'o-')
        label = r'$\lvert\nabla^2_\epsilon \cdot p f-\nabla^2 f\cdot p\rvert$'
        plt.ylabel(label)
        plt.xlabel(r'$\epsilon$')
        plt.show()

    return np.asarray(errors)


def scipy_gauss_hermite_pts_wts_1D(nn):
    x, w = roots_hermitenorm(nn)
    w /= np.sqrt(2*np.pi)
    return x, w


def scipy_gauss_legendre_pts_wts_1D(nn):
    x, w = np.polynomial.legendre.leggauss(nn)
    w *= 0.5
    return x, w


def get_tensor_product_quadrature_rule(
        nsamples, num_vars, univariate_quadrature_rules,
        transform_samples=None, density_function=None):
    r"""
    if get error about outer product failing it may be because
    univariate_quadrature rule is returning a weights array for every level,
    i.e. l=0,...level
    """
    nsamples = np.atleast_1d(nsamples)
    if nsamples.shape[0] == 1 and num_vars > 1:
        nsamples = np.array([nsamples[0]]*num_vars, dtype=int)

    if callable(univariate_quadrature_rules):
        univariate_quadrature_rules = [univariate_quadrature_rules]*num_vars

    x_1d = []
    w_1d = []
    for ii in range(len(univariate_quadrature_rules)):
        x, w = univariate_quadrature_rules[ii](nsamples[ii])
        x_1d.append(x)
        w_1d.append(w)
    samples = cartesian_product(x_1d, 1)
    weights = outer_product(w_1d)

    if density_function is not None:
        weights *= density_function(samples)
    if transform_samples is not None:
        samples = transform_samples(samples)
    return samples, weights


def split_indices(nelems, nsplits):
    indices = np.hstack((
        np.full((nelems % nsplits), nelems//nsplits+1),
        np.full(nsplits-(nelems % nsplits), nelems//nsplits)))
    return np.hstack((0, np.cumsum(indices)))
