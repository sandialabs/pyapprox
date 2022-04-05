import numpy as np
from itertools import combinations, combinations_with_replacement
from scipy.special import gammaln

from pyapprox.util.utilities import cartesian_product, hash_array
from pyapprox.util.pya_numba import njit


def get_total_degree(num_dims, num_pts):
    degree = 1
    while True:
        num_terms = nchoosek(num_dims+degree, degree)
        if (num_terms >= num_pts):
            break
        degree += 1
    return degree


@njit(cache=True)
def compute_next_combination(num_vars, level, extend, h, t, index):
    if (not extend):
        t = level
        h = 0
        index[0] = level
        for dd in range(1, num_vars):
            index[dd] = 0
    else:
        if (1 < t):
            h = 0
        t = index[h]
        index[h] = 0
        index[0] = t - 1
        index[h+1] += 1
        h += 1
    extend = (index[num_vars-1] != level)
    return index, extend, h, t


def num_total_degree_indices(num_vars, degree):
    num_indices = nchoosek(num_vars + degree, num_vars)
    return num_indices


@njit(cache=True)
def compute_combinations(num_vars, level):
    if (level > 0):
        num_indices = nchoosek(num_vars + level, num_vars) -\
            nchoosek(num_vars + level-1, num_vars)
        indices = np.empty((num_vars, num_indices), dtype=np.int64)
        extend = False
        h = 0
        t = 0
        i = 0
        # important this is initialized to zero
        index = np.zeros((num_vars), dtype=np.int64)
        while (True):
            index, extend, h, t = compute_next_combination(
                num_vars, level, extend, h, t, index)
            indices[:, i] = index.copy()
            i += 1

            if (not extend):
                break
    else:
        indices = np.zeros((num_vars, 1), dtype=np.int64)

    return indices


@njit(cache=True)
def pnorm(array, p):
    return np.sum(array**p)**(1.0/float(p))


@njit(cache=True)
def compute_hyperbolic_level_subdim_indices(
        num_vars, level, num_active_vars, p):
    assert p <= 1.0 and p > 0.0
    eps = 100 * np.finfo(np.double).eps

    indices = np.empty((num_active_vars, 1000), dtype=np.int64)
    ll = num_active_vars
    num_indices = 0
    while (True):
        level_data = compute_combinations(num_active_vars, ll)
        for ii in range(level_data.shape[1]):
            index = level_data[:, ii]
            if (np.count_nonzero(index) == num_active_vars):
                p_norm = pnorm(index, p)
                if ((p_norm > level-1+eps) and (p_norm < level+eps)):
                    # resize memory if needed
                    if (num_indices >= indices.shape[1]):
                        indices = np.hstack(
                            (indices,
                             np.empty((num_active_vars, num_indices+1000),
                                      dtype=np.int64)))
                    indices[:, num_indices] = index
                    num_indices += 1
        ll += 1
        if (ll > level):
            break

    # Remove unwanted memory
    return indices[:, :num_indices]


@njit(cache=True)
def compute_hyperbolic_level_indices(num_vars, level, p):
    assert level >= 0
    if (level == 0):
        return np.zeros((num_vars, 1), dtype=np.int64)

    indices = np.eye((num_vars), dtype=np.int64)*level

    for dd in range(2, min(level+1, num_vars+1)):

        # determine the values of the nonzero entries of each index
        level_indices = compute_hyperbolic_level_subdim_indices(
            num_vars, level, dd, p)

        if (level_indices.shape[1] == 0):
            break

        # determine which variables of each index at this level
        # are nonzero
        vars_comb = compute_combinations(num_vars, dd)
        var_indices = np.empty_like(vars_comb)
        num_var_indices = 0
        for ii in range(vars_comb.shape[1]):
            index = vars_comb[:, ii]
            if (np.count_nonzero(index) == dd):
                var_indices[:, num_var_indices] = index
                num_var_indices += 1

        # Chop off unused memory;
        var_indices = var_indices[:, :num_var_indices]
        # following does not work it has uninitialized values in answer
        # var_indices.resize((num_vars, num_var_indices))

        new_indices = np.zeros((
            num_vars, var_indices.shape[1]*level_indices.shape[1]),
                               dtype=np.int64)
        num_new_indices = 0
        for ii in range(var_indices.shape[1]):
            var_index = var_indices[:, ii]
            II = np.nonzero(var_index)[0]
            # for each permutation of the nonzero entries of the
            # index at this level put in each of the possible
            # permutatinons of the level. E.g in 3D level =3
            # var_indices will be [[1,1,0],[1,0,1],[0,1,1]] and level indices
            # will be [[2,1],[1,2]]. Then all indices at this level will
            # be [[2,1,0],[2,0,1],[0,2,1],[1,2,0],[1,0,2],[0,1,2]]
            for jj in range(level_indices.shape[1]):
                index = new_indices[:, num_new_indices]
                for kk in range(II.shape[0]):
                    index[II[kk]] = level_indices[kk, jj]
                num_new_indices += 1
        indices = np.hstack((indices, new_indices))
    return indices


@njit(cache=True)
def nchoosek(nn, kk):
    val = 1
    for jj in range(min(kk, nn-kk)):
        val = (val*(nn-jj))//(jj+1)
    return val


@njit(cache=True)
def compute_hyperbolic_indices(num_vars, level, p=1):
    assert level >= 0
    indices = compute_hyperbolic_level_indices(num_vars, 0, p)
    for ll in range(1, level+1):
        level_indices = compute_hyperbolic_level_indices(num_vars, ll, p)
        indices = np.hstack((indices, level_indices))
    return indices


def compute_tensor_product_level_indices(num_vars, degree, max_norm=True):
    indices = cartesian_product([np.arange(degree+1)]*num_vars, 1)
    if max_norm:
        II = np.where(np.linalg.norm(indices, axis=0, ord=np.inf) == degree)[0]
    else:
        II = np.where(indices.sum(axis=0) == degree)[0]
    return indices[:, II]


def tensor_product_indices(degrees):
    num_vars = len(degrees)
    indices_1d = []
    for ii in range(num_vars):
        indices_1d.append(np.arange(degrees[ii]+1))
    indices = cartesian_product(indices_1d, 1)
    return indices


def set_difference(indices1, indices2):
    r"""Compute the set difference A\B.
    That is find the indices in set A that are not in set B
    This function is not symmetric, i.e. A\B != B\A
    """
    arrays_are_1d = False
    if indices1.ndim == 1:
        indices1 = indices1[np.newaxis, :]
        indices2 = indices2[np.newaxis, :]
        arrays_are_1d = True

    indices1_set = set()
    for ii in range(indices1.shape[1]):
        key = hash_array(indices1[:, ii])
        indices1_set.add(key)

    difference_idx = []
    for jj in range(indices2.shape[1]):
        key = hash_array(indices2[:, jj])
        if key not in indices1_set:
            difference_idx.append(jj)

    if not arrays_are_1d:
        return indices2[:, difference_idx]
    else:
        return indices2[0, difference_idx]


def argsort_indices_leixographically(indices):
    r"""
    Argort a set of indices lexiographically. Sort by SUM of columns then
    break ties by value of first index then use the next index to break tie
    and so on

    E.g. multiindices [(1,1),(2,0),(1,2),(0,2)] -> [(0,2),(1,1),(2,0),(1,2)]

    Parameters
    ----------
    indices: np.ndarray (num_vars,num_indices)
         multivariate indices
    Return
    ------
    sorted_idx : np.ndarray (num_indices)
        The array indices of the sorted polynomial indices
    """
    tuple_indices = []
    for ii in range(indices.shape[1]):
        tuple_index = tuple(indices[:, ii])
        tuple_indices.append(tuple_index)
    sorted_idx = sorted(
        list(range(len(tuple_indices))),
        key=lambda x: (sum(tuple_indices[x]), tuple_indices[x]))
    return np.asarray(sorted_idx)


def argsort_indices_lexiographically_by_row(indices):
    r"""
    Argort a set of indices lexiographically.  Sort by sum of columns by
    value of first row. Break ties by value of first row then use the next
    row to break tie and so on

    E.g. multiindices [(1,1),(2,0),(1,2),(0,2)] -> [(0,2),(1,1),(1,2),(2,0)]

    Parameters
    ----------
    indices: np.ndarray (num_vars,num_indices)
         multivariate indices
    Return
    ------
    sorted_idx : np.ndarray (num_indices)
        The array indices of the sorted polynomial indices
    """
    tuple_indices = []
    for ii in range(indices.shape[1]):
        tuple_index = tuple(indices[:, ii])
        tuple_indices.append(tuple_index)
    sorted_idx = sorted(
        list(range(len(tuple_indices))),
        key=lambda x: tuple_indices[x])
    return np.asarray(sorted_idx)


def sort_indices_lexiographically(indices):
    r"""
    Sort by level then lexiographically
    The last key in the sequence is used for the primary sort order,
    the second-to-last key for the secondary sort order, and so on
    """
    index_tuple = (indices[0, :],)
    for ii in range(1, indices.shape[0]):
        index_tuple = index_tuple+(indices[ii, :],)
    index_tuple = index_tuple+(indices.sum(axis=0),)
    II = np.lexsort(index_tuple)
    return indices[:, II]


def get_maximal_indices(indices, indices_dict=None):
    r"""
    Get the maximal indices of a set of multivariate indices.

    An index is maximal if all of its forward neighbours are not present
    in indices.
    """
    if indices_dict is None:
        indices_dict = dict()
        for ii in range(indices.shape[1]):
            key = hash_array(indices[:, ii])
            indices_dict[key] = ii
    else:
        assert len(indices_dict) == indices.shape[1]

    num_vars = indices.shape[0]
    maximal_indices = []
    for ii in range(indices.shape[1]):
        index = indices[:, ii]
        forward_index = index.copy()
        maximal_index = True
        for jj in range(num_vars):
            forward_index[jj] += 1
            key = hash_array(forward_index)
            forward_index[jj] -= 1
            if key in indices_dict:
                maximal_index = False
                break
        if maximal_index:
            maximal_indices.append(ii)
    return indices[:, maximal_indices]


def get_backward_neighbor(subspace_index, var_num):
    neighbor = subspace_index.copy()
    neighbor[var_num] -= 1
    return neighbor


def get_forward_neighbor(subspace_index, var_num):
    neighbor = subspace_index.copy()
    neighbor[var_num] += 1
    return neighbor


def compute_downward_closed_indices(num_vars, admissibility_criteria):
    indices = np.zeros((num_vars, 0), dtype=np.int64)
    active_indices = np.zeros((num_vars, 1), dtype=np.int64)
    while active_indices.size > 0:
        new_indices = []
        new_indices_set = set()
        for ii in range(active_indices.shape[1]):
            active_index = active_indices[:, ii]
            neighbour = active_index.copy()
            for dd in range(num_vars):
                neighbour[dd] += 1
                if admissibility_criteria(neighbour):
                    key = hash_array(neighbour)
                    if key not in new_indices_set:
                        new_indices.append(neighbour.copy())
                        new_indices_set.add(key)
                neighbour[dd] -= 1
        indices = np.hstack((indices, active_indices))
        active_indices = np.asarray(new_indices).T
    return indices


def total_degree_admissibility_criteria(degree, index):
    return index.sum() <= degree


def pnorm_admissibility_criteria(degree, p, index):
    eps = 100 * np.finfo(float).eps
    p_norm = pnorm(index, p)
    return (p_norm < degree+eps)


def anisotropic_admissibility_criteria(
        anisotropic_weights, min_weight, degree, index):
    return index.dot(anisotropic_weights) <= min_weight*degree


def compute_anisotropic_indices(num_vars, level, anisotropic_weights):
    min_weight = np.asarray(anisotropic_weights).min()
    subspace_indices = compute_hyperbolic_indices(num_vars, level, 1.0)
    indices_to_keep = []
    for ii in range(subspace_indices.shape[1]):
        index = subspace_indices[:, ii]
        if index.dot(anisotropic_weights) <= min_weight*level:
            indices_to_keep.append(ii)

    subspace_indices = subspace_indices[:, indices_to_keep]
    return subspace_indices


def get_upper_triangular_matrix_scalar_index(ii, jj, nn):
    r"""
    Get the scalar index kk of the (ii,jj) etnry of an upper triangular matrix
    (excluding diagonal) stored in a 1D array

    For example
    .. math::

       \begin{bmatrix}
       0 & a_0 & a_1 & a_2\\
       0 & 0   & a_3 & a_4\\
       0 & 0   & 0   & a_5\\
       0 & 0   & 0   & 0
       \end{bmatrix}

    is stored as

    .. math::  \begin{bmatrix} [a_0,a_1,a_2,a_3,a_4,a_5]\end{bmatrix}
    """
    assert ii < jj
    kk = (nn*(nn-1)//2) - (nn-ii)*((nn-ii)-1)//2 + jj - ii - 1
    return int(kk)


def get_upper_triangular_matrix_indices(kk, nn):
    r"""
    Get the index tuple (ii,jj) entry kk of an upper triangular matrix
    (excluding diagonal) stored in a 1D array

    For example
    .. math::

       \begin{bmatrix}
       0 & a_0 & a_1 & a_2\\
       0 & 0   & a_3 & a_4\\
       0 & 0   & 0   & a_5\\
       0 & 0   & 0   & 0
       \end{bmatrix}

    is stored as

    .. math::  \begin{bmatrix} [a_0,a_1,a_2,a_3,a_4,a_5]\end{bmatrix}
    """
    ii = nn - 2 - np.floor(np.sqrt(-8*kk + 4*nn*(nn-1)-7)/2.0 - 0.5)
    jj = kk + ii + 1 - nn*(nn-1)/2 + (nn-ii)*((nn-ii)-1)/2
    return int(ii), int(jj)


def compute_anova_level_indices(num_vars, degree):
    if degree > num_vars:
        raise ValueError(f"degree {degree }is larger than num_vars {num_vars}")
    return list(combinations(np.arange(num_vars), degree))


def compute_anova_indices(num_vars, degree):
    indices = []
    for dd in range(degree+1):
        indices += compute_anova_level_indices(num_vars, dd)
    return indices


def unique_values_per_row(a):
    N = a.max()+1
    a_offs = a + np.arange(a.shape[0])[:, None]*N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1, N)


def compute_hyperbolic_level_indices_itertools(num_vars, degree, p):
    eps = 100 * np.finfo(np.double).eps
    if degree == 0:
        return np.zeros((num_vars, 1))
    tmp = np.array(
        list(combinations_with_replacement(np.arange(num_vars), degree)))
    # count number of times each element appears in tmp1
    indices = unique_values_per_row(tmp).T
    p_norms = np.sum(indices**p, axis=0)**(1.0/p)
    II = np.where(p_norms <= degree+eps)[0]
    return indices[:, II]


def compute_hyperbolic_indices_itertools(num_vars, degree, p):
    indices = np.empty((num_vars, 0), dtype=int)
    for dd in range(degree+1):
        new_indices = compute_hyperbolic_level_indices_itertools(
            num_vars, dd, p)
        indices = np.hstack((indices, new_indices))
    return indices


def total_degree_space_dimension(dimension, degree):
    r"""
    Return the number of basis functions in a total degree polynomial space,
    i.e. the space of all polynomials with degree at most degree.

    Parameters
    ----------
    num_vars : integer
        The number of variables of the polynomials

    degree :
        The degree of the total-degree space

    Returns
    -------
    num_terms : integer
        The number of basis functions in the total degree space

    Notes
    -----
    Note

    .. math:: {n \choose k} = frac{\Gamma(n+k+1)}{\Gamma(k+1)\Gamma{n-k+1}}, \qquad \Gamma(m)=(m-1)!

    So for dimension :math:`d` and degree :math:`p` number of terms in
    subspace is

    .. math:: {d+p \choose p} = frac{\Gamma(d+p+1)}{\Gamma(p+1)\Gamma{d+p-p+1}}, \qquad \Gamma(m)=(m-1)!

    """
    # return nchoosek(dimension+degree, degree)
    # Following more robust for large values
    return int(np.round(
        np.exp(gammaln(degree+dimension+1) - gammaln(degree+1) - gammaln(
            dimension+1))))


def total_degree_subspace_dimension(dimension, degree):
    r"""
    Return the number of basis functions in a total degree polynomial space,
    with degree equal to degree.

    Parameters
    ----------
    num_vars : integer
        The number of variables of the polynomials

    degree :
        The degree of the total-degree space

    Returns
    -------
    num_terms : integer
        The number of basis functions in the total degree space of a given
        degree
    """
    # subspace_dimension = nchoosek(nvars+degree-1, degree)
    # Following more robust for large values
    subspace_dimension = int(
        np.round(np.exp(gammaln(degree+dimension) - gammaln(degree+1) -
                        gammaln(dimension))))
    return subspace_dimension


def total_degree_encompassing_N(dimension, N):
    r"""
    Returns the smallest integer k such that the dimension of the total
    degree-k space is greater than N.
    """

    k = 0
    while total_degree_subspace_dimension(dimension, k) < N:
        k += 1
    return k


def total_degree_barrier_indices(dimension, max_degree):
    r"""
    Returns linear indices that bound total degree spaces

    Parameters
    ----------
    dimension: int
        Parametric dimension
    max_degree: int
        Maximum polynomial degree

    Returns
    -------
    degree_barrier_indices: list
        List of degree barrier indices up to (including) max_degree.
    """
    degree_barrier_indices = [0]

    for degree in range(1, max_degree+1):
        degree_barrier_indices.append(
            total_degree_subspace_dimension(dimension, degree))

    return degree_barrier_indices


def total_degree_orthogonal_transformation(coefficients, d):
    r"""
    Returns an orthogonal matrix transformation that "matches" the input
    coefficients.

    Parameters
    ----------
    coefficients: np.ndarray
        Length-N vector of expansion coefficients
    d: int
        Parametric dimension

    Returns
    -------
    Q: np.ndarray
        A size N x N orthogonal matrix transformation. The first column
        is a unit vector in the direction of coefficients.
    """

    from scipy.linalg import qr

    N = coefficients.size

    degree_barrier_indices = [1]
    max_degree = 0
    while degree_barrier_indices[-1] < N-1:
        max_degree += 1
        degree_barrier_indices.append(
            total_degree_subspace_dimension(d, max_degree))

    q = np.zeros([N, N])

    # Assume degree = 0 is just constant
    q[0, 0] = 1.

    for degree in range(1, max_degree+1):
        i1 = degree_barrier_indices[degree-1]
        i2 = degree_barrier_indices[degree]

        M = i2-i1
        q[i1:i2, i1:i2] = qr(coefficients[i1:i2].reshape([M, 1]))[0]

    return q
