import itertools
from typing import List, Iterable, Tuple, Type

import numpy as np
from scipy.special import roots_hermitenorm  # type: ignore

# from pyapprox.util.pya_numba import njit
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin


def argsort_indices_leixographically(
    indices: Array, bkd: Type[BackendMixin] = NumpyMixin
) -> Array:
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
        key=lambda x: (sum(tuple_indices[x]), tuple_indices[x]),
    )
    return bkd.array(sorted_idx, dtype=int)


def hash_array(matrix: Array, bkd: Type[BackendMixin] = NumpyMixin) -> int:
    if matrix.ndim != 1:
        raise ValueError("matrix must be 1D array")
    np_array = bkd.to_numpy(matrix)
    return hash(np_array.tobytes())


def unique_matrix_row_indices(
    matrix: Array, bkd: Type[BackendMixin] = NumpyMixin
):
    """Return the row numbers of the unique rows in a matrix"""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D array")
    unique_row_indices = []
    unique_rows_dict = dict()
    # the row number of each row equal to the ith unique_row
    idx_per_unique_row = []
    for ii in range(matrix.shape[0]):
        key = hash_array(matrix[ii, :], bkd)
        if key not in unique_rows_dict:
            unique_rows_dict[key] = len(unique_rows_dict)
            unique_row_indices.append(ii)
            idx_per_unique_row.append([ii])
        else:
            idx_per_unique_row[unique_rows_dict[key]].append(ii)
    return bkd.asarray(unique_row_indices, dtype=int), [
        bkd.array(idx, dtype=int) for idx in idx_per_unique_row
    ]


def unique_matrix_rows(matrix, bkd: Type[BackendMixin] = NumpyMixin) -> Array:
    """Return all the unique rows of a matrix"""
    return matrix[unique_matrix_row_indices(matrix, bkd)[0]]


def unique_matrix_columns(
    matrix, bkd: Type[BackendMixin] = NumpyMixin
) -> Array:
    """Return all the unique columns of a matrix"""
    return matrix[:, unique_matrix_row_indices(matrix.T, bkd)[0]]


def get_all_sample_combinations(
    samples1: Array, samples2: Array, bkd: Type[BackendMixin] = NumpyMixin
) -> Array:
    r"""
    For two sample sets of different random variables
    loop over all combinations

    samples1 vary slowest and samples2 vary fastest
    """
    samples = []
    for r in itertools.product(*[samples1.T, samples2.T]):
        samples.append(bkd.hstack(r))
    return bkd.stack(samples, axis=1)


def approx_jacobian_3D(
    f,
    x0,
    epsilon=np.sqrt(np.finfo(float).eps),
    bkd: Type[BackendMixin] = NumpyMixin,
):
    fval = f(x0)
    jacobian = bkd.full((fval.shape[0], fval.shape[1], x0.shape[0]), 0.0)
    for ii in range(len(x0)):
        dx = bkd.full((x0.shape[0],), 0.0)
        dx = bkd.up(dx, ii, epsilon)
        fval_perturbed = f(x0 + dx)
        jacobian = bkd.up(
            jacobian, ii, (fval_perturbed - fval) / epsilon, axis=-1
        )
    return jacobian


def split_indices(
    nelems: int, nsplits: int, bkd: Type[BackendMixin] = NumpyMixin
) -> Array:
    indices = bkd.hstack(
        (
            bkd.full((nelems % nsplits,), nelems // nsplits + 1, dtype=int),
            bkd.full(
                (nsplits - (nelems % nsplits),), nelems // nsplits, dtype=int
            ),
        )
    )
    return bkd.hstack((bkd.zeros((1,), dtype=int), bkd.cumsum(indices)))


def sublist(mylist: List, indices: Iterable) -> List:
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


def unique_elements_from_2D_list(list_2d: List[List]) -> List:
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


def flatten_2D_list(list_2d: List[List]) -> List:
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


def covariance_to_correlation(
    cov: Array, bkd: Type[BackendMixin] = NumpyMixin
) -> Array:
    r"""
    Compute the correlation matrix from a covariance matrix

    Parameters
    ----------
    cov : Array (nrows,nrows)
        The symetric covariance matrix

    Returns
    -------
    cor : Array (nrows,nrows)
        The symetric correlation matrix

    Examples
    --------
    >>> cov = np.asarray([[2,-1],[-1,2]])
    >>> covariance_to_correlation(cov)
    array([[ 1. , -0.5],
           [-0.5,  1. ]])
    """
    stdev_inv = 1 / bkd.sqrt(bkd.diag(cov))
    cor = stdev_inv[None, :] * cov * stdev_inv[:, None]
    return cor


def correlation_to_covariance(corr: Array, stdevs: Array) -> Array:
    return stdevs[None, :] * corr * stdevs[:, None]


# if njit used covarage.py does not pick up covarage of code
# @njit(cache=True)
def get_first_n_primes(n):
    primes = list()
    primes.append(2)
    num = 3
    while len(primes) < n:
        # np.all does not work with numba
        # if np.all([num % i != 0 for i in range(2, int(num**.5) + 1)]):
        flag = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                flag = False
                break
        if flag is True:
            primes.append(num)
        num += 2
    return np.asarray(primes)


def all_primes_less_than_or_equal_to_n(n: int) -> List:
    primes = []
    primes.append(2)
    for num in range(3, n + 1, 2):
        if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
            primes.append(num)
    return primes


def lists_of_arrays_equal(list1, list2, bkd: Type[BackendMixin] = NumpyMixin):
    if len(list1) != len(list2):
        return False
    for ll in range(len(list1)):
        if list1[ll].shape != list2[ll].shape or not bkd.allclose(
            list1[ll], list2[ll]
        ):
            return False
    return True


def scipy_gauss_hermite_pts_wts_1D(nn, bkd: Type[BackendMixin] = NumpyMixin):
    x, w = roots_hermitenorm(nn)
    w /= np.sqrt(2 * np.pi)
    return bkd.asarray(x), bkd.asarray(w)


def scipy_gauss_legendre_pts_wts_1D(nn, bkd: Type[BackendMixin] = NumpyMixin):
    x, w = np.polynomial.legendre.leggauss(nn)
    w *= 0.5
    return bkd.asarray(x), bkd.asarray(w)


def simpsons_rule(
    bounds: Tuple, nquad: int, bkd: Type[BackendMixin]
) -> Tuple[Array, Array]:
    if nquad % 2 == 0:
        raise ValueError("nquad must be odd")
    xx = bkd.linspace(*bounds, nquad)
    dx = xx[1] - xx[0]
    ww = bkd.full(xx.shape, dx / 3.0)
    ww[1:-1:2] *= 4
    ww[2:-1:2] *= 2
    return xx, ww


def trapezoid_rule(
    bounds: Tuple, nquad: int, bkd: Type[BackendMixin]
) -> Tuple[Array, Array]:
    if nquad % 2 == 0:
        raise ValueError("nquad must be odd")
    xx = bkd.linspace(*bounds, nquad)
    dx = xx[1] - xx[0]
    ww = bkd.full(xx.shape, dx)
    ww[0] /= 2
    ww[-1] /= 2
    return xx, ww


def composite_gauss_legendre_rule(
    bounds: Tuple,
    nquad_per_interval: int,
    nintervals: int,
    bkd: Type[BackendMixin],
) -> Tuple[Array, Array]:
    quadx, quadw = np.polynomial.legendre.leggauss(nquad_per_interval)
    quadx_01 = bkd.asarray((quadx + 1) / 2)
    quadw_01 = bkd.asarray(quadw / 2)
    intervals = bkd.linspace(*bounds, nintervals + 1)
    quadx_list = []
    quadw_list = []
    for ii in range(nintervals):
        a, b = intervals[ii : ii + 2]
        quadx = quadx_01 * (b - a) + a
        quadw = quadw_01 * (b - a)
        quadx_list.append(quadx)
        quadw_list.append(quadw)
    return bkd.hstack(quadx_list), bkd.hstack(quadw_list)


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
