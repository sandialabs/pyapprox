# TODO: Place all funcitons in utilities.py that have been converted to use backend in this file. Eventually there should be no references to utiltities.py
import itertools


from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


def argsort_indices_leixographically(
    indices: Array, bkd: LinAlgMixin = NumpyLinAlgMixin
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


def hash_array(matrix: Array, bkd: LinAlgMixin = NumpyLinAlgMixin) -> int:
    if matrix.ndim != 1:
        raise ValueError("matrix must be 1D array")
    np_array = bkd.to_numpy(matrix)
    return hash(np_array.tobytes())


def unique_matrix_row_indices(
    matrix: Array, bkd: LinAlgMixin = NumpyLinAlgMixin
):
    """Return the row numbers of the unique rows in a matrix"""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D array")
    unique_row_indices = []
    unique_rows_dict = dict()
    # the row number of each row equal to the ith unique_row
    idx_per_unique_row = []
    for ii in range(matrix.shape[0]):
        key = hash_array(matrix[ii, :])
        if key not in unique_rows_dict:
            unique_rows_dict[key] = len(unique_rows_dict)
            unique_row_indices.append(ii)
            idx_per_unique_row.append([ii])
        else:
            idx_per_unique_row[unique_rows_dict[key]].append(ii)
    return bkd.asarray(unique_row_indices, dtype=int), [
        bkd.array(idx, dtype=int) for idx in idx_per_unique_row
    ]


def unique_matrix_rows(matrix) -> Array:
    """Return all the unique rows of a matrix"""
    return matrix[unique_matrix_row_indices(matrix)[0]]


def unique_matrix_columns(matrix) -> Array:
    """Return all the unique columns of a matrix"""
    return matrix[:, unique_matrix_row_indices(matrix.T)[0]]


def get_all_sample_combinations(
    samples1: Array, samples2: Array, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Array:
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
    samples = []
    for r in itertools.product(*[samples1.T, samples2.T]):
        samples.append(bkd.hstack(r))
    return bkd.stack(samples, axis=1)
