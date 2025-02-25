# TODO: Place all funcitons in utilities.py that have been converted to use backend in this file

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


def argsort_indices_leixographically(indices, bkd=NumpyLinAlgMixin):
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
