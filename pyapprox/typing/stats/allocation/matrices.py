"""Allocation matrix utilities for multifidelity estimation.

Allocation matrices define which models are evaluated on which sample
partitions. These utilities convert between different representations.
"""

from typing import Generic, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


def get_npartitions_from_nmodels(nmodels: int) -> int:
    """Get number of partitions for standard ACV allocation.

    For M models, the standard ACV allocation uses 2*(M-1) + 1 partitions:
    - 1 partition for HF-only samples
    - 2 partitions per LF model (one shared with HF, one for LF mean estimation)

    Parameters
    ----------
    nmodels : int
        Number of models (including high-fidelity).

    Returns
    -------
    int
        Number of partitions.

    Examples
    --------
    >>> get_npartitions_from_nmodels(2)  # 2 models: HF + 1 LF
    3
    >>> get_npartitions_from_nmodels(3)  # 3 models: HF + 2 LF
    5
    """
    return 2 * (nmodels - 1) + 1


def get_allocation_matrix_from_recursion(
    nmodels: int,
    recursion_index: np.ndarray,
    bkd: Backend[Array],
) -> Array:
    """Build allocation matrix from recursion index.

    The recursion index defines how models are coupled for control variates:
    - recursion_index[m] = k means model m+1 is coupled with model k

    Different recursion indices give different estimators:
    - MFMC: [0, 0, 0, ...] (all LF coupled with HF)
    - MLMC: [0, 1, 2, ...] (successive coupling)

    Parameters
    ----------
    nmodels : int
        Number of models.
    recursion_index : ndarray
        Recursion index. Shape: (nmodels-1,)
        recursion_index[m] in [0, m] specifies which model m+1 is coupled with.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Allocation matrix. Shape: (nmodels, npartitions)
        A[i, j] = 1 if model i is evaluated on partition j.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # MFMC with 3 models
    >>> ridx = np.array([0, 0])
    >>> A = get_allocation_matrix_from_recursion(3, ridx, bkd)
    >>> print(bkd.to_numpy(A))
    [[1. 1. 0. 1. 0.]
     [0. 1. 1. 0. 0.]
     [0. 0. 0. 1. 1.]]
    """
    if len(recursion_index) != nmodels - 1:
        raise ValueError(
            f"recursion_index must have length nmodels-1={nmodels-1}, "
            f"got {len(recursion_index)}"
        )

    # Validate recursion index values
    for m, k in enumerate(recursion_index):
        if k < 0 or k > m:
            raise ValueError(
                f"recursion_index[{m}] = {k} invalid. Must be in [0, {m}]"
            )

    npartitions = get_npartitions_from_nmodels(nmodels)
    A = np.zeros((nmodels, npartitions))

    # Partition 0: HF only
    A[0, 0] = 1

    # For each LF model m (1-indexed), assign to partitions
    for m in range(1, nmodels):
        ridx = recursion_index[m - 1]

        # Partition for shared samples (model m and its coupled model ridx)
        # Shared partition index: 2*(m-1) + 1
        shared_part = 2 * (m - 1) + 1
        A[m, shared_part] = 1
        A[ridx, shared_part] = 1

        # Partition for model m only (for estimating mu_m)
        # Model-only partition index: 2*(m-1) + 2
        only_part = 2 * (m - 1) + 2
        A[m, only_part] = 1

    return bkd.asarray(A)


def get_nsamples_per_model(
    allocation_mat: np.ndarray,
    npartition_samples: np.ndarray,
) -> np.ndarray:
    """Compute samples per model from partition allocation.

    Parameters
    ----------
    allocation_mat : ndarray
        Allocation matrix. Shape: (nmodels, npartitions)
    npartition_samples : ndarray
        Samples in each partition. Shape: (npartitions,)

    Returns
    -------
    ndarray
        Samples per model. Shape: (nmodels,)

    Examples
    --------
    >>> A = np.array([[1, 1, 0], [0, 1, 1]])
    >>> n_part = np.array([10, 20, 30])
    >>> get_nsamples_per_model(A, n_part)
    array([30, 50])
    """
    return allocation_mat @ npartition_samples


def validate_allocation_matrix(allocation_mat: np.ndarray) -> None:
    """Validate an allocation matrix.

    Parameters
    ----------
    allocation_mat : ndarray
        Allocation matrix to validate. Shape: (nmodels, npartitions)

    Raises
    ------
    ValueError
        If matrix is invalid.
    """
    if allocation_mat.ndim != 2:
        raise ValueError(
            f"Allocation matrix must be 2D, got {allocation_mat.ndim}D"
        )

    nmodels, npartitions = allocation_mat.shape

    if nmodels < 1:
        raise ValueError("Must have at least 1 model")

    if npartitions < 1:
        raise ValueError("Must have at least 1 partition")

    # Check all entries are 0 or 1
    if not np.all((allocation_mat == 0) | (allocation_mat == 1)):
        raise ValueError("Allocation matrix must contain only 0s and 1s")

    # Check HF model (row 0) has at least one partition
    if not np.any(allocation_mat[0] == 1):
        raise ValueError("HF model (row 0) must have at least one partition")


def get_recursion_index_mfmc(nmodels: int) -> np.ndarray:
    """Get MFMC recursion index (all models coupled with HF).

    Parameters
    ----------
    nmodels : int
        Number of models.

    Returns
    -------
    ndarray
        Recursion index. Shape: (nmodels-1,)
    """
    return np.zeros(nmodels - 1, dtype=np.int64)


def get_recursion_index_mlmc(nmodels: int) -> np.ndarray:
    """Get MLMC recursion index (successive coupling).

    Parameters
    ----------
    nmodels : int
        Number of models.

    Returns
    -------
    ndarray
        Recursion index. Shape: (nmodels-1,)
    """
    return np.arange(nmodels - 1, dtype=np.int64)


def allocation_matrix_to_string(allocation_mat: np.ndarray) -> str:
    """Convert allocation matrix to readable string.

    Parameters
    ----------
    allocation_mat : ndarray
        Allocation matrix. Shape: (nmodels, npartitions)

    Returns
    -------
    str
        String representation.
    """
    nmodels, npartitions = allocation_mat.shape
    lines = [f"Allocation matrix ({nmodels} models, {npartitions} partitions):"]

    # Header
    header = "       " + " ".join(f"P{j:2d}" for j in range(npartitions))
    lines.append(header)

    # Rows
    for i in range(nmodels):
        row = f"M{i:2d}:   " + "   ".join(
            "X" if allocation_mat[i, j] == 1 else "."
            for j in range(npartitions)
        )
        lines.append(row)

    return "\n".join(lines)
