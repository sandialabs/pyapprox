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


def get_allocation_matrix_gis(
    nmodels: int,
    recursion_index: Array,
    bkd: Backend[Array],
) -> Array:
    """Build GIS (Generalized Independent Samples) allocation matrix.

    GIS differs from standard ACV allocation in step 3: it uses maximum()
    to merge even columns (shared samples) with odd columns (all samples),
    creating union semantics rather than hierarchical inclusion.

    The allocation matrix has shape (nmodels, 2*nmodels) with:
    - Column 2*i: shared samples for model i
    - Column 2*i+1: all samples for model i (after merge)

    Parameters
    ----------
    nmodels : int
        Number of models.
    recursion_index : Array
        Recursion index. Shape: (nmodels-1,)
        recursion_index[m] in [0, m] specifies which model m+1 is coupled with.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Allocation matrix. Shape: (nmodels, 2*nmodels)
        A[i, j] = 1 if model i is evaluated on partition j.

    Notes
    -----
    Uses numpy internally for construction because allocation matrices are
    static structures that don't participate in gradient computation. The
    result is converted to the backend array type at the end.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> ridx = bkd.asarray([0, 0])  # MFMC-like coupling
    >>> A = get_allocation_matrix_gis(3, ridx, bkd)
    """
    # Use numpy for construction - allocation matrices are static and don't
    # need gradient tracking. Convert to backend at the end.
    ridx = bkd.to_numpy(recursion_index).astype(int)
    if len(ridx) != nmodels - 1:
        raise ValueError(
            f"recursion_index must have length nmodels-1={nmodels-1}, "
            f"got {len(ridx)}"
        )

    # Validate recursion index values
    for m, k in enumerate(ridx):
        if k < 0 or k >= nmodels:
            raise ValueError(
                f"recursion_index[{m}] = {k} invalid. Must be in [0, {nmodels-1}]"
            )

    # Build allocation matrix using numpy
    mat = np.zeros((nmodels, 2 * nmodels))

    # Step 1: Set diagonal for odd columns (all samples per model)
    for ii in range(nmodels):
        mat[ii, 2 * ii + 1] = 1

    # Step 2: Set even columns from recursion (shared samples)
    for ii in range(1, nmodels):
        mat[:, 2 * ii] = mat[:, ridx[ii - 1] * 2 + 1]

    # Step 3 (GIS-specific): Merge via maximum
    for ii in range(1, nmodels):
        mat[:, 2 * ii + 1] = np.maximum(mat[:, 2 * ii], mat[:, 2 * ii + 1])

    return bkd.asarray(mat)


def get_allocation_matrix_grd(
    nmodels: int,
    recursion_index: Array,
    bkd: Backend[Array],
) -> Array:
    """Build GRD (Generalized Recursive Difference) allocation matrix.

    GRD keeps even and odd columns separate (no merge step),
    creating disjoint sample sets.

    Parameters
    ----------
    nmodels : int
        Number of models.
    recursion_index : Array
        Recursion index. Shape: (nmodels-1,)
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Allocation matrix. Shape: (nmodels, 2*nmodels)

    Notes
    -----
    Uses numpy internally for construction because allocation matrices are
    static structures that don't participate in gradient computation. The
    result is converted to the backend array type at the end.
    """
    # Use numpy for construction - allocation matrices are static and don't
    # need gradient tracking. Convert to backend at the end.
    ridx = bkd.to_numpy(recursion_index).astype(int)
    if len(ridx) != nmodels - 1:
        raise ValueError(
            f"recursion_index must have length nmodels-1={nmodels-1}, "
            f"got {len(ridx)}"
        )

    # Validate recursion index values
    for m, k in enumerate(ridx):
        if k < 0 or k >= nmodels:
            raise ValueError(
                f"recursion_index[{m}] = {k} invalid. Must be in [0, {nmodels-1}]"
            )

    # Build allocation matrix using numpy
    mat = np.zeros((nmodels, 2 * nmodels))

    # Step 1: Set diagonal for odd columns
    for ii in range(nmodels):
        mat[ii, 2 * ii + 1] = 1

    # Step 2: Set even columns from recursion
    for ii in range(1, nmodels):
        mat[:, 2 * ii] = mat[:, ridx[ii - 1] * 2 + 1]

    # No step 3 for GRD (no merge)

    return bkd.asarray(mat)


def get_allocation_matrix_gmf(
    nmodels: int,
    recursion_index: Array,
    bkd: Backend[Array],
) -> Array:
    """Build GMF (Generalized Multifidelity) allocation matrix.

    GMF fills rows above last 1 in each column, creating hierarchical
    inclusion (L-shaped pattern per model).

    Parameters
    ----------
    nmodels : int
        Number of models.
    recursion_index : Array
        Recursion index. Shape: (nmodels-1,)
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Allocation matrix. Shape: (nmodels, 2*nmodels)

    Notes
    -----
    Uses numpy internally for construction because allocation matrices are
    static structures that don't participate in gradient computation. The
    result is converted to the backend array type at the end.
    """
    # Use numpy for construction - allocation matrices are static and don't
    # need gradient tracking. Convert to backend at the end.
    ridx = bkd.to_numpy(recursion_index).astype(int)
    if len(ridx) != nmodels - 1:
        raise ValueError(
            f"recursion_index must have length nmodels-1={nmodels-1}, "
            f"got {len(ridx)}"
        )

    # Validate recursion index values
    for m, k in enumerate(ridx):
        if k < 0 or k >= nmodels:
            raise ValueError(
                f"recursion_index[{m}] = {k} invalid. Must be in [0, {nmodels-1}]"
            )

    # Build allocation matrix using numpy
    mat = np.zeros((nmodels, 2 * nmodels))

    # Step 1: Set diagonal for odd columns
    for ii in range(nmodels):
        mat[ii, 2 * ii + 1] = 1

    # Step 2: Set even columns from recursion
    for ii in range(1, nmodels):
        mat[:, 2 * ii] = mat[:, ridx[ii - 1] * 2 + 1]

    # Step 3 (GMF-specific): Fill rows above last 1 in each column
    # This creates hierarchical inclusion (L-shaped pattern)
    for ii in range(2, 2 * nmodels):
        rows_with_one = np.where(mat[:, ii] == 1)[0]
        if len(rows_with_one) > 0:
            last_row = rows_with_one[-1]
            mat[:last_row, ii] = 1.0

    return bkd.asarray(mat)


def compute_samples_per_model(
    npartition_samples: Array,
    allocation_matrix: Array,
    bkd: Backend[Array],
) -> Array:
    """Compute total samples evaluated by each model.

    Uses formula from tutorial: n_i = Σₖ p[k] · χ[A[k, 2i] + A[k, 2i+1]]
    where χ[·] = 1 if argument > 0, else 0.

    This is NOT simple matrix multiplication A @ p, because each model's
    samples depend on whether it participates in either the starred OR
    unstarred set for each partition.

    Parameters
    ----------
    npartition_samples : Array
        Samples in each partition. Shape: (nmodels,)
    allocation_matrix : Array
        Allocation matrix. Shape: (nmodels, 2*nmodels)
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Samples per model. Shape: (nmodels,)

    Examples
    --------
    >>> # ACVMF with 4 models, p = [2, 3, 4, 5]
    >>> # Model 0: active in partition 0 only -> n_0 = 2
    >>> # Model 1: active in partitions 0, 1 -> n_1 = 2 + 3 = 5
    >>> # Model 2: active in partitions 0, 1, 2 -> n_2 = 2 + 3 + 4 = 9
    >>> # Model 3: active in all partitions -> n_3 = 2 + 3 + 4 + 5 = 14
    """
    nmodels = allocation_matrix.shape[0]
    nsamples_list = []
    for ii in range(nmodels):
        # Find partitions where model ii is active (starred OR unstarred set)
        # Column 2*ii is Z_ii* (starred), column 2*ii+1 is Z_ii (unstarred)
        col_starred = 2 * ii
        col_unstarred = 2 * ii + 1
        if col_unstarred < allocation_matrix.shape[1]:
            active = (
                (allocation_matrix[:, col_starred] == 1) |
                (allocation_matrix[:, col_unstarred] == 1)
            )
        else:
            active = (allocation_matrix[:, col_starred] == 1)
        # Sum partition samples where model is active
        nsamples_ii = bkd.sum(bkd.where(active, npartition_samples, bkd.zeros_like(npartition_samples)))
        nsamples_list.append(bkd.reshape(nsamples_ii, (1,)))
    return bkd.concatenate(nsamples_list)


def get_allocation_matrix_mfmc(nmodels: int, bkd: Backend[Array]) -> Array:
    """Build MFMC allocation matrix (upper triangular structure).

    MFMC uses nested sample sets where each model evaluates on all
    samples used by higher-fidelity models plus its own.

    Parameters
    ----------
    nmodels : int
        Number of models.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Allocation matrix. Shape: (nmodels, 2*nmodels)
        - Rows: independent partitions k = 0, ..., M-1
        - Columns: sample sets Z₀*, Z₀, Z₁*, Z₁, ...

    Examples
    --------
    >>> # For 3 models, MFMC allocation:
    >>> # Partition 0 (HF only): model 0 uses all columns except 0
    >>> # Partition 1: models 1+ use columns from 2*1+1 onward
    >>> # etc.
    """
    mat = np.zeros((nmodels, 2 * nmodels))
    # Partition 0 (HF): uses all columns from 1 onward
    mat[0, 1:] = 1
    # Partitions 1 to M-1: LF model ii uses columns from 2*ii+1 onward
    for ii in range(1, nmodels):
        mat[ii, 2 * ii + 1:] = 1
    return bkd.asarray(mat)


def get_allocation_matrix_mlmc(nmodels: int, bkd: Backend[Array]) -> Array:
    """Build MLMC allocation matrix (banded structure).

    MLMC uses a telescoping structure where adjacent levels share samples.

    Parameters
    ----------
    nmodels : int
        Number of models.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Allocation matrix. Shape: (nmodels, 2*nmodels)
        - Rows: independent partitions k = 0, ..., M-1
        - Columns: sample sets Z₀*, Z₀, Z₁*, Z₁, ...

    Examples
    --------
    >>> # For 3 models (levels), MLMC allocation:
    >>> # Partition 0: difference f_0 - f_1 -> columns 1, 2
    >>> # Partition 1: difference f_1 - f_2 -> columns 3, 4
    >>> # Partition 2: coarsest f_2 only -> column 5
    """
    mat = np.zeros((nmodels, 2 * nmodels))
    # Partitions 0 to M-2: level differences
    for ii in range(nmodels - 1):
        mat[ii, 2 * ii + 1 : 2 * ii + 3] = 1
    # Partition M-1: coarsest level only
    mat[-1, -1] = 1
    return bkd.asarray(mat)


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
