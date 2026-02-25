"""
Scope utilities for Gaussian Bayesian network operations.

These utilities handle variable ID tracking and index management
for factor operations in graphical models.
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def get_unique_variable_blocks(
    var_ids_list: List[List[int]],
    nvars_per_var_list: List[List[int]],
) -> Tuple[List[int], List[int]]:
    """
    Get the union of variable sets from multiple factors.

    Parameters
    ----------
    var_ids_list : List[List[int]]
        List of variable ID lists from each factor.
    nvars_per_var_list : List[List[int]]
        List of nvars_per_var lists from each factor.

    Returns
    -------
    unique_var_ids : List[int]
        Union of all variable IDs.
    unique_nvars_per_var : List[int]
        Dimensions for each unique variable.

    Raises
    ------
    ValueError
        If the same variable has different dimensions in different factors.
    """
    var_dims: Dict[int, int] = {}

    for var_ids, nvars_per_var in zip(var_ids_list, nvars_per_var_list):
        for var_id, nvars in zip(var_ids, nvars_per_var):
            if var_id in var_dims:
                if var_dims[var_id] != nvars:
                    raise ValueError(
                        f"Variable {var_id} has inconsistent dimensions: "
                        f"{var_dims[var_id]} vs {nvars}"
                    )
            else:
                var_dims[var_id] = nvars

    unique_var_ids = sorted(var_dims.keys())
    unique_nvars_per_var = [var_dims[var_id] for var_id in unique_var_ids]

    return unique_var_ids, unique_nvars_per_var


def expand_scope(
    precision: Array,
    shift: Array,
    var_ids: List[int],
    nvars_per_var: List[int],
    target_var_ids: List[int],
    target_nvars_per_var: List[int],
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    """
    Expand precision and shift to a larger scope.

    New variables are added with zero precision (vacuous information).

    Parameters
    ----------
    precision : Array
        Current precision matrix. Shape: (n, n)
    shift : Array
        Current shift vector. Shape: (n,)
    var_ids : List[int]
        Current variable IDs.
    nvars_per_var : List[int]
        Dimensions for current variables.
    target_var_ids : List[int]
        Target variable IDs (must include current var_ids).
    target_nvars_per_var : List[int]
        Dimensions for target variables.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    new_precision : Array
        Expanded precision matrix.
    new_shift : Array
        Expanded shift vector.
    """
    total_target_dims = sum(target_nvars_per_var)

    new_precision = bkd.zeros((total_target_dims, total_target_dims))
    new_shift = bkd.zeros((total_target_dims,))

    # Map current variables to target indices
    for i, var_id in enumerate(var_ids):
        if var_id not in target_var_ids:
            raise ValueError(f"Variable {var_id} not in target scope {target_var_ids}")

        target_idx = target_var_ids.index(var_id)
        src_start = sum(nvars_per_var[:i])
        src_end = src_start + nvars_per_var[i]
        tgt_start = sum(target_nvars_per_var[:target_idx])
        tgt_end = tgt_start + target_nvars_per_var[target_idx]

        # Copy shift
        shift_np = bkd.to_numpy(shift)
        new_shift_np = bkd.to_numpy(new_shift)
        if shift_np.ndim > 1:
            shift_np = shift_np.flatten()
        new_shift_np[tgt_start:tgt_end] = shift_np[src_start:src_end]
        new_shift = bkd.asarray(new_shift_np)

        # Copy precision blocks
        for j, var_id_j in enumerate(var_ids):
            target_idx_j = target_var_ids.index(var_id_j)
            src_start_j = sum(nvars_per_var[:j])
            src_end_j = src_start_j + nvars_per_var[j]
            tgt_start_j = sum(target_nvars_per_var[:target_idx_j])
            tgt_end_j = tgt_start_j + target_nvars_per_var[target_idx_j]

            prec_np = bkd.to_numpy(precision)
            new_prec_np = bkd.to_numpy(new_precision)
            new_prec_np[tgt_start:tgt_end, tgt_start_j:tgt_end_j] = prec_np[
                src_start:src_end, src_start_j:src_end_j
            ]
            new_precision = bkd.asarray(new_prec_np)

    return new_precision, new_shift


def get_partition_indices(
    var_ids: List[int],
    nvars_per_var: List[int],
    split_ids: Set[int],
) -> Tuple[List[int], List[int]]:
    """
    Get indices for partitioning variables into two groups.

    Parameters
    ----------
    var_ids : List[int]
        All variable IDs.
    nvars_per_var : List[int]
        Dimensions for each variable.
    split_ids : Set[int]
        Variable IDs for the first partition.

    Returns
    -------
    indices_1 : List[int]
        Indices for variables in split_ids.
    indices_2 : List[int]
        Indices for remaining variables.
    """
    indices_1: List[int] = []
    indices_2: List[int] = []

    offset = 0
    for var_id, nvars in zip(var_ids, nvars_per_var):
        var_indices = list(range(offset, offset + nvars))
        if var_id in split_ids:
            indices_1.extend(var_indices)
        else:
            indices_2.extend(var_indices)
        offset += nvars

    return indices_1, indices_2


def get_var_start_end_indices(
    var_ids: List[int],
    nvars_per_var: List[int],
    target_var_id: int,
) -> Tuple[int, int]:
    """
    Get start and end indices for a specific variable.

    Parameters
    ----------
    var_ids : List[int]
        All variable IDs.
    nvars_per_var : List[int]
        Dimensions for each variable.
    target_var_id : int
        Variable ID to find.

    Returns
    -------
    start : int
        Start index (inclusive).
    end : int
        End index (exclusive).

    Raises
    ------
    ValueError
        If target_var_id not in var_ids.
    """
    if target_var_id not in var_ids:
        raise ValueError(f"Variable {target_var_id} not in {var_ids}")

    idx = var_ids.index(target_var_id)
    start = sum(nvars_per_var[:idx])
    end = start + nvars_per_var[idx]
    return start, end


def reorder_scope(
    precision: Array,
    shift: Array,
    var_ids: List[int],
    nvars_per_var: List[int],
    new_order: List[int],
    bkd: Backend[Array],
) -> Tuple[Array, Array, List[int], List[int]]:
    """
    Reorder variables in a factor.

    Parameters
    ----------
    precision : Array
        Precision matrix.
    shift : Array
        Shift vector.
    var_ids : List[int]
        Current variable IDs.
    nvars_per_var : List[int]
        Dimensions per variable.
    new_order : List[int]
        New ordering (indices into var_ids).
    bkd : Backend[Array]
        Backend.

    Returns
    -------
    new_precision : Array
        Reordered precision.
    new_shift : Array
        Reordered shift.
    new_var_ids : List[int]
        Reordered variable IDs.
    new_nvars_per_var : List[int]
        Reordered dimensions.
    """
    if set(new_order) != set(range(len(var_ids))):
        raise ValueError("new_order must be a permutation of 0..n-1")

    # Build permutation for indices
    perm: List[int] = []
    for idx in new_order:
        start = sum(nvars_per_var[:idx])
        end = start + nvars_per_var[idx]
        perm.extend(range(start, end))

    perm_arr = np.array(perm)

    # Reorder arrays
    prec_np = bkd.to_numpy(precision)
    shift_np = bkd.to_numpy(shift)
    if shift_np.ndim > 1:
        shift_np = shift_np.flatten()

    new_prec_np = prec_np[np.ix_(perm_arr, perm_arr)]
    new_shift_np = shift_np[perm_arr]

    new_var_ids = [var_ids[i] for i in new_order]
    new_nvars_per_var = [nvars_per_var[i] for i in new_order]

    return (
        bkd.asarray(new_prec_np),
        bkd.asarray(new_shift_np),
        new_var_ids,
        new_nvars_per_var,
    )
