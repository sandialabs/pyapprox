"""Utility functions for GroupACV estimators.

This module provides helper functions for model subset generation,
allocation matrix construction, and covariance block computation.
"""

from itertools import combinations
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.statest.statistics import MultiOutputStatistic


def get_model_subsets(
    nmodels: int, bkd: Backend[Array], max_subset_nmodels: Optional[int] = None
) -> List[Array]:
    """
    Generate all model subsets up to a maximum size.

    Parameters
    ----------
    nmodels : int
        The number of models

    bkd : Backend[Array]
        The backend for array operations

    max_subset_nmodels : int, optional
        The maximum number of models in a subset. Defaults to nmodels.

    Returns
    -------
    List[Array]
        List of arrays, each containing model indices for a subset
    """
    if max_subset_nmodels is None:
        max_subset_nmodels = nmodels
    assert max_subset_nmodels > 0
    assert max_subset_nmodels <= nmodels
    subsets = []
    model_indices = bkd.arange(nmodels)
    for nsubset_lfmodels in range(1, max_subset_nmodels + 1):
        for subset_indices in combinations(model_indices, nsubset_lfmodels):
            idx = bkd.asarray(subset_indices, dtype=int)
            subsets.append(idx)
    return subsets


def get_model_subsets_limited(
    nmodels: int,
    bkd: Backend[Array],
    max_group_size: Optional[int] = None,
    min_group_size: int = 1,
    require_model_0: bool = True,
) -> List[Array]:
    """Generate candidate subsets with size constraints.

    Parameters
    ----------
    nmodels : int
        Number of models in the ensemble.
    bkd : Backend
        Backend for array construction.
    max_group_size : int, optional
        Maximum subset size. If None, no upper limit.
    min_group_size : int, optional
        Minimum subset size. Default 1.
    require_model_0 : bool, optional
        If True, only return subsets containing model 0.
        Default True.

    Returns
    -------
    List[Array]
        Filtered list of candidate subsets.
    """
    all_subsets = get_model_subsets(nmodels, bkd)
    filtered = []
    for s in all_subsets:
        size = int(s.shape[0])
        if size < min_group_size:
            continue
        if max_group_size is not None and size > max_group_size:
            continue
        if require_model_0 and 0 not in bkd.to_numpy(s):
            continue
        filtered.append(s)
    return filtered


def _get_allocation_matrix_is(subsets: List[Array], bkd: Backend[Array]) -> Array:
    """
    Get allocation matrix for independent sampling.

    Each subset gets its own independent partition.

    Parameters
    ----------
    subsets : List[Array]
        List of model subsets

    bkd : Backend[Array]
        The backend for array operations

    Returns
    -------
    Array
        Allocation matrix of shape (nsubsets, npartitions)
    """
    nsubsets = len(subsets)
    npartitions = nsubsets
    allocation_mat = bkd.full((nsubsets, npartitions), 0.0, dtype=bkd.double_dtype())
    for ii, subset in enumerate(subsets):
        allocation_mat[ii, ii] = 1.0
    return allocation_mat


def _get_allocation_matrix_nested(subsets: List[Array], bkd: Backend[Array]) -> Array:
    """
    Get allocation matrix for nested sampling.

    Partitions are nested according to the order of subsets.

    Parameters
    ----------
    subsets : List[Array]
        List of model subsets

    bkd : Backend[Array]
        The backend for array operations

    Returns
    -------
    Array
        Allocation matrix of shape (nsubsets, npartitions)
    """
    nsubsets = len(subsets)
    npartitions = nsubsets
    allocation_mat = bkd.full((nsubsets, npartitions), 0.0, dtype=bkd.double_dtype())
    for ii, subset in enumerate(subsets):
        allocation_mat[ii, : ii + 1] = 1.0
    return allocation_mat


def _nest_subsets(
    subsets: List[Array], nmodels: int, bkd: Backend[Array]
) -> Tuple[List[Array], Array]:
    """
    Reorder subsets for nested sampling configuration.

    Parameters
    ----------
    subsets : List[Array]
        List of model subsets

    nmodels : int
        Number of models

    bkd : Backend[Array]
        The backend for array operations

    Returns
    -------
    tuple
        (reordered_subsets, reorder_indices)
    """
    for subset in subsets:
        if np.allclose(bkd.to_numpy(subset), [0]):
            raise ValueError("Cannot use subset [0]")
    idx = sorted(
        list(range(len(subsets))),
        key=lambda ii: (len(subsets[ii]), tuple(nmodels - subsets[ii])),
        reverse=True,
    )
    return [subsets[ii] for ii in idx], bkd.array(idx)


def _grouped_acv_sigma_block(
    subset0: Array,
    subset1: Array,
    nsamples_intersect: "int | Array",
    nsamples_subset0: "int | Array",
    nsamples_subset1: "int | Array",
    stat: "MultiOutputStatistic[Array]",
) -> Array:
    """
    Compute a single block of the grouped ACV covariance matrix.

    Parameters
    ----------
    subset0 : Array
        First subset indices

    subset1 : Array
        Second subset indices

    nsamples_intersect : int
        Number of samples in the intersection of the two subsets

    nsamples_subset0 : int
        Number of samples in subset0

    nsamples_subset1 : int
        Number of samples in subset1

    stat : MultiOutputStatistic
        The statistic object with covariance information

    Returns
    -------
    Array
        Covariance block of shape (len(subset0), len(subset1))
    """
    nsubset0 = len(subset0)
    nsubset1 = len(subset1)
    zero_block = stat.bkd().full((nsubset0, nsubset1), 0.0)
    if (nsamples_subset0 * nsamples_subset1) == 0:
        return zero_block
    threshold = stat.continuous_dead_threshold()
    if threshold > 0.0:
        if float(nsamples_intersect) < threshold:
            return zero_block
    block = stat._group_acv_sigma_block(
        subset0,
        subset1,
        nsamples_intersect,
        nsamples_subset0,
        nsamples_subset1,
    )
    return block


def _grouped_acv_sigma(
    nmodels: int,
    nsamples_intersect: Array,
    subsets: List[Array],
    stat: "MultiOutputStatistic[Array]",
) -> List[List[Array]]:
    """
    Compute the full grouped ACV covariance matrix as nested lists of blocks.

    Parameters
    ----------
    nmodels : int
        Number of models

    nsamples_intersect : Array
        Matrix of intersection sample counts between subsets

    subsets : List[Array]
        List of model subsets

    stat : MultiOutputStatistic
        The statistic object with covariance information

    Returns
    -------
    List[List[Array]]
        Nested list of covariance blocks, Sigma[i][j] is the (i,j) block
    """
    bkd = stat.bkd()
    nsubsets = len(subsets)
    Sigma: List[List[Array]] = [
        [bkd.zeros((0,)) for jj in range(nsubsets)] for ii in range(nsubsets)
    ]
    for ii, subset0 in enumerate(subsets):
        N_ii = nsamples_intersect[ii, ii]
        Sigma[ii][ii] = _grouped_acv_sigma_block(
            subset0, subset0, N_ii, N_ii, N_ii, stat
        )
        for jj, subset1 in enumerate(subsets[:ii]):
            N_jj = nsamples_intersect[jj, jj]
            Sigma[ii][jj] = _grouped_acv_sigma_block(
                subset0, subset1,
                nsamples_intersect[ii, jj], N_ii, N_jj, stat,
            )
            Sigma[jj][ii] = Sigma[ii][jj].T
    return Sigma
