"""Cartesian product utilities for tensor product computations.

This module provides backend-agnostic functions for computing cartesian
products of arrays, which are fundamental operations for tensor product
interpolation, quadrature, and sparse grids.

Functions
---------
cartesian_product_indices(dims, bkd)
    Generate multi-indices for a full tensor product grid.
cartesian_product_samples(samples_1d, bkd)
    Build tensor product of 1D sample locations.
outer_product_weights(weights_1d, bkd)
    Compute tensor product of 1D quadrature weights.
"""

from typing import List

from pyapprox.typing.util.backends.protocols import Array, Backend


def cartesian_product_indices(dims: List[int], bkd: Backend[Array]) -> Array:
    """Generate multi-indices for a full tensor product grid.

    Creates an array where each column represents a multi-index identifying
    a point in the tensor product grid. The ordering has the last dimension
    varying fastest (C-order / row-major).

    Parameters
    ----------
    dims : List[int]
        Number of points in each dimension.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Multi-indices with shape (nvars, prod(dims)).
        Column i contains the 1D indices for grid point i.

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> indices = cartesian_product_indices([2, 3], bkd)
    >>> indices
    array([[0, 0, 0, 1, 1, 1],
           [0, 1, 2, 0, 1, 2]])

    Notes
    -----
    The ordering convention (last dimension varies fastest) matches NumPy's
    default array ordering and is consistent with sparse grid conventions.
    """
    nvars = len(dims)
    total = 1
    for n in dims:
        total *= n

    indices = bkd.zeros((nvars, total), dtype=bkd.int64_dtype())

    # Last dimension varies fastest (C-order)
    repeat_inner = 1
    for dim in range(nvars - 1, -1, -1):
        npts = dims[dim]
        repeat_outer = total // (npts * repeat_inner)

        col = 0
        for _ in range(repeat_outer):
            for pt_idx in range(npts):
                for _ in range(repeat_inner):
                    indices[dim, col] = pt_idx
                    col += 1
        repeat_inner *= npts

    return indices


def cartesian_product_samples(
    samples_1d: List[Array], bkd: Backend[Array]
) -> Array:
    """Build tensor product of 1D sample locations.

    Creates a 2D array where each column represents a point in the tensor
    product grid. The ordering has the last dimension varying fastest.

    Parameters
    ----------
    samples_1d : List[Array]
        1D sample arrays for each dimension. Each array should be 1D with
        shape (npts,) or 2D with shape (1, npts).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Tensor product samples with shape (nvars, total_samples) where
        total_samples = prod(npts_1d).

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> x = bkd.asarray([0.0, 1.0])
    >>> y = bkd.asarray([0.0, 0.5, 1.0])
    >>> samples = cartesian_product_samples([x, y], bkd)
    >>> samples
    array([[0., 0., 0., 1., 1., 1.],
           [0., 0.5, 1., 0., 0.5, 1.]])

    Notes
    -----
    The ordering convention matches `cartesian_product_indices`.
    """
    nvars = len(samples_1d)

    # Get dimensions from 1D sample arrays
    dims = []
    for samples in samples_1d:
        flat = samples.flatten()
        dims.append(flat.shape[0])

    total = 1
    for n in dims:
        total *= n

    result = bkd.zeros((nvars, total))

    repeat_inner = 1
    for dim in range(nvars - 1, -1, -1):
        npts = dims[dim]
        samples_flat = samples_1d[dim].flatten()
        repeat_outer = total // (npts * repeat_inner)

        idx = 0
        for _ in range(repeat_outer):
            for pt_idx in range(npts):
                for _ in range(repeat_inner):
                    result[dim, idx] = samples_flat[pt_idx]
                    idx += 1
        repeat_inner *= npts

    return result


def outer_product_weights(
    weights_1d: List[Array], bkd: Backend[Array]
) -> Array:
    """Compute tensor product of 1D quadrature weights.

    Creates a 1D array containing the products of 1D weights at each point
    in the tensor product grid. This is the vectorized version of computing
    w_i = prod_d w_d[i_d] for each multi-index i.

    Parameters
    ----------
    weights_1d : List[Array]
        1D weight arrays for each dimension. Each array can be 1D with
        shape (npts,) or 2D with shape (npts, 1) or (1, npts).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Tensor product weights with shape (total_samples,) where
        total_samples = prod(npts_1d).

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> wx = bkd.asarray([0.5, 0.5])  # 2-point rule
    >>> wy = bkd.asarray([1/3, 1/3, 1/3])  # 3-point rule
    >>> weights = outer_product_weights([wx, wy], bkd)
    >>> weights.shape
    (6,)
    >>> bkd.sum(weights)  # Should equal sum(wx) * sum(wy) = 1.0 * 1.0
    1.0

    Notes
    -----
    The ordering convention matches `cartesian_product_indices` and
    `cartesian_product_samples`.
    """
    nvars = len(weights_1d)

    # Get dimensions and flatten weights
    dims = []
    flat_weights = []
    for weights in weights_1d:
        flat = weights.flatten()
        dims.append(flat.shape[0])
        flat_weights.append(flat)

    total = 1
    for n in dims:
        total *= n

    result = bkd.ones((total,))

    repeat_inner = 1
    for dim in range(nvars - 1, -1, -1):
        npts = dims[dim]
        w = flat_weights[dim]
        repeat_outer = total // (npts * repeat_inner)

        idx = 0
        for _ in range(repeat_outer):
            for pt_idx in range(npts):
                for _ in range(repeat_inner):
                    result[idx] = result[idx] * w[pt_idx]
                    idx += 1
        repeat_inner *= npts

    return result
