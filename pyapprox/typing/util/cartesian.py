"""Cartesian product utilities for tensor product computations.

This module provides backend-agnostic functions for computing cartesian
products of arrays, which are fundamental operations for tensor product
interpolation, quadrature, and sparse grids.

Functions
---------
cartesian_product(bkd, vectors)
    Generic cartesian product of 1D vectors using meshgrid.
outer_product(bkd, vectors)
    Generic outer product of 1D vectors using einsum.
cartesian_product_indices(dims, bkd)
    Generate multi-indices for a full tensor product grid.
cartesian_product_samples(samples_1d, bkd)
    Build tensor product of 1D sample locations.
outer_product_weights(weights_1d, bkd)
    Compute tensor product of 1D quadrature weights.
"""

from typing import List

from pyapprox.typing.util.backends.protocols import Array, Backend


def cartesian_product(bkd: Backend[Array], vectors: List[Array]) -> Array:
    """Compute cartesian product of 1D vectors using meshgrid.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend (numpy or torch).
    vectors : List[Array]
        List of 1D arrays (at least 2 required).

    Returns
    -------
    Array
        Cartesian product with shape (ndim, prod(lens)).
        Last vector in input list varies fastest (C-order).

    Raises
    ------
    ValueError
        If fewer than 2 vectors provided.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> x = bkd.asarray([0, 1])
    >>> y = bkd.asarray([0, 1, 2])
    >>> cartesian_product(bkd, [x, y])
    array([[0, 0, 0, 1, 1, 1],
           [0, 1, 2, 0, 1, 2]])
    """
    if len(vectors) < 2:
        raise ValueError(
            f"cartesian_product requires at least 2 vectors, got {len(vectors)}"
        )

    # Flatten all vectors to 1D
    flat_vectors = [bkd.ravel(v) for v in vectors]

    # Use meshgrid with indexing='ij' for C-order (last dim varies fastest)
    grids = bkd.meshgrid(*flat_vectors, indexing="ij")

    # Stack raveled grids: each grid flattened becomes a row
    # Result shape: (ndim, prod(lens))
    stacked = bkd.stack([bkd.ravel(g) for g in grids], axis=0)
    return stacked


def outer_product(bkd: Backend[Array], vectors: List[Array]) -> Array:
    """Compute outer product of 1D vectors using einsum.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend (numpy or torch).
    vectors : List[Array]
        List of 1D arrays (at least 2 required, max 26).

    Returns
    -------
    Array
        Outer product with shape (len(v1), len(v2), ...).

    Raises
    ------
    ValueError
        If fewer than 2 or more than 26 vectors provided.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> x = bkd.asarray([1.0, 2.0])
    >>> y = bkd.asarray([3.0, 4.0, 5.0])
    >>> outer_product(bkd, [x, y])
    array([[ 3.,  4.,  5.],
           [ 6.,  8., 10.]])
    """
    if len(vectors) < 2:
        raise ValueError(
            f"outer_product requires at least 2 vectors, got {len(vectors)}"
        )
    if len(vectors) > 26:
        raise ValueError(
            f"outer_product supports at most 26 vectors, got {len(vectors)}"
        )

    # Build einsum subscripts: 'a,b,c->abc'
    letters = "abcdefghijklmnopqrstuvwxyz"
    inputs = ",".join(letters[i] for i in range(len(vectors)))
    output = letters[: len(vectors)]
    subscripts = f"{inputs}->{output}"

    flat_vectors = [bkd.ravel(v) for v in vectors]
    return bkd.einsum(subscripts, *flat_vectors)


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
    if len(dims) < 2:
        # Handle 1D case directly
        if len(dims) == 1:
            return bkd.reshape(
                bkd.arange(dims[0], dtype=bkd.int64_dtype()), (1, dims[0])
            )
        raise ValueError("dims must have at least 1 element")

    vectors = [bkd.arange(d, dtype=bkd.int64_dtype()) for d in dims]
    return cartesian_product(bkd, vectors)


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
    if len(samples_1d) < 2:
        # Handle 1D case directly
        if len(samples_1d) == 1:
            flat = bkd.ravel(samples_1d[0])
            return bkd.reshape(flat, (1, flat.shape[0]))
        raise ValueError("samples_1d must have at least 1 element")

    # cartesian_product already flattens inputs
    return cartesian_product(bkd, samples_1d)


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
    if len(weights_1d) < 2:
        # Handle 1D case directly
        if len(weights_1d) == 1:
            return bkd.ravel(weights_1d[0])
        raise ValueError("weights_1d must have at least 1 element")

    # outer_product returns shape (n1, n2, ...), flatten to 1D
    return bkd.ravel(outer_product(bkd, weights_1d))
