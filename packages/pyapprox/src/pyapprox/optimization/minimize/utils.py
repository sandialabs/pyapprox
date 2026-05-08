"""Shared utilities for the optimization/minimize module."""

from typing import List

from pyapprox.util.backends.protocols import Array, Backend


def assemble_full_samples(
    design_sample: Array,
    quad_samples: Array,
    design_indices: List[int],
    random_indices: List[int],
    nvars_full: int,
    bkd: Backend[Array],
) -> Array:
    """Build full-dimensional samples by combining design + quadrature points.

    Replicates each design variable across all quadrature points and
    fills random variable rows with the quadrature samples.

    Parameters
    ----------
    design_sample : Array
        Design variable values. Shape ``(n_design, 1)``.
    quad_samples : Array
        Quadrature points for random variables.
        Shape ``(n_random_vars, n_quad_pts)``.
    design_indices : List[int]
        Indices of design variables in the full input vector.
    random_indices : List[int]
        Indices of random variables in the full input vector.
    nvars_full : int
        Total number of variables (design + random).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Full samples. Shape ``(nvars_full, n_quad_pts)``.
    """
    n_quad = quad_samples.shape[1]
    full = bkd.zeros((nvars_full, n_quad))
    for kk, idx in enumerate(design_indices):
        full[idx] = bkd.repeat(design_sample[kk], n_quad, axis=0)
    for kk, idx in enumerate(random_indices):
        full[idx] = quad_samples[kk]
    return full
