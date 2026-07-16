"""Signature aliases and shape validators for Derivatives bundle fields.

A leaf module with no in-package imports: both
:mod:`pyapprox.interface.functions.derivatives` (which re-exports these
as its public surface) and
:mod:`pyapprox.interface.functions._shape_checked` (whose wrappers the
bundle builds) need the field signatures, and importing them from a
shared leaf keeps the package acyclic — the shape-checked wrappers must
not import the bundle module they are built by.
"""

from __future__ import annotations

from typing import Callable

from pyapprox.util.backends.protocols import Array

JacobianFn = Callable[[Array], Array]
JacobianBatchFn = Callable[[Array], Array]
JVPFn = Callable[[Array, Array], Array]
HVPFn = Callable[[Array, Array], Array]
WHVPFn = Callable[[Array, Array, Array], Array]
HessianFn = Callable[[Array], Array]
HessianBatchFn = Callable[[Array], Array]
HVPBatchFn = Callable[[Array, Array], Array]
WHVPBatchFn = Callable[[Array, Array, Array], Array]
InexactValueFn = Callable[[Array, float], Array]
InexactJacobianFn = Callable[[Array, float], Array]


def _check_shape(
    label: str, got: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    if got != expected:
        raise ValueError(f"{label}: expected shape {expected}, got {got}")

def validate_samples(nvars: int, samples: Array) -> None:
    """
    Validate that the given samples are a 2D array with the correct number of rows.

    Parameters
    ----------
    nvars : int
        Number of variables (expected number of rows in the samples).
    samples : Array
        Input samples to validate.

    Raises
    ------
    ValueError
        If the samples do not have 2 dimensions or if the number of rows is
        incorrect.
    """
    # Check if samples are 2D
    if samples.ndim != 2:
        if samples.ndim == 1:
            raise ValueError(
                f"Invalid samples shape: expected 2D array with shape "
                f"({nvars}, nsamples), got 1D array with shape {samples.shape}. "
                f"Use .reshape(-1, 1) for a single sample or .reshape({nvars}, -1) "
                f"for multiple samples."
            )
        raise ValueError(
            f"Invalid samples shape: expected 2D array with shape "
            f"({nvars}, nsamples), got {samples.ndim}D array."
        )

    # Validate the number of rows
    expected_rows = nvars
    actual_rows, actual_cols = samples.shape
    if actual_rows != expected_rows:
        raise ValueError(
            f"Invalid samples shape: expected {expected_rows} rows, "
            f"got {actual_rows} rows."
        )

def validate_sample(nvars: int, samples: Array) -> None:
    """
    Validate that the sample has shape (nvars, 1).

    Single-sample methods require 2D input with shape (nvars, 1).
    1D arrays are not accepted; use .reshape(-1, 1) to convert.

    Parameters
    ----------
    nvars : int
        Number of variables.
    samples : Array
        Single sample to validate.

    Raises
    ------
    ValueError
        If the sample does not have shape (nvars, 1).
    """
    if samples.ndim == 1:
        raise ValueError(
            f"Invalid sample shape: expected 2D array with shape ({nvars}, 1), "
            f"got 1D array with shape {samples.shape}. "
            f"Use .reshape(-1, 1) to convert a 1D array to a column vector."
        )
    expected_shape = (nvars, 1)
    actual_shape = samples.shape
    if actual_shape != expected_shape:
        raise ValueError(
            f"Invalid sample shape: expected {expected_shape}, got {actual_shape}."
        )

def validate_jacobian(nqoi: int, nvars: int, jac: Array) -> None:
    if jac.shape != (nqoi, nvars):
        raise ValueError(
            f"Jacobian shape mismatch: expected ({nqoi, nvars}), got {jac.shape}"
        )

def validate_vector_for_apply(nvars: int, vec: Array) -> None:
    """
    Validate that the vector has the correct shape for apply operations
    (e.g., jvp, hvp).

    Single-sample apply methods require 2D input with shape (nvars, 1).
    1D arrays are not accepted; use .reshape(-1, 1) to convert.

    Parameters
    ----------
    nvars : int
        The expected number of variables (length of the vector).
    vec : Array
        The input vector to validate.

    Raises
    ------
    ValueError
        If the vector does not have the expected shape.
    """
    if vec.ndim == 1:
        raise ValueError(
            f"Invalid vector shape: expected 2D array with shape ({nvars}, 1), "
            f"got 1D array with shape {vec.shape}. "
            f"Use .reshape(-1, 1) to convert a 1D array to a column vector."
        )
    if vec.shape != (nvars, 1):
        raise ValueError(
            f"Invalid vector shape for apply operation: expected ({nvars}, 1), "
            f"got {vec.shape}."
        )
