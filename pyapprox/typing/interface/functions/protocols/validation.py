from typing import Any

from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)

from pyapprox.typing.util.backend import Array


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
        raise ValueError(
            "Invalid samples shape: expected 2 dimensions, "
            f"got {samples.ndim} dimensions."
        )

    # Validate the number of rows
    expected_rows = nvars
    actual_rows, actual_cols = samples.shape
    if actual_rows != expected_rows:
        raise ValueError(
            f"Invalid samples shape: expected {expected_rows} rows, "
            f"got {actual_rows} rows."
        )


def validate_values(nqoi: int, samples: Array, values: Array) -> None:
    expected_shape = (nqoi, samples.shape[1])
    if values.shape != expected_shape:
        raise ValueError(
            f"Invalid values shape: expected {expected_shape}, "
            f"got {values.shape}."
        )


def validate_sample(nvars: int, samples: Array) -> None:
    """
    Validate that the sample has shape (nvars, 1).
    Some member functions may only use 1 sample.
    """
    expected_shape = (nvars, 1)
    actual_shape = samples.shape
    if actual_shape != expected_shape:
        raise ValueError(
            f"Invalid sample shape: expected {expected_shape}, "
            f"got {actual_shape}."
        )


def validate_jacobian(nqoi: int, nvars: int, jac: Array) -> None:
    if jac.shape != (nqoi, nvars):
        raise ValueError(
            f"Jacobian shape mismatch: expected ({nqoi, nvars}), "
            f"got {jac.shape}"
        )


def validate_jacobians(
    nqoi: int, nvars: int, samples: Array, jac: Array
) -> None:
    if jac.shape != (samples.shape[1], nqoi, nvars):
        raise ValueError(
            f"Jacobian shape mismatch: expected "
            f"({samples.shape[1], nqoi, nvars}), got {jac.shape}"
        )


def validate_vector_for_apply(nvars: int, vec: Array) -> None:
    """
    Validate that the vector has the correct shape for apply operations
    (e.g., jvp).

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
    if vec.shape != (nvars, 1):
        raise ValueError(
            f"Invalid vector shape for apply operation: expected ({nvars}, 1), "
            f"got {vec.shape}."
        )


def validate_hvp(nvars: int, hvp: Array) -> None:
    if hvp.shape != (nvars, 1):
        raise ValueError(
            f"Hvp shape mismatch: expected " f"({nvars, 1}), got {hvp.shape}"
        )


def validate_function(function: Any) -> None:
    if not isinstance(function, FunctionProtocol):
        raise TypeError(
            f"Invalid function type: expected an object implementing "
            f"FunctionProtocol, got {type(function).__name__}. "
        )
