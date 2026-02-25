from typing import Any

from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)

from pyapprox.util.backends.protocols import Array


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
            f"Invalid sample shape: expected {expected_shape}, "
            f"got {actual_shape}."
        )


def validate_jacobian(nqoi: int, nvars: int, jac: Array) -> None:
    if jac.shape != (nqoi, nvars):
        raise ValueError(
            f"Jacobian shape mismatch: expected ({nqoi, nvars}), "
            f"got {jac.shape}"
        )


def validate_jacobian_batch(
    nqoi: int, nvars: int, nsamples: int, jac: Array
) -> None:
    """Validate batch Jacobian output shape.

    Parameters
    ----------
    nqoi : int
        Number of quantities of interest.
    nvars : int
        Number of input variables.
    nsamples : int
        Number of samples.
    jac : Array
        Jacobian array to validate.

    Raises
    ------
    ValueError
        If the Jacobian does not have shape (nsamples, nqoi, nvars).
    """
    expected_shape = (nsamples, nqoi, nvars)
    if jac.shape != expected_shape:
        raise ValueError(
            f"Jacobian batch shape mismatch: expected {expected_shape}, "
            f"got {jac.shape}"
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


def validate_1d_array(nvars: int, samples: Array) -> None:
    """
    Validate that the array has shape (nvars,).
    Some member functions may only use 1 sample.
    """
    expected_shape = (nvars,)
    actual_shape = samples.shape
    if actual_shape != expected_shape:
        raise ValueError(
            f"Invalid sample shape: expected {expected_shape}, "
            f"got {actual_shape}."
        )


def validate_hessian_batch(
    nqoi: int, nvars: int, nsamples: int, hess: Array
) -> None:
    """Validate batch Hessian output shape.

    Parameters
    ----------
    nqoi : int
        Number of quantities of interest.
    nvars : int
        Number of input variables.
    nsamples : int
        Number of samples.
    hess : Array
        Hessian array to validate.

    Raises
    ------
    ValueError
        If the Hessian does not have shape (nsamples, nqoi, nvars, nvars).
    """
    expected_shape = (nsamples, nqoi, nvars, nvars)
    if hess.shape != expected_shape:
        raise ValueError(
            f"Hessian batch shape mismatch: expected {expected_shape}, "
            f"got {hess.shape}"
        )


def validate_hvp_batch(nvars: int, nsamples: int, hvps: Array) -> None:
    """Validate batch HVP output shape.

    Parameters
    ----------
    nvars : int
        Number of input variables.
    nsamples : int
        Number of samples.
    hvps : Array
        HVP array to validate.

    Raises
    ------
    ValueError
        If the HVP does not have shape (nsamples, nvars).
    """
    expected_shape = (nsamples, nvars)
    if hvps.shape != expected_shape:
        raise ValueError(
            f"HVP batch shape mismatch: expected {expected_shape}, "
            f"got {hvps.shape}"
        )
