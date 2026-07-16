from typing import Any

from pyapprox.interface.functions._field_types import (
    validate_jacobian as validate_jacobian,
)
from pyapprox.interface.functions._field_types import (
    validate_sample as validate_sample,
)
from pyapprox.interface.functions._field_types import (
    validate_samples as validate_samples,
)
from pyapprox.interface.functions._field_types import (
    validate_vector_for_apply as validate_vector_for_apply,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.util.backends.protocols import Array


def validate_values(nqoi: int, samples: Array, values: Array) -> None:
    expected_shape = (nqoi, samples.shape[1])
    if values.shape != expected_shape:
        raise ValueError(
            f"Invalid values shape: expected {expected_shape}, got {values.shape}."
        )






def validate_jacobian_batch(nqoi: int, nvars: int, nsamples: int, jac: Array) -> None:
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
            f"Jacobian batch shape mismatch: expected {expected_shape}, got {jac.shape}"
        )




def validate_hvp(nvars: int, hvp: Array) -> None:
    if hvp.shape != (nvars, 1):
        raise ValueError(f"Hvp shape mismatch: expected ({nvars, 1}), got {hvp.shape}")


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
            f"Invalid sample shape: expected {expected_shape}, got {actual_shape}."
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
            f"HVP batch shape mismatch: expected {expected_shape}, got {hvps.shape}"
        )
