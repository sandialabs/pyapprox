# TODO: should this one function just be moved into another file
# and this one deleted

from typing import Any

from pyapprox.interface.functions.parameterized.protocols import (
    ParameterizedFunctionProtocol,
)


def validate_parameterized_function(function: Any) -> None:
    if not isinstance(function, ParameterizedFunctionProtocol):
        raise TypeError(
            f"Invalid function type: expected an object implementing "
            f"ParameterizedFunctionProtocol, got {type(function).__name__}."
        )
