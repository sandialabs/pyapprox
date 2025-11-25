from typing import Protocol, Any, runtime_checkable

from pyapprox.typing.interface.functions.function import (
    FunctionProtocol,
    Function,
    validate_samples,
)
from pyapprox.typing.util.backend import Array, Backend


@runtime_checkable
class ParameterizedFunctionProtocol(FunctionProtocol, Protocol):
    def set_parameter(self, p: Array) -> None:
        """
        Set the parameter p.

        Parameters
        ----------
        p : Array
            The new value of the parameter p.
        """
        ...


@runtime_checkable
class ParameterizedFunctionWithJacobianProtocol(
    ParameterizedFunctionProtocol, Protocol
):
    def jacobian_wrt_parameters(self, x: Array) -> Array:
        """
        Compute the Jacobian of f(x, p) with respect to the parameters p.

        Parameters
        ----------
        x : Array
            The fixed value of x.

        Returns
        -------
        Array
            The Jacobian matrix with respect to p.
        """
        ...


@runtime_checkable
class ParameterizedFunctionWithHVPProtocol(
    ParameterizedFunctionWithJacobianProtocol, Protocol
):
    def hvp_wrt_parameters(self, x: Array, vec: Array) -> Array:
        """
        Compute the Hessian-vector product of f(x, p) with respect to the parameters p.

        Parameters
        ----------
        x : Array
            The fixed value of x.
        vec : Array
            The vector to multiply with the Hessian.

        Returns
        -------
        Array
            The Hessian-vector product with respect to p.
        """
        ...


def validate_parameterized_function(function: Any) -> None:
    if not isinstance(function, ParameterizedFunctionProtocol):
        raise TypeError(
            f"Invalid function type: expected an object implementing "
            f"ParameterizedFunctionProtocol, got {type(function).__name__}."
        )
