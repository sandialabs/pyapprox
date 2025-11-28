from typing import Union, overload, cast

from pyapprox.typing.util.backend import Array
from pyapprox.typing.interface.functions.function import FunctionProtocol
from pyapprox.typing.interface.functions.jacobian_protocols import (
    FunctionWithJacobianProtocol,
)
from pyapprox.typing.interface.functions.hessian_protocols import (
    FunctionWithJacobianAndHVPProtocol,
)
from pyapprox.typing.interface.functions.numpy.wrappers import (
    NumpyFunctionWrapper,
    NumpyFunctionWithJacobianWrapper,
    NumpyFunctionWithJacobianAndHVPWrapper,
)


# The type checker (e.g., mypy) processes overloads from top
# to bottom, and it uses the first matching overload to determine
# the type signature. This means that more specific overloads
# should come before more general ones.
# This fixes type errrors in downstream files but introduces errors likely
# Overloaded function signatures 1 and 2 overlap with incompatible return types
# in this file. I need to resolve this but I stick with current convention
# so that at least users will not see type errors in downstream files


@overload
def numpy_function_wrapper_factory(
    function: FunctionWithJacobianAndHVPProtocol[Array],
) -> NumpyFunctionWithJacobianAndHVPWrapper[Array]: ...


@overload
def numpy_function_wrapper_factory(
    function: FunctionWithJacobianProtocol[Array],
) -> NumpyFunctionWithJacobianWrapper[Array]: ...


@overload
def numpy_function_wrapper_factory(
    function: FunctionProtocol[Array],
) -> NumpyFunctionWrapper[Array]: ...


def numpy_function_wrapper_factory(
    function: Union[
        FunctionProtocol[Array],
        FunctionWithJacobianProtocol[Array],
        FunctionWithJacobianAndHVPProtocol[Array],
    ],
) -> Union[
    NumpyFunctionWrapper[Array],
    NumpyFunctionWithJacobianWrapper[Array],
    NumpyFunctionWithJacobianAndHVPWrapper[Array],
]:
    """
    Factory function to create a wrapper for functions based on the type of function.

    Parameters
    ----------
    function : Union[
        FunctionProtocol[Array],
        FunctionWithJacobianProtocol[Array],
        FunctionWithJacobianAndHVPProtocol[Array],
    ]
        The function object.

    Returns
    -------
    Union[
        NumpyFunctionWrapper[Array],
        NumpyFunctionWithJacobianWrapper[Array],
        NumpyFunctionWithJacobianAndHVPWrapper[Array],
    ]
        The appropriate wrapper for the function.

    Raises
    ------
    TypeError
        If the function does not satisfy any of the protocols.
    """
    if isinstance(function, FunctionWithJacobianAndHVPProtocol):
        return cast(
            NumpyFunctionWithJacobianAndHVPWrapper[Array],
            NumpyFunctionWithJacobianAndHVPWrapper(function),
        )

    if isinstance(function, FunctionWithJacobianProtocol):
        return cast(
            NumpyFunctionWithJacobianWrapper[Array],
            NumpyFunctionWithJacobianWrapper(function),
        )

    if isinstance(function, FunctionProtocol):
        return NumpyFunctionWrapper(function)

    raise TypeError(
        "The provided function must satisfy one of the following protocols: "
        "'FunctionProtocol', 'FunctionWithJacobianProtocol', or "
        "'FunctionWithJacobianAndHVPProtocol'. "
        f"Got an object of type {type(function).__name__}."
    )
