\from typing import Union, cast, overload

from pyapprox.interface.functions.numpy.wrappers import (
    NumpyFunctionWithJacobianAndHVPWrapper,
    NumpyFunctionWithJacobianAndWHVPWrapper,
    NumpyFunctionWithJacobianWrapper,
    NumpyFunctionWrapper,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.interface.functions.protocols.hessian import (
    FunctionWithJacobianAndHVPProtocol,
    FunctionWithJacobianAndWHVPProtocol,
)
from pyapprox.interface.functions.protocols.jacobian import (
    FunctionWithJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array

# The type checker (e.g., mypy) processes overloads from top
# to bottom, and it uses the first matching overload to determine
# the type signature. This means that more specific overloads
# should come before more general ones.
# This fixes type errrors in downstream files but introduces errors likely
# Overloaded function signatures 1 and 2 overlap with incompatible return types
# in this file. I need to resolve this but I stick with current convention
# so that at least users will not see type errors in downstream files


# TODO: Overloading is not really used in PyApprox.
# Is there a better solution or is this a good use of overload

@overload
def numpy_function_wrapper_factory(
    function: FunctionWithJacobianAndHVPProtocol[Array],
) -> NumpyFunctionWithJacobianAndWHVPWrapper[Array]: ...


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
    function: FunctionProtocol[Array],
) -> Union[
    NumpyFunctionWrapper[Array],
    NumpyFunctionWithJacobianWrapper[Array],
    NumpyFunctionWithJacobianAndHVPWrapper[Array],
]:
    """
    Factory function to create a wrapper for functions based on the type of function.

    Parameters
    ----------
    function : FunctionProtocol[Array]
        The function object.

    Returns
    -------
    Union[
        NumpyFunctionWrapper[Array],
        NumpyFunctionWithJacobianWrapper[Array],
        NumpyFunctionWithJacobianAndHVPWrapper[Array],
        NumpyFunctionWithJacobianAndWHVPWrapper[Array],
    ]
        The appropriate wrapper for the function.

    Raises
    ------
    TypeError
        If the function does not satisfy any of the protocols.
    """
    if isinstance(function, FunctionWithJacobianAndWHVPProtocol):
        return cast(
            NumpyFunctionWithJacobianAndWHVPWrapper[Array],
            NumpyFunctionWithJacobianAndWHVPWrapper(function),
        )

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
