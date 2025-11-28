from typing import Union

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
        return NumpyFunctionWithJacobianAndHVPWrapper(function)

    if isinstance(function, FunctionWithJacobianProtocol):
        return NumpyFunctionWithJacobianWrapper(function)

    if isinstance(function, FunctionProtocol):
        return NumpyFunctionWrapper(function)

    raise TypeError(
        "The provided function must satisfy one of the following protocols: "
        "'FunctionProtocol', 'FunctionWithJacobianProtocol', or "
        "'FunctionWithJacobianAndHVPProtocol'. "
        f"Got an object of type {type(function).__name__}."
    )
