from typing import Union
from pyapprox.typing.interface.functions.parameterized.protocols import (
    ParameterizedFunctionProtocol,
    ParameterizedFunctionWithJacobianProtocol,
    ParameterizedFunctionWithJacobianAndHVPProtocol,
)
from pyapprox.typing.interface.functions.parameterized.wrappers import (
    FunctionOfParameters,
    FunctionOfParametersWithJacobian,
    FunctionOfParametersWithJacobianAndHVP,
    _FunctionOfParameters,
)
from pyapprox.typing.util.backend import Array
from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)


def convert_to_function_of_parameters(
    param_fun: ParameterizedFunctionProtocol[Array],
    fixed_x: Array,
) -> Union[
    ParameterizedFunctionProtocol[Array],
    ParameterizedFunctionWithJacobianProtocol[Array],
    ParameterizedFunctionWithJacobianAndHVPProtocol[Array],
]:
    """
    Factory function to create an appropriate FunctionOfParameters object
    based on the capabilities of the provided parameterized function.

    Parameters
    ----------
    param_fun : ParameterizedFunctionProtocol
        The parameterized function f(x, p).
    fixed_x : Array
        The fixed value of x.

    Returns
    -------
    Union[FunctionOfParameters, FunctionOfParametersWithJacobian, FunctionOfParametersWithHVP]
        An instance of the appropriate FunctionOfParameters class.
    """
    if isinstance(param_fun, ParameterizedFunctionWithJacobianAndHVPProtocol):
        return FunctionOfParametersWithJacobianAndHVP(param_fun, fixed_x)
    elif isinstance(param_fun, ParameterizedFunctionWithJacobianProtocol):
        return FunctionOfParametersWithJacobian(param_fun, fixed_x)
    elif isinstance(param_fun, ParameterizedFunctionProtocol):
        return FunctionOfParameters(param_fun, fixed_x)
    else:
        raise TypeError(
            "Invalid function type: expected an object implementing one of "
            "ParameterizedFunctionProtocol, "
            "ParameterizedFunctionWithJacobianProtocol, "
            "or ParameterizedFunctionWithHVPProtocol."
        )


def _convert_to_function_of_parameters(
    param_fun: ParameterizedFunctionProtocol[Array],
    fixed_x: Array,
) -> FunctionProtocol[Array]:
    """
    Factory function to create an appropriate FunctionOfInputs object
    based on the capabilities of the provided parameterized function.

    Note, this function is another way of carrying out a wrapper
    relative to convert_to_function_of_parameters. It returns
    a function that can have jacobian and hvp but it will not look
    like it has this to type checkers. On the postive, side
    it substantially reduces code.
    TODO: need to decide what approach to use and use it everywhere.

    Parameters
    ----------
    param_fun : ParameterizedFunctionProtocol
        The parameterized function f(x, p).
    fixed_x : Array
        The fixed value of x.

    Returns
    -------
    FunctionProtocol
        An instance of the FunctionOfInputs class.
    """
    if isinstance(param_fun, ParameterizedFunctionProtocol):
        return _FunctionOfParameters(param_fun, fixed_x)
    else:
        raise TypeError(
            "Invalid function type: expected an object implementing "
            "ParameterizedFunctionProtocol."
        )
