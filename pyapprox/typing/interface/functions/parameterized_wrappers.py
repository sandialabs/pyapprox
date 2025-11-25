from pyapprox.typing.interface.functions.function import (
    validate_samples,
    validate_sample,
)
from pyapprox.typing.interface.functions.function import Function
from pyapprox.typing.interface.functions.parameterized_function import (
    ParameterizedFunctionProtocol,
    ParameterizedFunctionWithJacobianProtocol,
    ParameterizedFunctionWithHVPProtocol,
    validate_parameterized_function,
)
from pyapprox.typing.util.backend import Array


class FunctionOfParameters(Function[Array]):
    """
    Converts a parameterized function f(x, p) into f(p) for fixed x.

    Parameters
    ----------
    param_func : ParameterizedFunction
        The parameterized function f(x, p).
    fixed_x : Array
        The fixed value of x.

    Attributes
    ----------
    param_func : ParameterizedFunction
        The parameterized function f(x, p).
    fixed_x : Array
        The fixed value of x.
    """

    def __init__(
        self, param_fun: ParameterizedFunctionProtocol[Array], fixed_x: Array
    ):
        validate_parameterized_function(param_fun)
        super().__init__(param_fun._bkd)
        self._param_fun = param_fun
        validate_samples(self.nvars(), fixed_x)
        self._fixed_x = fixed_x

    def nvars(self) -> int:
        return self._param_fun.nqoi()  # Number of parameters p

    def nqoi(self) -> int:
        return self._param_fun.nqoi()

    def nparams(self) -> int:
        return self._param_fun.nparams()

    def __call__(self, params: Array) -> Array:
        """
        Evaluate f(p) for the fixed value of x.

        Parameters
        ----------
        params : Array
            The parameter vectors.

        Returns
        -------
        Array
            The result of f(x, p) for the fixed x.
        """
        validate_samples(self.nvars(), params)
        vals = []
        for p in params.T:
            self._param_fun.set_parameter(p)
            vals.append(self._param_fun(self._fixed_x))
        return self._bkd.stack(vals, axis=0)


class FunctionOfParametersWithJacobian(FunctionOfParameters[Array]):
    """
    Extends FunctionOfParameters to support Jacobian computation.

    Parameters
    ----------
    param_fun : ParameterizedFunctionWithJacobianProtocol
        The parameterized function f(x, p) with Jacobian functionality.
    fixed_x : Array
        The fixed value of x.
    """

    def __init__(
        self,
        param_fun: ParameterizedFunctionWithJacobianProtocol[Array],
        fixed_x: Array,
    ):
        if not isinstance(
            param_fun, ParameterizedFunctionWithJacobianProtocol
        ):
            raise TypeError(
                f"Invalid function type: expected an object implementing "
                "ParameterizedFunctionWithJacobianProtocol, "
                f"got {type(param_fun).__name__}."
            )
        validate_parameterized_function(param_fun)
        self._bkd = param_fun._bkd
        self._param_fun: ParameterizedFunctionWithJacobianProtocol[Array] = (
            param_fun
        )
        self._fixed_x = fixed_x

    def jacobian(self, param: Array) -> Array:
        """
        Compute the Jacobian of f(p) with respect to the parameters p.

        Parameters
        ----------
        param : Array
            The parameter vector.

        Returns
        -------
        Array
            The Jacobian matrix with respect to p.
        """
        validate_sample(self.nparams(), param)
        self._param_fun.set_parameter(param)
        return self._param_fun.jacobian_wrt_parameters(self._fixed_x)


class FunctionOfParametersWithHVP(FunctionOfParametersWithJacobian[Array]):
    """
    Extends FunctionOfParametersWithJacobian to support Hessian-vector
    product computation.

    Parameters
    ----------
    param_fun : ParameterizedFunctionWithHVPProtocol
        The parameterized function f(x, p) with HVP functionality.
    fixed_x : Array
        The fixed value of x.
    """

    def __init__(
        self,
        param_fun: ParameterizedFunctionWithHVPProtocol[Array],
        fixed_x: Array,
    ):
        if not isinstance(param_fun, ParameterizedFunctionWithHVPProtocol):
            raise TypeError(
                "Invalid function type: expected an object implementing "
                "ParameterizedFunctionWithHVPProtocol, got "
                f"{type(param_fun).__name__}."
            )
        validate_parameterized_function(param_fun)
        self._bkd = param_fun._bkd
        self._param_fun = param_fun
        self._fixed_x = fixed_x

    def hvp(self, param: Array, vec: Array) -> Array:
        """
        Compute the Hessian-vector product of f(p) with respect to the parameters p.

        Parameters
        ----------
        param : Array
            The parameter vector.
        vec : Array
            The vector to multiply with the Hessian.

        Returns
        -------
        Array
            The Hessian-vector product with respect to p.
        """
        validate_sample(self.nparams(), param)
        self._param_fun.set_parameter(param)
        return self._param_fun.hvp_wrt_parameters(self._fixed_x, vec)
