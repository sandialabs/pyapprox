from typing import Generic

from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.interface.functions.protocols.validation import (
    validate_samples,
    validate_sample,
    validate_values,
    validate_jacobian,
    validate_hvp,
)
from pyapprox.typing.interface.functions.parameterized.protocols import (
    ParameterizedFunctionProtocol,
    ParameterizedFunctionWithJacobianProtocol,
    ParameterizedFunctionWithJacobianAndHVPProtocol,
)
from pyapprox.typing.interface.functions.parameterized.validation import (
    validate_parameterized_function,
)
from pyapprox.typing.util.backends.protocols import Array, Backend


class FunctionOfParameters(Generic[Array]):
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
        self._bkd = param_fun.bkd()
        self._param_fun = param_fun
        validate_samples(self.nvars(), fixed_x)
        self._fixed_x = fixed_x

    def nvars(self) -> int:
        return self._param_fun.nparams()  # Number of parameters p

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

    def bkd(self) -> Backend[Array]:
        return self._bkd


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
        self._bkd = param_fun.bkd()
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


class FunctionOfParametersWithJacobianAndHVP(
    FunctionOfParametersWithJacobian[Array]
):
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
        param_fun: ParameterizedFunctionWithJacobianAndHVPProtocol[Array],
        fixed_x: Array,
    ):
        if not isinstance(
            param_fun, ParameterizedFunctionWithJacobianAndHVPProtocol
        ):
            raise TypeError(
                "Invalid function type: expected an object implementing "
                "ParameterizedFunctionWithHVPProtocol, got "
                f"{type(param_fun).__name__}."
            )
        validate_parameterized_function(param_fun)
        self._bkd = param_fun.bkd()
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


class _FunctionOfParameters(FunctionProtocol[Array]):
    """
    Adapts a ParameterizedFunction f(x, p) to a Function of p with x frozen.

    This class dynamically exposes jacobian/hvp based on underlying capabilities.

    Parameters
    ----------
    param_fun : ParameterizedFunctionProtocol[Array]
        The parameterized function f(x, p).
    fixed_x : Array
        The fixed value of x.

    Optional Methods
    ----------------
    This class uses dynamic method binding based on param_fun capabilities:

    - ``jacobian(params)``: Available if param_fun implements
      ``ParameterizedFunctionWithJacobianProtocol``.
    - ``hvp(params, vec)``: Available if param_fun implements
      ``ParameterizedFunctionWithJacobianAndHVPProtocol``.

    Check availability with ``hasattr(func, 'jacobian')`` or ``hasattr(func, 'hvp')``.

    Notes
    -----
    This class follows the dynamic binding pattern for optional methods.
    See docs/OPTIONAL_METHODS_CONVENTION.md for details.
    """

    def __init__(
        self,
        param_fun: ParameterizedFunctionProtocol[Array],
        fixed_x: Array,
    ):
        validate_parameterized_function(param_fun)
        self._bkd = param_fun.bkd()
        self._param_fun = param_fun
        validate_samples(param_fun.nvars(), fixed_x)
        self._fixed_x = fixed_x

        # Conditionally add derivative methods based on param_fun capabilities
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        """
        Conditionally add jacobian and hvp methods.

        Methods are only exposed if the underlying parameterized function
        supports them.
        """
        if isinstance(self._param_fun, ParameterizedFunctionWithJacobianProtocol):
            self.jacobian = self._jacobian

        if isinstance(
            self._param_fun, ParameterizedFunctionWithJacobianAndHVPProtocol
        ):
            self.hvp = self._hvp

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._param_fun.nparams()  # Dimension of p

    def nqoi(self) -> int:
        return self._param_fun.nqoi()  # Number of quantities of interest

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

    def _jacobian(self, params: Array) -> Array:
        """
        Compute the Jacobian of f(p) with respect to the parameters p.

        Parameters
        ----------
        params : Array
            The parameter vector.

        Returns
        -------
        Array
            The Jacobian matrix with respect to p.
        """
        validate_sample(self.nvars(), params)
        self._param_fun.set_parameter(params.squeeze())
        jac = self._param_fun.jacobian_wrt_parameters(self._fixed_x)
        validate_jacobian(self.nqoi(), self.nvars(), jac)
        return jac

    def _hvp(self, params: Array, vec: Array) -> Array:
        """
        Compute the Hessian-vector product of f(p) with respect to the parameters p.

        Parameters
        ----------
        params : Array
            The parameter vector.
        vec : Array
            The vector to multiply with the Hessian.

        Returns
        -------
        Array
            The Hessian-vector product with respect to p.
        """
        validate_sample(self.nvars(), params)
        self._param_fun.set_parameter(params.squeeze())
        hvp = self._param_fun.hvp_wrt_parameters(self._fixed_x, vec)
        validate_hvp(self.nvars(), hvp)
        return hvp
