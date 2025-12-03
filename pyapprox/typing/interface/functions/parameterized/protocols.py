from typing import Protocol, runtime_checkable, Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class ParameterizedFunctionProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int:
        """
        Return the number of variables in the function.
        """
        ...

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest in the function.
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the function with the given samples.
        """
        ...

    def set_parameter(self, p: Array) -> None:
        """
        Set the parameter p.

        Parameters
        ----------
        p : Array
            The new value of the parameter p.
        """
        ...

    def nparams(self) -> int: ...


@runtime_checkable
class ParameterizedFunctionWithJacobianProtocol(
    ParameterizedFunctionProtocol[Array], Protocol
):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int:
        """
        Return the number of variables in the function.
        """
        ...

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest in the function.
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the function with the given samples.
        """
        ...

    def set_parameter(self, p: Array) -> None:
        """
        Set the parameter p.

        Parameters
        ----------
        p : Array
            The new value of the parameter p.
        """
        ...

    def nparams(self) -> int: ...

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
class ParameterizedFunctionWithJacobianAndHVPProtocol(
    ParameterizedFunctionWithJacobianProtocol[Array], Protocol
):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int:
        """
        Return the number of variables in the function.
        """
        ...

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest in the function.
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the function with the given samples.
        """
        ...

    def set_parameter(self, p: Array) -> None:
        """
        Set the parameter p.

        Parameters
        ----------
        p : Array
            The new value of the parameter p.
        """
        ...

    def nparams(self) -> int: ...

    def jacobian_wrt_parameters(self, x: Array) -> Array: ...

    def hvp_wrt_parameters(self, x: Array, vec: Array) -> Array:
        """
        Compute the Hessian-vector product of f(x, p) with respect to the
        parameters p.

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
