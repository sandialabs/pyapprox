from typing import Protocol, Generic, runtime_checkable

from pyapprox.typing.util.backend import Array
from pyapprox.typing.interface.functions.function import FunctionProtocol


class JacobianProtocol(Protocol, Generic[Array]):
    """
    Protocol for functions with Jacobian functionality.
    """

    def jacobian(self, sample: Array) -> Array:
        pass


class ApplyJacobianProtocol(Protocol, Generic[Array]):
    """
    Protocol for functions that implement Jacobian-vector product.
    """

    def apply_jacobian(self, sample: Array, vec: Array) -> Array:
        """
        Apply the Jacobian to a vector.

        Parameters
        ----------
        x : Array
            Input array of shape (nvars,).
        vce : Array
            Vector to apply the Jacobian to, of shape (nvars,).

        Returns
        -------
        Array
            Result of Jacobian-vector product of shape (nqoi,).
        """
        ...


@runtime_checkable
class FunctionWithJacobianProtocol(
    FunctionProtocol[Array], JacobianProtocol[Array], Protocol
):
    pass


@runtime_checkable
class FunctionWithApplyJacobianProtocol(
    FunctionProtocol[Array], ApplyJacobianProtocol[Array], Protocol
):
    pass
