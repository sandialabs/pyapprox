from typing import Generic, Protocol, runtime_checkable

from pyapprox.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)
from pyapprox.optimization.implicitfunction.operator.storage import (
    AdjointOperatorStorage,
)
from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    ParameterizedStateEquationWithJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class AdjointOperatorWithJacobianProtocol(Generic[Array], Protocol):
    """
    Protocol for adjoint operators with Jacobian support.

    This protocol defines the interface for adjoint operator implementations
    that support Jacobian computations for implicit function derivative checking.
    """

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend for numerical computations.
        """
        ...

    def jacobian(self, init_fwd_state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the adjoint operator.

        Parameters
        ----------
        init_fwd_state : Array
            Initial forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Jacobian matrix.
        """
        ...

    def storage(self) -> AdjointOperatorStorage:
        """
        Return the storage object for adjoint operator data.

        Returns
        -------
        AdjointOperatorStorage
            Storage object.
        """
        ...

    def state_equation(
        self,
    ) -> ParameterizedStateEquationWithJacobianProtocol[Array]:
        """
        Return the state equation.

        Returns
        -------
        ParameterizedStateEquationWithJacobianProtocol[Array]
            State equation object.
        """
        ...

    def functional(
        self,
    ) -> ParameterizedFunctionalWithJacobianProtocol[Array]:
        """
        Return the functional.

        Returns
        -------
        ParameterizedFunctionalWithJacobianProtocol[Array]
            Functional object.
        """
        ...
