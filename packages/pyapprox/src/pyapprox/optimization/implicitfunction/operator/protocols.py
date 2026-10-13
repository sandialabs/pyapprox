from typing import Generic, Protocol, runtime_checkable

from pyapprox.optimization.implicitfunction.functionals.protocols import (
    SteadyFunctionalProtocol,
)
from pyapprox.optimization.implicitfunction.operator.derivatives import (
    ImplicitOperatorDerivatives,
)
from pyapprox.optimization.implicitfunction.operator.storage import (
    AdjointOperatorStorage,
)
from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    SteadyStateEquationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ImplicitOperatorProtocol(Generic[Array], Protocol):
    """
    Protocol for operators differentiating a functional of the solution
    of a steady state equation (``AdjointOperator``,
    ``ForwardSensitivityOperator``).

    The parameter Jacobian is required; the Hessian-vector product is
    optional and comes from ``derivatives()``. Consumed by the implicit
    function derivative checker.
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

    def derivatives(self) -> ImplicitOperatorDerivatives[Array]:
        """Return the optional derivatives this operator provides."""
        ...

    def storage(self) -> AdjointOperatorStorage[Array]:
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
    ) -> SteadyStateEquationProtocol[Array]:
        """
        Return the state equation.

        Returns
        -------
        SteadyStateEquationProtocol[Array]
            State equation object.
        """
        ...

    def functional(
        self,
    ) -> SteadyFunctionalProtocol[Array]:
        """
        Return the functional.

        Returns
        -------
        SteadyFunctionalProtocol[Array]
            Functional object.
        """
        ...
