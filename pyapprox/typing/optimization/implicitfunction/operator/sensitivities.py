from typing import Generic
from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.optimization.implicitfunction.state_equations.protocols import (
    ParameterizedStateEquationWithJacobianProtocol,
)
from pyapprox.typing.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)
from pyapprox.typing.optimization.implicitfunction.operator.storage import (
    AdjointOperatorStorage,
)


class VectorAdjointOperatorWithJacobian(Generic[Array]):
    """
    Vector adjoint operator for solving adjoint equations and computing
    sensitivities.

    This class encapsulates adjoint operator functionality for vector
    functionals with Jacobian computations.
    """

    def __init__(
        self,
        residual_eq: ParameterizedStateEquationWithJacobianProtocol[Array],
        functional: ParameterizedFunctionalWithJacobianProtocol[Array],
    ):
        """
        Initialize the VectorAdjointOperatorWithJacobian object.

        Parameters
        ----------
        residual_eq : ParameterizedStateEquationWithJacobianProtocol
            Residual equation object implementing the parameterized state
            equation protocol.
        functional : ParameterizedFunctionalWithJacobianProtocol
            Functional object implementing the parameterized functional
            protocol.

        Raises
        ------
        TypeError
            If the residual equation or functional are not valid instances of
            their respective protocols.
        """
        self.validate_residual_eq(residual_eq)
        self.validate_functional(functional)
        self._bkd = residual_eq.bkd()
        self._residual_eq = residual_eq
        if not self._bkd.bkd_equal(self._bkd, functional.bkd()):
            raise TypeError(
                "residual_eq backend does not match functional backend"
            )
        self._functional = functional
        self._adjoint_data = AdjointOperatorStorage(self._bkd)

    def validate_residual_eq(
        self,
        residual_eq: ParameterizedStateEquationWithJacobianProtocol[Array],
    ) -> None:
        """
        Validate the residual equation.

        Parameters
        ----------
        residual_eq : ParameterizedStateEquationWithJacobianProtocol
            Residual equation object.

        Raises
        ------
        TypeError
            If the residual equation is not a valid instance of
            ParameterizedStateEquationWithJacobianProtocol.
        """
        if not isinstance(
            residual_eq, ParameterizedStateEquationWithJacobianProtocol
        ):
            raise TypeError(
                "residual_eq must be an instance of "
                "ParameterizedStateEquationWithJacobianProtocol."
            )

    def validate_functional(
        self, functional: ParameterizedFunctionalWithJacobianProtocol[Array]
    ) -> None:
        """
        Validate the functional.

        Parameters
        ----------
        functional : ParameterizedFunctionalWithJacobianProtocol
            Functional object.

        Raises
        ------
        TypeError
            If the functional is not a valid instance of
            ParameterizedFunctionalWithJacobianProtocol.
        """
        if not isinstance(
            functional, ParameterizedFunctionalWithJacobianProtocol
        ):
            raise TypeError(
                "functional must be an instance of "
                "ParameterizedFunctionalWithJacobianProtocol."
            )

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend
            Backend used for computations.
        """
        return self._bkd

    def adjoint_data(self) -> AdjointOperatorStorage:
        """
        Return the adjoint operator storage.

        Returns
        -------
        AdjointOperatorStorage
            Storage for adjoint operator data.
        """
        return self._adjoint_data

    def nvars(self) -> int:
        """
        Return the number of variables in the system.

        Returns
        -------
        int
            Number of variables.
        """
        return self._residual_eq.nvars()

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        int
            Number of quantities of interest.
        """
        return self._functional.nqoi()

    def value(self, init_fwd_state: Array, param: Array) -> Array:
        """
        Compute the value of the functional.

        Parameters
        ----------
        init_fwd_state : Array
            Initial forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Value of the functional.
        """
        fwd_state = self._get_forward_state(init_fwd_state, param)
        return self._functional(fwd_state, param)

    def _get_forward_state(self, init_fwd_state: Array, param: Array) -> Array:
        """
        Get the forward state, computing it if necessary.

        Parameters
        ----------
        init_fwd_state : Array
            Initial forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Forward state.
        """
        if (
            not self._adjoint_data.has_parameter(param)
            or not self._adjoint_data.has_forward_state()
        ):
            self._adjoint_data._clear()
            fwd_state = self._residual_eq.solve(init_fwd_state, param)
            self._adjoint_data.set_forward_state(fwd_state)
        return self._adjoint_data.get_forward_state()

    def _sensitivities(self, fwd_state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the state with respect to the parameters.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Sensitivities (Jacobian of the state with respect to the
            parameters).
        """
        drdy = self._residual_eq.state_jacobian(fwd_state, param)
        drdp = self._residual_eq.param_jacobian(fwd_state, param)
        sens = self._bkd.solve(drdy, -drdp)
        return sens

    def jacobian(self, init_fwd_state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the functional with respect to the parameters.

        Parameters
        ----------
        init_fwd_state : Array
            Initial forward state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Jacobian of the functional with respect to the parameters.
        """
        fwd_state = self._get_forward_state(init_fwd_state, param)
        sens = self._sensitivities(fwd_state, param)
        dqdy = self._functional.state_jacobian(fwd_state, param)
        dqdp = self._functional.param_jacobian(fwd_state, param)
        return dqdy @ sens + dqdp
