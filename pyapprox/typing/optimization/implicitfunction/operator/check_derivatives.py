from typing import Optional, List
import unittest  # Enable check_derivatives with good error messages

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.derivative_checks.wrappers import (
    FunctionWithJVP,
    FunctionWithJVPFromHVP,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


class ImplicitFunctionDerivativeChecker:
    """
    Class for checking derivatives of implicit functions.

    This class encapsulates derivative checking functionality, including
    first-order derivatives (Jacobian) and second-order derivatives (HVPs).
    """

    def __init__(self, backend: Backend, residual_eq, functional):
        """
        Initialize the ImplicitFunctionDerivativeChecker object.

        Parameters
        ----------
        backend : Backend
            Backend used for computations (e.g., NumPy or PyTorch).
        residual_eq : object
            Residual equation object implementing derivative checking methods.
        functional : object
            Functional object implementing derivative checking methods.
        """
        self._backend = backend
        self._residual_eq = residual_eq
        self._functional = functional

    def check_state_jacobian(
        self,
        init_state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        disp: bool = False,
    ) -> Array:
        """
        Check the state Jacobian of the residual equation.

        Returns
        -------
        Array
            Errors for state Jacobian check.
        """
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=self._residual_eq.nstates(),
            nvars=self._residual_eq.nstates(),
            fun=lambda state: self._residual_eq.value(state),
            jacobian=lambda state: self._residual_eq.state_jacobian(
                state, param
            ),
            bkd=self._backend,
        )
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            init_state,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=disp,
        )[0]

    def check_param_jacobian(
        self,
        init_state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        disp: bool = False,
    ) -> Array:
        """
        Check the parameter Jacobian of the residual equation.

        Returns
        -------
        Array
            Errors for parameter Jacobian check.
        """
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=self._residual_eq.nstates(),
            nvars=self._residual_eq.nvars(),
            fun=lambda param: self._residual_eq.value(init_state),
            jacobian=lambda param: self._residual_eq.param_jacobian(
                init_state, param
            ),
            bkd=self._backend,
        )
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param, fd_eps=fd_eps, direction=None, relative=True, verbosity=disp
        )

    def check_param_param_hvp(
        self,
        init_state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        disp: bool = False,
    ) -> Array:
        """
        Check the parameter-parameter Hessian-vector product of the residual equation.

        Returns
        -------
        Array
            Errors for parameter-parameter HVP check.
        """
        wrapper = FunctionWithJVPFromHVP(
            function=self._residual_eq,
            weights=adj_state,
        )
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param, fd_eps=fd_eps, direction=None, relative=True, verbosity=disp
        )

    def check_state_state_hvp(
        self,
        init_state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        disp: bool = False,
    ) -> Array:
        """
        Check the state-state Hessian-vector product of the residual equation.

        Returns
        -------
        Array
            Errors for state-state HVP check.
        """
        wrapper = FunctionWithJVPFromHVP(
            function=self._residual_eq,
            weights=adj_state,
        )
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            init_state,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=disp,
        )

    def check_param_state_hvp(
        self,
        init_state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        disp: bool = False,
    ) -> Array:
        """
        Check the parameter-state Hessian-vector product of the residual equation.

        Returns
        -------
        Array
            Errors for parameter-state HVP check.
        """
        wrapper = FunctionWithJVPFromHVP(
            function=self._residual_eq,
            weights=adj_state,
        )
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param, fd_eps=fd_eps, direction=None, relative=True, verbosity=disp
        )

    def check_state_param_hvp(
        self,
        init_state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        disp: bool = False,
    ) -> Array:
        """
        Check the state-parameter Hessian-vector product of the residual equation.

        Returns
        -------
        Array
            Errors for state-parameter HVP check.
        """
        wrapper = FunctionWithJVPFromHVP(
            function=self._residual_eq,
            weights=adj_state,
        )
        checker = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            init_state,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=disp,
        )

    def check_derivatives(
        self,
        init_state: Array,
        param: Array,
        tols: Array,
        fd_eps: Optional[Array] = None,
        disp: bool = False,
    ) -> None:
        """
        Check all derivatives one by one.

        Parameters
        ----------
        init_state : Array
            Initial state for derivative checks.
        param : Array
            Parameters for derivative checks.
        tols : Array
            Tolerances for derivative checks.
        fd_eps : Optional[Array], optional
            Finite difference step size.
        disp : bool, optional
            Whether to display detailed output. Default is False.

        Raises
        ------
        AssertionError
            If any derivative check fails.
        """
        # Create an instance of TestCase for assertions
        self._unittest = unittest.TestCase()

        # Check first-order derivatives of the residual equation
        errors = self.check_state_jacobian(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[0])

        errors = self.check_param_jacobian(
            init_state, param, fd_eps, disp=disp
        )
        self._assert_derivatives_close(errors, tols[1])

        # Check second-order derivatives of the residual equation
        adj_state = self._residual_eq.adjoint_data().get_adjoint_state()

        errors = self.check_param_param_hvp(
            init_state, param, adj_state, disp=disp
        )
        self._assert_derivatives_close(errors, tols[2])

        errors = self.check_state_state_hvp(
            init_state, param, adj_state, disp=disp
        )
        self._assert_derivatives_close(errors, tols[3])

        errors = self.check_param_state_hvp(
            init_state, param, adj_state, disp=disp
        )
        self._assert_derivatives_close(errors, tols[4])

        errors = self.check_state_param_hvp(
            init_state, param, adj_state, disp=disp
        )
        self._assert_derivatives_close(errors, tols[5])

    def _assert_derivatives_close(self, errors: Array, tol: float) -> None:
        """
        Assert that the derivatives are within the specified tolerance.

        Parameters
        ----------
        errors : Array
            Errors from derivative checks.
        tol : float
            Tolerance for derivative checks.

        Raises
        ------
        AssertionError
            If the derivatives are not within the specified tolerance.
        """
        if self._backend.min(errors) == self._backend.max(errors):
            assert self._backend.min(errors) == 0.0
        else:
            self._unittest.assertLessEqual(
                self._backend.min(errors) / self._backend.max(errors), tol
            )

    def get_derivative_tolerances(self, tol: float) -> Array:
        """
        Get tolerances for derivative checks.

        Parameters
        ----------
        tol : float
            Tolerance value for derivative checks.

        Returns
        -------
        Array
            Array of tolerances for derivative checks.
        """
        nchecks = 6  # Total number of derivative checks
        return self._backend.full((nchecks,), tol)
