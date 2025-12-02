from typing import Optional, List, Generic, Protocol, runtime_checkable
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
    FunctionWithJVPFromCallable,
)
from pyapprox.typing.optimization.implicitfunction.operator.operator_with_jacobian import (
    AdjointOperatorWithJacobian,
)
from pyapprox.typing.optimization.implicitfunction.operator.sensitivities import (
    VectorAdjointOperatorWithJacobian,
)
from pyapprox.typing.interface.functions.protocols.validation import (
    validate_sample,
)
from pyapprox.typing.optimization.implicitfunction.operator.storage import (
    AdjointOperatorStorage,
)
from pyapprox.typing.optimization.implicitfunction.state_equations.protocols import (
    ParameterizedStateEquationWithJacobianProtocol,
    ParameterizedStateEquationWithJacobianAndHVPProtocol,
)
from pyapprox.typing.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)


@runtime_checkable
class AdjointOperatorWithJacobianProtocol(Generic[Array], Protocol):
    def bkd(self) -> Backend[Array]: ...
    def jacobian(self, init_fwd_state: Array, param: Array) -> Array: ...
    def storage(self) -> AdjointOperatorStorage: ...
    def state_equation(
        self,
    ) -> ParameterizedStateEquationWithJacobianProtocol[Array]: ...

    def functional(
        self,
    ) -> ParameterizedFunctionalWithJacobianProtocol[Array]: ...


class ImplicitFunctionDerivativeChecker(Generic[Array]):
    """
    Class for checking derivatives of implicit functions.

    This class encapsulates derivative checking functionality, including
    first-order derivatives (Jacobian) and second-order derivatives (HVPs).
    """

    def __init__(
        self, adjoint_operator: AdjointOperatorWithJacobianProtocol[Array]
    ):
        """
        Initialize the ImplicitFunctionDerivativeChecker object.

        Parameters
        ----------
        bkd : Backend
            Backend used for computations (e.g., NumPy or PyTorch).
        state_eq : object
            State equation object implementing derivative checking methods.
        functional : object
            Functional object implementing derivative checking methods.
        """
        self._validate_adjoint_operator(adjoint_operator)
        self._bkd = adjoint_operator.bkd()
        self._adjoint_operator = adjoint_operator
        self._state_eq = adjoint_operator.state_equation()
        self._functional = adjoint_operator.functional()

    def _validate_adjoint_operator(
        self, adjoint_operator: AdjointOperatorWithJacobianProtocol[Array]
    ) -> None:
        """
        Validate the adjoint operator.

        Parameters
        ----------
        adjoint_operator : AdjointOperatorWithJacobian
            Adjoint operator object.

        Raises
        ------
        TypeError
            If the adjoint operator is not a valid instance of
            AdjointOperatorWithJacobian.
        """
        if not isinstance(
            adjoint_operator, AdjointOperatorWithJacobianProtocol
        ):
            raise TypeError(
                "adjoint_operator must be an instance of "
                "AdjointOperatorWithJacobianProtocol."
            )

    def check_state_equation_state_jacobian(
        self,
        init_state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the state Jacobian of the state equation.

        Returns
        -------
        Array
            Errors for state Jacobian check.
        """
        if verbosity > 0:
            print(f"{self._state_eq}.check_state_jacobian")
        validate_sample(self._state_eq.nstates(), init_state)
        validate_sample(self._state_eq.nparams(), param)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=self._state_eq.nstates(),
            nvars=self._state_eq.nstates(),
            fun=lambda state: self._state_eq(state, param),
            jacobian=lambda state: self._state_eq.state_jacobian(state, param),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            init_state,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_state_equation_param_jacobian(
        self,
        init_state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the parameter Jacobian of the state equation.

        Returns
        -------
        Array
            Errors for parameter Jacobian check.
        """
        if verbosity > 0:
            print(f"{self._state_eq}.check_param_jacobian")
        validate_sample(self._state_eq.nstates(), init_state)
        validate_sample(self._state_eq.nparams(), param)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=self._state_eq.nstates(),
            nvars=self._state_eq.nparams(),
            fun=lambda param: self._state_eq(init_state, param),
            jacobian=lambda param: self._state_eq.param_jacobian(
                init_state, param
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_functional_state_jacobian(
        self,
        init_state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the state Jacobian of the state equation.

        Returns
        -------
        Array
            Errors for state Jacobian check.
        """
        if verbosity > 0:
            print(f"{self._functional}.check_state_jacobian")
        validate_sample(self._state_eq.nstates(), init_state)
        validate_sample(self._state_eq.nparams(), param)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=self._functional.nqoi(),
            nvars=self._functional.nstates(),
            fun=lambda state: self._functional(state, param),
            jacobian=lambda state: self._functional.state_jacobian(
                state, param
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            init_state,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_functional_param_jacobian(
        self,
        init_state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the parameter Jacobian of the state equation.

        Returns
        -------
        Array
            Errors for parameter Jacobian check.
        """
        if verbosity > 0:
            print(f"{self._functional}.check_param_jacobian")
        validate_sample(self._state_eq.nstates(), init_state)
        validate_sample(self._state_eq.nparams(), param)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=self._functional.nqoi(),
            nvars=self._functional.nparams(),
            fun=lambda param: self._functional(init_state, param),
            jacobian=lambda param: self._functional.param_jacobian(
                init_state, param
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_jacobian(
        self,
        init_state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        if verbosity > 0:
            print(f"{self}.check_jacobian")
        validate_sample(self._state_eq.nstates(), init_state)
        validate_sample(self._state_eq.nparams(), param)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=self._functional.nqoi(),
            nvars=self._functional.nparams(),
            fun=lambda param: self._functional(
                self._state_eq.solve(init_state, param), param
            ),
            jacobian=lambda param: self._adjoint_operator.jacobian(
                init_state, param
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_state_equation_param_param_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the parameter-parameter Hessian-vector product of the state equation.

        Returns
        -------
        Array
            Errors for parameter-parameter HVP check.
        """
        if verbosity > 0:
            print(f"{self._state_eq}.check_param_param_hvp")
        validate_sample(self._state_eq.nstates(), state)
        validate_sample(self._state_eq.nstates(), adj_state)
        validate_sample(self._state_eq.nparams(), param)
        wrapper = FunctionWithJVPFromCallable(
            nqoi=self._functional.nparams(),
            nvars=self._functional.nparams(),
            fun=lambda param: self._state_eq.param_jacobian(state, param).T
            @ adj_state,
            jvp=lambda p, v: self._state_eq.param_param_hvp(
                state, p, adj_state, v
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_state_equation_state_state_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the state-state Hessian-vector product of the state equation.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        adj_state : Array
            Adjoint state for derivative checks.
        fd_eps : Optional[Array], optional
            Finite difference step size.
        verbosity : int, optional
            Verbosity level for detailed output. Default is 0.

        Returns
        -------
        Array
            Errors for state-state HVP check.
        """
        if verbosity > 0:
            print(f"{self._state_eq}.check_state_state_hvp")
        validate_sample(self._state_eq.nstates(), state)
        validate_sample(self._state_eq.nstates(), adj_state)
        validate_sample(self._state_eq.nparams(), param)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=self._state_eq.nstates(),
            nvars=self._state_eq.nstates(),
            fun=lambda state: self._state_eq.state_jacobian(state, param).T
            @ adj_state,
            jvp=lambda state, vec: self._state_eq.state_state_hvp(
                state, param, adj_state, vec
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            state,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_state_equation_param_state_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the parameter-state Hessian-vector product of the state equation.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        adj_state : Array
            Adjoint state for derivative checks.
        fd_eps : Optional[Array], optional
            Finite difference step size.
        verbosity : int, optional
            Verbosity level for detailed output. Default is 0.

        Returns
        -------
        Array
            Errors for parameter-state HVP check.
        """
        if verbosity > 0:
            print(f"{self._state_eq}.check_param_state_hvp")
        validate_sample(self._state_eq.nstates(), state)
        validate_sample(self._state_eq.nstates(), adj_state)
        validate_sample(self._state_eq.nparams(), param)
        wrapper = FunctionWithJVPFromCallable(
            nqoi=self._state_eq.nparams(),
            nvars=self._state_eq.nstates(),
            fun=lambda state: self._state_eq.param_jacobian(state, param).T
            @ adj_state,
            jvp=lambda state, vec: self._state_eq.param_state_hvp(
                state, param, adj_state, vec
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            state,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_state_equation_state_param_hvp(
        self,
        state: Array,
        param: Array,
        adj_state: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the state-parameter Hessian-vector product of the state equation.

        Returns
        -------
        Array
            Errors for state-parameter HVP check.
        """
        if verbosity > 0:
            print(f"{self._state_eq}.check_state_param_hvp")
        validate_sample(self._state_eq.nstates(), state)
        validate_sample(self._state_eq.nstates(), adj_state)
        validate_sample(self._state_eq.nparams(), param)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=self._state_eq.nstates(),
            nvars=self._state_eq.nparams(),
            fun=lambda param: self._state_eq.state_jacobian(state, param)
            @ adj_state,
            jvp=lambda param, vec: self._state_eq.state_param_hvp(
                state, param, adj_state, vec
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_functional_param_param_hvp(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the state-state Hessian-vector product of the state equation.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        fd_eps : Optional[Array], optional
            Finite difference step size.
        verbosity : int, optional
            Verbosity level for detailed output. Default is 0.

        Returns
        -------
        Array
            Errors for state-state HVP check.
        """
        if verbosity > 0:
            print(f"{self._functional}.check_state_state_hvp")
        validate_sample(self._functional.nstates(), state)
        validate_sample(self._functional.nparams(), param)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=self._functional.nparams(),
            nvars=self._functional.nparams(),
            fun=lambda param: self._functional.param_jacobian(state, param).T,
            jvp=lambda param, vec: self._functional.param_param_hvp(
                state, param, vec
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_functional_state_state_hvp(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the state-state Hessian-vector product of the functional.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        fd_eps : Optional[Array], optional
            Finite difference step size.
        verbosity : int, optional
            Verbosity level for detailed output. Default is 0.

        Returns
        -------
        Array
            Errors for state-state HVP check.
        """
        if verbosity > 0:
            print(f"{self._functional}.check_state_state_hvp")
        validate_sample(self._functional.nstates(), state)
        validate_sample(self._functional.nparams(), param)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=self._functional.nstates(),
            nvars=self._functional.nstates(),
            fun=lambda state: self._functional.state_jacobian(state, param).T,
            jvp=lambda state, vec: self._functional.state_state_hvp(
                state, param, vec
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            state,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_functional_param_state_hvp(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the parameter-state Hessian-vector product of the functional.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        fd_eps : Optional[Array], optional
            Finite difference step size.
        verbosity : int, optional
            Verbosity level for detailed output. Default is 0.

        Returns
        -------
        Array
            Errors for parameter-state HVP check.
        """
        if verbosity > 0:
            print(f"{self._functional}.check_param_state_hvp")
        validate_sample(self._functional.nstates(), state)
        validate_sample(self._functional.nparams(), param)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=self._functional.nparams(),
            nvars=self._functional.nstates(),
            fun=lambda state: self._functional.param_jacobian(state, param).T,
            jvp=lambda state, vec: self._functional.param_state_hvp(
                state, param, vec
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            state,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_functional_state_param_hvp(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        """
        Check the state-parameter Hessian-vector product of the functional.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.
        fd_eps : Optional[Array], optional
            Finite difference step size.
        verbosity : int, optional
            Verbosity level for detailed output. Default is 0.

        Returns
        -------
        Array
            Errors for state-parameter HVP check.
        """
        if verbosity > 0:
            print(f"{self._functional}.check_state_param_hvp")
        validate_sample(self._functional.nstates(), state)
        validate_sample(self._functional.nparams(), param)

        wrapper = FunctionWithJVPFromCallable(
            nqoi=self._functional.nstates(),
            nvars=self._functional.nparams(),
            fun=lambda param: self._functional.state_jacobian(state, param).T,
            jvp=lambda param, vec: self._functional.state_param_hvp(
                state, param, vec
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_hvp(
        self,
        state: Array,
        param: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
    ) -> Array:
        if verbosity > 0:
            print(f"{self}.check_jacobian")
        validate_sample(self._state_eq.nstates(), state)
        validate_sample(self._state_eq.nparams(), param)
        wrapper = FunctionWithJVPFromCallable(
            nqoi=self._functional.nparams(),
            nvars=self._functional.nparams(),
            fun=lambda param: self._adjoint_operator.jacobian(state, param).T,
            jvp=lambda param, vec: self._adjoint_operator.hvp(
                state, param, vec
            ),
            bkd=self._bkd,
        )
        checker: DerivativeChecker[Array] = DerivativeChecker(wrapper)
        return checker.check_derivatives(
            param,
            fd_eps=fd_eps,
            direction=None,
            relative=True,
            verbosity=verbosity,
        )[0]

    def check_derivatives(
        self,
        init_state: Array,
        param: Array,
        tols: Array,
        fd_eps: Optional[Array] = None,
        verbosity: int = 0,
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
        verbosity : bool, optional
            Whether to verbositylay detailed output. Default is False.

        Raises
        ------
        AssertionError
            If any derivative check fails.
        """
        validate_sample(self._state_eq.nstates(), init_state)
        validate_sample(self._state_eq.nparams(), param)
        # Create an instance of TestCase for assertions
        self._unittest = unittest.TestCase()

        # Check first-order derivatives of the state equation
        errors = self.check_state_equation_state_jacobian(
            init_state, param, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[0])

        errors = self.check_state_equation_param_jacobian(
            init_state, param, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[1])

        # Check first order derivatives of the functional
        errors = self.check_functional_state_jacobian(
            init_state, param, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[2])

        errors = self.check_functional_param_jacobian(
            init_state, param, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[3])

        # Check Jacobian
        errors = self.check_jacobian(
            init_state, param, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[4])

        if (
            not isinstance(
                self._functional, ParameterizedFunctionalWithJacobianProtocol
            )
            or not isinstance(
                self._state_eq,
                ParameterizedStateEquationWithJacobianAndHVPProtocol,
            )
            or isinstance(
                self._adjoint_operator, VectorAdjointOperatorWithJacobian
            )
        ):
            return

        # Check second-order derivatives of the state equation
        state = self._adjoint_operator.storage().get_forward_state()
        adj_state = self._adjoint_operator.storage().get_adjoint_state()

        errors = self.check_state_equation_param_param_hvp(
            state, param, adj_state, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[5])

        errors = self.check_state_equation_state_state_hvp(
            state, param, adj_state, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[6])

        errors = self.check_state_equation_param_state_hvp(
            state, param, adj_state, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[7])

        errors = self.check_state_equation_state_param_hvp(
            state, param, adj_state, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[8])

        errors = self.check_functional_param_param_hvp(
            state, param, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[9])

        errors = self.check_functional_state_state_hvp(
            state, param, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[10])

        errors = self.check_functional_param_state_hvp(
            state, param, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[11])

        errors = self.check_functional_state_param_hvp(
            state, param, fd_eps, verbosity=verbosity
        )
        self._assert_derivatives_close(errors, tols[12])

        # check Hessian
        errors = self.check_hvp(state, param, fd_eps, verbosity=verbosity)
        self._assert_derivatives_close(errors, tols[13])

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
        if self._bkd.min(errors) == self._bkd.max(errors):
            assert self._bkd.min(errors) == 0.0
        else:
            self._unittest.assertLessEqual(
                self._bkd.min(errors) / self._bkd.max(errors), tol
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
        nchecks = 14  # Total number of derivative checks
        return self._bkd.full((nchecks,), tol)

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns
        -------
        repr : str
            String representation of the class, including the parameter values.
        """
        return "{0}(\n    {1},\n    {2}\n)".format(
            self.__class__.__name__, self._state_eq, self._functional
        )
