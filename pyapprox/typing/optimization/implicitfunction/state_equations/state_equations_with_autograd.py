from abc import ABC, abstractmethod
from typing import Generic

from pyapprox.typing.util.backend import Array, Backend


class StateEquationsWithAutoGrad(Generic[Array]):
    """
    State equations with automatic differentiation capabilities.

    This class implements parameterized residual equations with adjoint
    capabilities using automatic differentiation provided by the backend.

    Warning
    -------
    This is only intended to help debugging custom expressions for
    state equation derivatives. It is often faster to just
    use autograd on the output of the functional applied to the state equation
    rather than computing all these individual jacobians and HVPs
    """

    def __init__(self, backend: Backend):
        """
        Initialize the StateEquationsWithAutoGrad object.

        Parameters
        ----------
        backend : Backend
            Backend used for computations (e.g., NumPy or PyTorch).
        """
        self._validate_backend(backend)
        self._bkd = backend
        self._param = self._bkd.zeros((self.nparams(),))

    def _validate_backend(self, backend: Backend) -> None:
        """
        Validate the backend.

        Parameters
        ----------
        backend : Backend
            Backend to validate.

        Raises
        ------
        TypeError
            If the backend is not a valid instance of Backend or does not
            support automatic differentiation.
        """
        if not isinstance(backend, Backend):
            raise TypeError("Backend must be an instance of Backend.")
        if not hasattr(backend, "jacobian") or not hasattr(backend, "hvp"):
            raise TypeError(
                "Backend must support automatic differentiation. "
                "Ensure the backend has 'jacobian' and 'hvp' methods."
            )

    @abstractmethod
    def nparams(self) -> int:
        """
        Return the number of variables in the system.

        Returns
        -------
        int
            Number of variables.
        """
        raise NotImplementedError

    @abstractmethod
    def nstates(self) -> int:
        """
        Return the number of states in the system.

        Returns
        -------
        int
            Number of states.
        """
        raise NotImplementedError

    def set_parameters(self, param: Array) -> None:
        """
        Set the parameters for the residual equation.

        Parameters
        ----------
        param : Array
            Parameters to set.

        Raises
        ------
        ValueError
            If the parameter shape is invalid.
        """
        if param.shape != (self.nparams(),):
            raise ValueError(
                f"param has shape {param.shape} but must have "
                f"shape {(self.nparams(),)}"
            )
        self._param = param

    def get_parameters(self) -> Array:
        """
        Get the current parameters of the residual equation.

        Returns
        -------
        Array
            Current parameters.
        """
        if not hasattr(self, "_param"):
            raise AttributeError(
                "must call set_parameters before accessing parameters."
            )
        return self._param

    @abstractmethod
    def _value(self, iterate: Array, param: Array) -> Array:
        """
        Compute the residual value for the given iterate and parameters.

        Parameters
        ----------
        iterate : Array
            Current iterate.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Residual value.
        """
        raise NotImplementedError

    def value(self, init_state: Array) -> Array:
        """
        Compute the residual value for the given initial state.

        Parameters
        ----------
        init_state : Array
            Initial state.

        Returns
        -------
        Array
            Residual value.
        """
        self.get_parameters()  # Ensure parameters are set
        if init_state.shape != (self.nstates(),):
            raise ValueError(
                f"init_state has shape {init_state.shape} but must have "
                f"shape {(self.nstates(),)}"
            )
        val = self._value(init_state, self.get_parameters())
        if val.shape != (self.nstates(),):
            raise ValueError(
                f"value has shape {val.shape} but must be {(self.nstates(),)}"
            )
        return val

    def solve(self, init_state: Array, param: Array) -> Array:
        """
        Solve the residual equation for the given initial state and parameters.

        Parameters
        ----------
        init_state : Array
            Initial state.
        param : Array
            Parameters.

        Returns
        -------
        Array
            Solution to the residual equation.
        """
        self.set_parameters(param)
        return self.value(init_state)

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the residual with respect to the parameters.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Jacobian matrix with respect to the parameters.
        """
        if state.shape != (self.nstates(),):
            raise ValueError(
                f"state has shape {state.shape} but must have "
                f"shape {(self.nstates(),)}"
            )
        jac = self._bkd.jacobian(  # type: ignore
            lambda p: self._value(state, p), param
        )
        if jac.shape != (self.nstates(), self.nparams()):
            raise RuntimeError(
                f"jac has shape {jac.shape} but must have "
                f"shape {(self.nstates(), self.nparams())}"
            )
        return jac

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the residual with respect to the state.

        Parameters
        ----------
        state : Array
            State of the system.
        param : Array
            Parameters of the system.

        Returns
        -------
        Array
            Jacobian matrix with respect to the state.
        """
        if state.shape != (self.nstates(),):
            raise ValueError(
                f"state has shape {state.shape} but must have "
                f"shape {(self.nstates(),)}"
            )
        jac = self._bkd.jacobian(  # type: ignore
            lambda y: self._value(y, param), state
        )
        if jac.shape != (self.nstates(), self.nstates()):
            raise RuntimeError(
                f"jac has shape {jac.shape} but must have "
                f"shape {(self.nstates(), self.nstates())}"
            )
        return jac

    def _adjoint_dot_residual_param_wrapper(
        self, fwd_state: Array, param: Array, adj_state: Array
    ) -> Array:
        """
        Compute adjoint dot product with the residual with respect to
        parameters.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.

        Returns
        -------
        Array
            Adjoint dot product result.
        """
        return adj_state @ self._value(fwd_state, param)

    def _adjoint_dot_residual_state_wrapper(
        self, fwd_state: Array, param: Array, adj_state: Array
    ) -> Array:
        """
        Compute adjoint dot product with the residual with respect to state.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.

        Returns
        -------
        Array
            Adjoint dot product result.
        """
        return adj_state @ self._value(fwd_state, param)

    def param_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute Hessian-vector product with respect to parameters.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.
        vvec : Array
            Vector for Hessian-vector product.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        hvp = self._bkd.hvp(
            lambda p: adj_state @ self._value(fwd_state, p),
            param,
            vvec,
        )
        if hvp.ndim != 1:
            raise RuntimeError("_param_param_hvp must return 1D array")
        return hvp

    def state_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute Hessian-vector product with respect to the state.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.
        wvec : Array
            Vector for Hessian-vector product.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        hvp = self._bkd.hvp(
            lambda y: adj_state @ self._value(y, param),
            fwd_state,
            wvec,
        )
        if hvp.ndim != 1 or hvp.shape[0] != fwd_state.shape[0]:
            raise RuntimeError("_state_state_hvp must return 1D array")
        return hvp

    def param_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute Hessian-vector product with respect to parameters and state.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.
        wvec : Array
            Vector for Hessian-vector product.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        hvp = self._bkd.jacobian(
            lambda p: adj_state @ self.state_jacobian(fwd_state, p) @ wvec,
            param,
        )
        if hvp.ndim != 1 or hvp.shape[0] != param.shape[0]:
            raise RuntimeError(
                f"_param_state_hvp returned shape {hvp.shape} but "
                f"must have shape (param.shape[0],)"
            )
        return hvp

    def state_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute Hessian-vector product with respect to state and parameters.

        Parameters
        ----------
        fwd_state : Array
            Forward state.
        param : Array
            Parameters.
        adj_state : Array
            Adjoint state.
        vvec : Array
            Vector for Hessian-vector product.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        hvp = self._bkd.jacobian(
            lambda y: adj_state @ self.param_jacobian(y, param) @ vvec,
            fwd_state,
        )
        if hvp.ndim != 1 or hvp.shape[0] != fwd_state.shape[0]:
            raise RuntimeError(
                f"_state_param_hvp returned shape {hvp.shape} but "
                f"must have shape (fwd_state.shape[0],)"
            )
        return hvp
