"""
Linear ODE benchmark for testing adjoint gradient computation.

Implements the linear ODE:
    dy/dt = A·y + B·p

where A is a stability matrix and B maps parameters to forcing.
The analytical solution is available for verification.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.validation import validate_backend


class LinearODEResidual(Generic[Array]):
    """
    Linear ODE residual: f(y, t; p) = A·y + B·p.

    This is a simple benchmark problem for testing time integrators
    and adjoint gradient computation.

    Parameters
    ----------
    Amat : Array
        Stability matrix. Shape: (nstates, nstates)
    Bmat : Array
        Parameter-to-forcing matrix. Shape: (nstates, nparams)
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        Amat: Array,
        Bmat: Array,
        bkd: Backend[Array],
    ):
        validate_backend(bkd)
        self._Amat = Amat
        self._Bmat = Bmat
        self._bkd = bkd
        self._time = 0.0
        self._param = None
        self._nstates = Amat.shape[0]
        self._nparams = Bmat.shape[1]

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._nparams

    def set_time(self, time: float) -> None:
        """Set the current time."""
        self._time = time

    def set_param(self, param: Array) -> None:
        """
        Set the parameters.

        Parameters
        ----------
        param : Array
            Parameters. Shape: (nparams, 1) or (nparams,)
        """
        if param.ndim == 1:
            param = param[:, None]
        self._param = param

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the ODE right-hand side.

        f(y) = A·y + B·p

        Parameters
        ----------
        state : Array
            State. Shape: (nstates,)

        Returns
        -------
        Array
            f(y). Shape: (nstates,)
        """
        forcing = (self._Bmat @ self._param).flatten()
        return self._Amat @ state + forcing

    def jacobian(self, state: Array) -> Array:
        """
        Compute df/dy = A.

        Parameters
        ----------
        state : Array
            State. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        return self._Amat

    def mass_matrix(self, nstates: int) -> Array:
        """
        Return the identity mass matrix.

        Parameters
        ----------
        nstates : int
            Number of states.

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        return self._bkd.eye(nstates)

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix to a vector (identity, returns vec)."""
        return vec

    def param_jacobian(self, state: Array) -> Array:
        """
        Compute df/dp = B.

        Parameters
        ----------
        state : Array
            State. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        return self._Bmat

    def initial_param_jacobian(self) -> Array:
        """
        Compute derivative of initial condition with respect to parameters.

        For this benchmark, initial condition is fixed, so this is zero.

        Returns
        -------
        Array
            Shape: (nstates, nparams)
        """
        return self._bkd.zeros((self._nstates, self._nparams))

    # =========================================================================
    # HVP Methods for second-order adjoints
    # =========================================================================

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute (d^2f/dy^2)·w contracted with adjoint.

        For linear ODE, this is zero (no second derivatives).
        """
        return self._bkd.zeros(state.shape)

    def state_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute (d^2f/dy dp)·v contracted with adjoint.

        For linear ODE, this is zero.
        """
        return self._bkd.zeros(state.shape)

    def param_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute (d^2f/dp dy)·w contracted with adjoint.

        For linear ODE, this is zero.
        """
        return self._bkd.zeros((self._nparams,))

    def param_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute (d^2f/dp^2)·v contracted with adjoint.

        For linear ODE, this is zero.
        """
        return self._bkd.zeros((self._nparams,))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self._nstates}, "
            f"nparams={self._nparams})"
        )


class QuadraticODEResidual(Generic[Array]):
    """
    Quadratic ODE residual for testing HVP computation.

    Implements: f(y, t; p) = A·y + p[0]·y² + p[1]

    where the quadratic term provides non-zero second derivatives for
    proper HVP testing.

    Parameters
    ----------
    Amat : Array
        Linear stability matrix. Shape: (nstates, nstates)
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        Amat: Array,
        bkd: Backend[Array],
    ):
        validate_backend(bkd)
        self._Amat = Amat
        self._bkd = bkd
        self._time = 0.0
        self._param = None
        self._nstates = Amat.shape[0]
        self._nparams = 2  # p[0] = quadratic coeff, p[1] = constant

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._nparams

    def set_time(self, time: float) -> None:
        """Set the current time."""
        self._time = time

    def set_param(self, param: Array) -> None:
        """
        Set the parameters.

        Parameters
        ----------
        param : Array
            Parameters [quad_coeff, constant]. Shape: (2, 1) or (2,)
        """
        if param.ndim == 1:
            param = param[:, None]
        self._param = param

    def __call__(self, state: Array) -> Array:
        """
        Evaluate f(y) = A·y + p[0]·y² + p[1].

        Parameters
        ----------
        state : Array
            State. Shape: (nstates,)

        Returns
        -------
        Array
            f(y). Shape: (nstates,)
        """
        p0 = float(self._param[0, 0])
        p1 = float(self._param[1, 0])
        return self._Amat @ state + p0 * state**2 + p1

    def jacobian(self, state: Array) -> Array:
        """
        Compute df/dy = A + 2·p[0]·diag(y).

        Parameters
        ----------
        state : Array
            State. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        p0 = float(self._param[0, 0])
        return self._Amat + 2.0 * p0 * self._bkd.diag(state)

    def mass_matrix(self, nstates: int) -> Array:
        """Return the identity mass matrix."""
        return self._bkd.eye(nstates)

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix to a vector (identity, returns vec)."""
        return vec

    def param_jacobian(self, state: Array) -> Array:
        """
        Compute df/dp.

        df/dp[0] = y²
        df/dp[1] = 1

        Parameters
        ----------
        state : Array
            State. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        jac = self._bkd.zeros((self._nstates, self._nparams))
        jac = self._bkd.copy(jac)
        jac[:, 0] = state**2
        jac[:, 1] = 1.0
        return jac

    def initial_param_jacobian(self) -> Array:
        """Initial condition is fixed, so this is zero."""
        return self._bkd.zeros((self._nstates, self._nparams))

    # =========================================================================
    # HVP Methods - Non-zero for quadratic ODE
    # =========================================================================

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute λᵀ·(d²f/dy²)·w.

        For f(y) = Ay + p[0]·y² + p[1]:
        d²f_i/(dy_j dy_k) = 2·p[0]·δ_{ij}·δ_{ik}

        (d²f/dy²)·w contracts last index:
        For each (i,j): Σ_k d²f_i/(dy_j dy_k)·w_k = 2·p[0]·δ_{ij}·w_i

        λᵀ·(d²f/dy²)·w contracts first index:
        For each j: Σ_i λ_i·2·p[0]·δ_{ij}·w_i = 2·p[0]·λ_j·w_j

        Returns shape (nstates,) for use in second adjoint RHS.
        """
        p0 = float(self._param[0, 0])
        return 2.0 * p0 * adj_state * wvec

    def state_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute λᵀ·(d²f/dy dp)·v.

        d²f_i/(dy_j dp_0) = 2·y_i·δ_{ij} (for p_0 = quadratic coeff)
        d²f_i/(dy_j dp_1) = 0

        (d²f/dydp)·v contracts p index:
        For each (i,j): 2·y_i·δ_{ij}·v_0 = 2·y_j·v_0·δ_{ij}

        λᵀ·(d²f/dydp)·v contracts i index:
        For each j: Σ_i λ_i·2·y_j·v_0·δ_{ij} = 2·λ_j·y_j·v_0

        Returns shape (nstates,).
        """
        v0 = float(vvec[0, 0]) if vvec.ndim == 2 else float(vvec[0])
        return 2.0 * adj_state * state * v0

    def param_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute λᵀ·(d²f/dp dy)·w.

        By symmetry of mixed partials, this equals:
        For p_0: Σ_i λ_i·2·y_i·w_i = 2·(λ * y * w).sum()
        For p_1: 0

        Returns shape (nparams,).
        """
        result = self._bkd.zeros((self._nparams,))
        result = self._bkd.copy(result)
        result[0] = 2.0 * float(self._bkd.sum(adj_state * state * wvec))
        return result

    def param_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute λᵀ·(d²f/dp²)·v.

        All second derivatives w.r.t. parameters are zero.
        """
        return self._bkd.zeros((self._nparams,))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self._nstates}, "
            f"nparams={self._nparams})"
        )
