"""
Adjoint functional for local OED.

The functional computes the quadratic form J(x, w) = x^T @ M0(w) @ x
and its derivatives with respect to state x and parameters w.
"""

from typing import Generic

from pyapprox.expdesign.local.protocols import DesignMatricesProtocol
from pyapprox.util.backends.protocols import Array, Backend


class QuadraticFunctional(Generic[Array]):
    """
    Quadratic functional for adjoint-based OED criteria.

    Computes J(x, w) = x^T @ M0(w) @ x and its derivatives, where:
    - x is the state vector (solution to forward problem)
    - w is the parameter vector (design weights)
    - M0(w) = sum_k w_k * M0_k

    Parameters
    ----------
    design_matrices : DesignMatricesProtocol[Array]
        Object that computes M0, M1 matrices.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        design_matrices: DesignMatricesProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        self._design_matrices = design_matrices
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nstates(self) -> int:
        """Number of state variables."""
        return self._design_matrices.ndesign_vars()

    def nparams(self) -> int:
        """Number of parameters (design weights)."""
        return self._design_matrices.ndesign_pts()

    def value(self, state: Array, params: Array) -> Array:
        """
        Evaluate the functional J(x, w) = x^T @ M0(w) @ x.

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)

        Returns
        -------
        Array
            Functional value. Shape: (1, 1)
        """
        M0 = self._design_matrices.M0(params)
        val = state @ M0 @ state
        return self._bkd.reshape(val, (1, 1))

    def state_jacobian(self, state: Array, params: Array) -> Array:
        """
        Jacobian of functional w.r.t. state: dJ/dx = 2 * M0 @ x.

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)

        Returns
        -------
        Array
            State Jacobian. Shape: (nstates,)
        """
        M0 = self._design_matrices.M0(params)
        return 2 * M0 @ state

    def param_jacobian(self, state: Array, params: Array) -> Array:
        """
        Jacobian of functional w.r.t. parameters: dJ/dw_k = x^T @ M0_k @ x.

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nparams,)
        """
        M0k = self._design_matrices.M0k()
        # For each k: state^T @ M0k[k] @ state
        return self._bkd.einsum("i,kij,j->k", state, M0k, state)

    def state_state_hvp(self, state: Array, params: Array, wvec: Array) -> Array:
        """
        Hessian-vector product d^2J/dx^2 @ w = 2 * M0 @ w.

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)
        wvec : Array
            Direction in state space. Shape: (nstates,)

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        M0 = self._design_matrices.M0(params)
        return 2 * M0 @ wvec

    def param_param_hvp(self, state: Array, params: Array, vvec: Array) -> Array:
        """
        Hessian-vector product d^2J/dw^2 @ v = 0 (J is linear in w).

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)
        vvec : Array
            Direction in parameter space. Shape: (nparams,)

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        return self._bkd.zeros((self.nparams(),))

    def state_param_hvp(self, state: Array, params: Array, vvec: Array) -> Array:
        """
        Mixed Hessian-vector product d^2J/dx dw @ v.

        d^2J/dx_i dw_k = 2 * M0k[i, :] @ x
        So: (d^2J/dx dw @ v)_i = 2 * sum_k v_k * (M0k[k] @ x)_i

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)
        vvec : Array
            Direction in parameter space. Shape: (nparams,)

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        M0k = self._design_matrices.M0k()
        # sum_k v_k * 2 * M0k[k] @ state
        return 2 * self._bkd.einsum("k,kij,j->i", vvec, M0k, state)

    def param_state_hvp(self, state: Array, params: Array, wvec: Array) -> Array:
        """
        Mixed Hessian-vector product d^2J/dw dx @ w.

        d^2J/dw_k dx_i = 2 * (M0k[k] @ x)_i + 2 * x^T @ M0k[k, :, i]
                       = 2 * (M0k[k] + M0k[k]^T)[i, :] @ ...
        Actually: d/dx_i (x^T @ M0k @ x) = 2 * M0k @ x (if M0k symmetric)
        So: d^2J/dw_k dx @ w = 2 * M0k @ w

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)
        wvec : Array
            Direction in state space. Shape: (nstates,)

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        M0k = self._design_matrices.M0k()
        # For each k: 2 * state^T @ M0k[k] @ wvec
        return 2 * self._bkd.einsum("i,kij,j->k", state, M0k, wvec)
