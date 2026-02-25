"""
Linear residual for local OED adjoint method.

The residual represents the linear system M1(w) @ x = vec, which defines
the forward problem in the adjoint method.
"""

from typing import Generic

from pyapprox.expdesign.local.protocols import DesignMatricesProtocol
from pyapprox.util.backends.protocols import Array, Backend


class LinearResidual(Generic[Array]):
    """
    Linear residual for the forward problem M1(w) @ x = vec.

    This class represents the linear system that defines the state x
    as a function of parameters w: M1(w) @ x(w) = vec.

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
        self._vec: Array = None  # type: ignore

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nstates(self) -> int:
        """Number of state variables."""
        return self._design_matrices.ndesign_vars()

    def nparams(self) -> int:
        """Number of parameters (design weights)."""
        return self._design_matrices.ndesign_pts()

    def set_vector(self, vec: Array) -> None:
        """
        Set the right-hand side vector.

        Parameters
        ----------
        vec : Array
            Right-hand side. Shape: (nstates,)
        """
        self._vec = vec

    def solve_forward(self, params: Array) -> Array:
        """
        Solve the forward problem: M1(w) @ x = vec.

        Parameters
        ----------
        params : Array
            Design weights. Shape: (nparams, 1)

        Returns
        -------
        Array
            Solution x. Shape: (nstates,)
        """
        M1inv = self._design_matrices.M1inv(params)
        return M1inv @ self._vec

    def solve_adjoint(self, params: Array, rhs: Array) -> Array:
        """
        Solve the adjoint problem: M1(w)^T @ lambda = rhs.

        Since M1 is symmetric, this is the same as the forward solve.

        Parameters
        ----------
        params : Array
            Design weights. Shape: (nparams, 1)
        rhs : Array
            Right-hand side. Shape: (nstates,)

        Returns
        -------
        Array
            Adjoint solution lambda. Shape: (nstates,)
        """
        M1inv = self._design_matrices.M1inv(params)
        return M1inv @ rhs

    def jacobian(self, params: Array) -> Array:
        """
        Jacobian of residual w.r.t. state (= M1).

        Parameters
        ----------
        params : Array
            Design weights. Shape: (nparams, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nstates, nstates)
        """
        return self._design_matrices.M1(params)

    def param_jacobian(self, state: Array, params: Array) -> Array:
        """
        Jacobian of residual w.r.t. parameters.

        d(M1 @ x)/dw_k = M1_k @ x

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nparams, nstates)
        """
        M1k = self._design_matrices.M1k()
        # For each k: M1k[k] @ state
        return self._bkd.einsum("kij,j->ki", M1k, state)

    def state_state_hvp(
        self, state: Array, adj_sol: Array, params: Array, wvec: Array
    ) -> Array:
        """
        Hessian-vector product for state-state block (= 0 for linear system).

        Parameters
        ----------
        state : Array
            Forward solution. Shape: (nstates,)
        adj_sol : Array
            Adjoint solution. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)
        wvec : Array
            Direction in state space. Shape: (nstates,)

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        return self._bkd.zeros((self.nstates(),))

    def param_param_hvp(
        self, state: Array, adj_sol: Array, params: Array, vvec: Array
    ) -> Array:
        """
        Hessian-vector product for param-param block (= 0 for linear in w).

        Parameters
        ----------
        state : Array
            Forward solution. Shape: (nstates,)
        adj_sol : Array
            Adjoint solution. Shape: (nstates,)
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

    def param_state_hvp(
        self, state: Array, adj_sol: Array, params: Array, wvec: Array
    ) -> Array:
        """
        Hessian-vector product for param-state block.

        Contribution to dL/dw from adjoint: -lambda^T @ dR/dw
        where R = M1 @ x - vec.
        d^2R/dw_k dx @ w = M1k @ w
        So: d(-lambda^T @ M1k @ x)/dx @ w = -lambda^T @ M1k @ w

        Parameters
        ----------
        state : Array
            Forward solution. Shape: (nstates,)
        adj_sol : Array
            Adjoint solution. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)
        wvec : Array
            Direction in state space. Shape: (nstates,)

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        M1k = self._design_matrices.M1k()
        # For each k: -adj_sol^T @ M1k[k] @ wvec
        return -self._bkd.einsum("i,kij,j->k", adj_sol, M1k, wvec)

    def state_param_hvp(
        self, state: Array, adj_sol: Array, params: Array, vvec: Array
    ) -> Array:
        """
        Hessian-vector product for state-param block.

        Parameters
        ----------
        state : Array
            Forward solution. Shape: (nstates,)
        adj_sol : Array
            Adjoint solution. Shape: (nstates,)
        params : Array
            Design weights. Shape: (nparams, 1)
        vvec : Array
            Direction in parameter space. Shape: (nparams,)

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        M1k = self._design_matrices.M1k()
        # sum_k v_k * (-M1k[k]^T @ adj_sol) = -sum_k v_k * M1k[k] @ adj_sol
        return -self._bkd.einsum("k,kij,j->i", vvec, M1k, adj_sol)
