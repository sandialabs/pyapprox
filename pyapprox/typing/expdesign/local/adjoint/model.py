"""
Adjoint model for local OED.

Combines the functional and residual to compute gradients and HVPs
using the adjoint method.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.local.protocols import DesignMatricesProtocol

from .functional import QuadraticFunctional
from .residual import LinearResidual


class AdjointModel(Generic[Array]):
    """
    Adjoint model for computing gradients of quadratic functionals.

    For the functional J(w) = x(w)^T @ M0(w) @ x(w) where M1(w) @ x(w) = vec,
    the adjoint method computes the gradient:
        dJ/dw = dJ/dw|explicit - lambda^T @ dR/dw @ x

    where:
    - dJ/dw|explicit = x^T @ dM0/dw @ x
    - lambda is the adjoint solution: M1^T @ lambda = dJ/dx = 2 * M0 @ x
    - dR/dw @ x = dM1/dw @ x

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
        self._functional = QuadraticFunctional(design_matrices, bkd)
        self._residual = LinearResidual(design_matrices, bkd)

        # Cache for forward and adjoint solutions
        self._cached_params: Optional[Array] = None
        self._cached_fwd_sol: Optional[Array] = None
        self._cached_adj_sol: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nparams(self) -> int:
        """Number of parameters (design weights)."""
        return self._design_matrices.ndesign_pts()

    def nstates(self) -> int:
        """Number of state variables."""
        return self._design_matrices.ndesign_vars()

    def set_vector(self, vec: Array) -> None:
        """
        Set the right-hand side vector for the forward problem.

        Parameters
        ----------
        vec : Array
            Right-hand side. Shape: (nstates,)
        """
        self._residual.set_vector(vec)
        self._clear_cache()

    def _clear_cache(self) -> None:
        """Clear cached solutions."""
        self._cached_params = None
        self._cached_fwd_sol = None
        self._cached_adj_sol = None

    def _cache_valid(self, params: Array) -> bool:
        """Check if cache is valid for given parameters."""
        if self._cached_params is None:
            return False
        return bool(
            self._bkd.allclose(
                self._cached_params, params, atol=1e-15, rtol=1e-15
            )
        )

    def _ensure_solved(self, params: Array) -> None:
        """Ensure forward and adjoint problems are solved."""
        if not self._cache_valid(params):
            # Solve forward problem
            self._cached_fwd_sol = self._residual.solve_forward(params)

            # Compute adjoint RHS and solve
            adj_rhs = self._functional.state_jacobian(
                self._cached_fwd_sol, params
            )
            self._cached_adj_sol = self._residual.solve_adjoint(params, adj_rhs)

            self._cached_params = self._bkd.copy(params)

    def forward_solution(self, params: Array) -> Array:
        """
        Get the forward solution x(w).

        Parameters
        ----------
        params : Array
            Design weights. Shape: (nparams, 1)

        Returns
        -------
        Array
            Forward solution. Shape: (nstates,)
        """
        self._ensure_solved(params)
        return self._cached_fwd_sol  # type: ignore

    def adjoint_solution(self, params: Array) -> Array:
        """
        Get the adjoint solution lambda(w).

        Parameters
        ----------
        params : Array
            Design weights. Shape: (nparams, 1)

        Returns
        -------
        Array
            Adjoint solution. Shape: (nstates,)
        """
        self._ensure_solved(params)
        return self._cached_adj_sol  # type: ignore

    def value(self, params: Array) -> Array:
        """
        Evaluate the functional J(w) = x(w)^T @ M0(w) @ x(w).

        Parameters
        ----------
        params : Array
            Design weights. Shape: (nparams, 1)

        Returns
        -------
        Array
            Functional value. Shape: (1, 1)
        """
        self._ensure_solved(params)
        return self._functional.value(self._cached_fwd_sol, params)  # type: ignore

    def jacobian(self, params: Array) -> Array:
        """
        Compute the gradient dJ/dw using the adjoint method.

        dJ/dw_k = x^T @ M0_k @ x - lambda^T @ M1_k @ x

        Parameters
        ----------
        params : Array
            Design weights. Shape: (nparams, 1)

        Returns
        -------
        Array
            Gradient. Shape: (1, nparams)
        """
        self._ensure_solved(params)
        fwd_sol = self._cached_fwd_sol
        adj_sol = self._cached_adj_sol

        # Explicit derivative: dJ/dw|explicit = x^T @ M0_k @ x
        explicit_grad = self._functional.param_jacobian(fwd_sol, params)

        # Adjoint contribution: -lambda^T @ M1_k @ x
        M1k = self._design_matrices.M1k()
        adjoint_contrib = -self._bkd.einsum("i,kij,j->k", adj_sol, M1k, fwd_sol)

        grad = explicit_grad + adjoint_contrib
        return self._bkd.reshape(grad, (1, self.nparams()))

    def hvp(self, params: Array, vvec: Array) -> Array:
        """
        Compute Hessian-vector product d^2J/dw^2 @ v.

        Parameters
        ----------
        params : Array
            Design weights. Shape: (nparams, 1)
        vvec : Array
            Direction vector. Shape: (nparams, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nparams, 1)
        """
        self._ensure_solved(params)
        fwd_sol = self._cached_fwd_sol
        adj_sol = self._cached_adj_sol

        v = vvec[:, 0]  # Convert to 1D

        # We need to compute d/dw (dJ/dw) @ v
        # dJ/dw = explicit_grad + adjoint_contrib
        # d(explicit_grad)/dw @ v + d(adjoint_contrib)/dw @ v

        # 1. d(explicit_grad)/dw @ v involves dx/dw @ v
        # dx/dw @ v is computed by solving: M1 @ (dx/dw @ v) = -dM1/dw @ v @ x
        # Let w_tilde = dx/dw @ v, solve: M1 @ w_tilde = -sum_k v_k M1_k @ x
        M1k = self._design_matrices.M1k()
        rhs_fwd = -self._bkd.einsum("k,kij,j->i", v, M1k, fwd_sol)
        w_tilde = self._residual.solve_forward(params)
        # Actually need to solve with different RHS
        M1inv = self._design_matrices.M1inv(params)
        w_tilde = M1inv @ rhs_fwd

        # 2. Contribution from functional second derivatives
        # d(x^T @ M0_k @ x)/dw @ v
        # = 2 * x^T @ M0_k @ (dx/dw @ v) + 0 (since M0_k doesn't depend on w)
        M0k = self._design_matrices.M0k()
        func_contrib = 2 * self._bkd.einsum("i,kij,j->k", fwd_sol, M0k, w_tilde)

        # 3. Contribution from adjoint term
        # d(-lambda^T @ M1_k @ x)/dw @ v
        # Need dlambda/dw @ v and dx/dw @ v (= w_tilde)

        # Solve for adjoint sensitivity: d lambda/dw @ v
        # M1^T @ (dlambda/dw @ v) = d(2*M0@x)/dw @ v - M1_k @ v @ lambda
        # = 2 * M0 @ w_tilde + 2 * sum_k v_k M0_k @ x - sum_k v_k M1_k @ lambda
        adj_rhs = (
            2 * self._design_matrices.M0(params) @ w_tilde
            + 2 * self._bkd.einsum("k,kij,j->i", v, M0k, fwd_sol)
            - self._bkd.einsum("k,kij,j->i", v, M1k, adj_sol)
        )
        lambda_tilde = M1inv @ adj_rhs

        # Adjoint contribution derivatives
        # d(-lambda^T @ M1_k @ x)/dw @ v
        # = -lambda_tilde^T @ M1_k @ x - lambda^T @ M1_k @ w_tilde
        adj_contrib = (
            -self._bkd.einsum("i,kij,j->k", lambda_tilde, M1k, fwd_sol)
            - self._bkd.einsum("i,kij,j->k", adj_sol, M1k, w_tilde)
        )

        hvp_result = func_contrib + adj_contrib
        return self._bkd.reshape(hvp_result, (self.nparams(), 1))
