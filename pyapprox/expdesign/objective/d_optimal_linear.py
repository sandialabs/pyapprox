"""
D-Optimal design objective for linear models with prior.

This implements Bayesian D-optimal design for linear inverse problems:
    obj(w) = -1/2 log(det(A^T W A * prior_cov / noise_cov + I))

where A is the design matrix (forward model), w are design weights,
and the goal is to maximize information gain about parameters.

References
----------
Alexanderian, A. and Saibaba, A.K.
"Efficient D-Optimal Design of Experiments for Infinite-Dimensional
Bayesian Linear Inverse Problems"
SIAM Journal on Scientific Computing 2018 40:5, A2956-A2985
https://doi.org/10.1137/17M115712X
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class DOptimalLinearModelObjective(Generic[Array]):
    """
    D-optimal objective for linear model with Gaussian prior.

    For linear model f(x) = A @ x with Gaussian prior and noise,
    computes the D-optimal criterion:
        obj(w) = -1/2 log(det(G + I))
    where G = A^T diag(w) A * prior_cov / noise_cov

    The negative sign makes this a minimization objective (minimizing
    -log(det) is equivalent to maximizing det, i.e., D-optimality).

    Parameters
    ----------
    design_matrix : Array
        Forward model matrix A. Shape: (nobs, nparams)
    noise_cov : Array
        Noise variance (scalar).
    prior_cov : Array
        Prior variance (scalar).
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    - noise_cov and prior_cov must be scalars (0-dimensional arrays)
    - Returns objective as shape (1, 1) for consistency with other objectives
    - Jacobian has shape (1, nobs)
    - Hessian has shape (1, nobs, nobs)
    """

    def __init__(
        self,
        design_matrix: Array,
        noise_cov: Array,
        prior_cov: Array,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._design_matrix = design_matrix
        self._noise_cov = noise_cov
        self._prior_cov = prior_cov
        self._nobs = design_matrix.shape[0]
        self._nparams = design_matrix.shape[1]

        # Validate scalar covariances
        if noise_cov.ndim != 0:
            raise TypeError("noise_cov must be a scalar (0-dimensional array)")
        if prior_cov.ndim != 0:
            raise TypeError("prior_cov must be a scalar (0-dimensional array)")

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observations (design locations)."""
        return self._nobs

    def nparams(self) -> int:
        """Number of model parameters."""
        return self._nparams

    def nvars(self) -> int:
        """Number of design variables (= nobs)."""
        return self._nobs

    def nqoi(self) -> int:
        """Number of outputs (= 1 for scalar objective)."""
        return 1

    def jacobian_implemented(self) -> bool:
        """Whether analytical Jacobian is available."""
        return True

    def hessian_implemented(self) -> bool:
        """Whether analytical Hessian is available."""
        return True

    def hvp_implemented(self) -> bool:
        """Whether Hessian-vector product is available."""
        return True

    def _Y(self, weights: Array) -> Array:
        """
        Compute Y = A^T diag(w) A * prior_cov / noise_cov + I.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Y : Array
            Shape: (nparams, nparams)
        """
        A = self._design_matrix
        # A^T @ diag(w) @ A = A^T @ (w * A)
        hess_misfit = (
            self._bkd.dot(A.T, weights * A) * self._prior_cov / self._noise_cov
        )
        ident = self._bkd.eye(self._nparams)
        return hess_misfit + ident

    def __call__(self, weights: Array) -> Array:
        """
        Compute D-optimal objective value.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        objective : Array
            Shape: (1, 1)
        """
        Y = self._Y(weights)
        # log(det(Y)) via slogdet for numerical stability
        _, logdet = self._bkd.slogdet(Y)
        # Return -1/2 * log(det(Y)) as (1, 1) array
        return self._bkd.reshape(-0.5 * logdet, (1, 1))

    def evaluate(self, weights: Array) -> Array:
        """Alias for __call__."""
        return self.__call__(weights)

    def jacobian(self, weights: Array) -> Array:
        """
        Compute Jacobian of objective w.r.t. weights.

        d/dw_i[-1/2 log(det(Y))] = -1/2 * tr(Y^{-1} dY/dw_i)

        where dY/dw_i = a_i @ a_i^T * prior_cov / noise_cov
        and a_i is the i-th row of A.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        jacobian : Array
            Shape: (1, nobs)
        """
        Y = self._Y(weights)
        inv_Y = self._bkd.inv(Y)
        const = self._prior_cov / self._noise_cov

        # For each row a_i of A, compute tr(Y^{-1} @ a_i @ a_i^T)
        # = a_i^T @ Y^{-1} @ a_i (since trace of outer product)
        A = self._design_matrix
        # Efficient: (A @ inv_Y) * A summed over columns = diag(A @ inv_Y @ A^T)
        jac_log_det = self._bkd.sum(
            self._bkd.dot(A, inv_Y) * A, axis=1
        ) * const

        return self._bkd.reshape(-0.5 * jac_log_det, (1, self._nobs))

    def hessian(self, weights: Array) -> Array:
        """
        Compute Hessian of objective w.r.t. weights.

        d^2/dw_i dw_j[-1/2 log(det(Y))]

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        hessian : Array
            Shape: (1, nobs, nobs)
        """
        Y = self._Y(weights)
        inv_Y = self._bkd.inv(Y)
        # Compute det via slogdet: det = sign * exp(logdet)
        sign, logdet = self._bkd.slogdet(Y)
        det_Y = sign * self._bkd.exp(logdet)
        const = self._prior_cov / self._noise_cov

        A = self._design_matrix

        # Compute Y^{-1} @ a_i @ a_i^T for each row i
        # Pre-compute A @ inv_Y for efficiency
        A_inv_Y = self._bkd.dot(A, inv_Y)

        # tr(Y^{-1} @ a_i @ a_i^T) = sum_k (A @ inv_Y)[i, k] * A[i, k]
        traces = self._bkd.sum(A_inv_Y * A, axis=1)

        # Jacobian of det(Y)
        jac_det_Y = traces * (det_Y * const)

        # Compute Hessian of det(Y) by building rows and stacking
        # This avoids item assignment which isn't supported by all backends
        const2 = det_Y * const**2

        # Pre-compute Y_inv_dY_i for all i
        Y_inv_dY_list = []
        for ii in range(self._nobs):
            ai = A[ii : ii + 1, :].T  # (nparams, 1)
            Y_inv_dY_i = self._bkd.dot(inv_Y, self._bkd.dot(ai, ai.T))
            Y_inv_dY_list.append(Y_inv_dY_i)

        # Build hessian rows
        hess_rows = []
        for ii in range(self._nobs):
            row_vals = []
            for jj in range(self._nobs):
                # tr(Y^{-1} dY_i) * tr(Y^{-1} dY_j) - tr(Y^{-1} dY_i @ Y^{-1} dY_j)
                term1 = traces[ii] * traces[jj]
                term2 = self._bkd.trace(
                    self._bkd.dot(Y_inv_dY_list[ii], Y_inv_dY_list[jj])
                )
                row_vals.append(const2 * (term1 - term2))
            hess_rows.append(self._bkd.stack(row_vals))

        hess_det_Y = self._bkd.stack(hess_rows)

        # Hessian of log(det(Y))
        # d^2 log(f) = (d^2 f) / f - (df)^2 / f^2
        jac_det_Y_col = self._bkd.reshape(jac_det_Y, (self._nobs, 1))
        hess_log_det_Y = (
            hess_det_Y / det_Y
            - self._bkd.dot(jac_det_Y_col, jac_det_Y_col.T) / det_Y**2
        )

        return self._bkd.reshape(-0.5 * hess_log_det_Y, (1, self._nobs, self._nobs))

    def hvp(self, weights: Array, vec: Array) -> Array:
        """
        Compute Hessian-vector product.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)
        vec : Array
            Vector to multiply. Shape: (nobs, 1)

        Returns
        -------
        result : Array
            Hessian-vector product. Shape: (1, nobs)
        """
        # Compute full hessian and multiply by vec
        # For small problems this is efficient; for large problems
        # a matrix-free approach would be better
        hess = self.hessian(weights)  # (1, nobs, nobs)
        hess_2d = self._bkd.reshape(hess, (self._nobs, self._nobs))
        result = self._bkd.dot(hess_2d, vec)  # (nobs, 1)
        return result.T  # (1, nobs)
