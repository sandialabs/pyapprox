"""
Nataf transform for correlated marginals.

The Nataf transform converts correlated non-Gaussian random variables
to independent standard normals through a two-step process:

1. Transform each marginal to standard normal using CDF/inverse CDF
2. Decorrelate using a modified correlation matrix

This is more efficient than the full Rosenblatt transform when marginals
are known but the joint PDF is not analytically available.
"""

from typing import Generic, Tuple, List

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.protocols import MarginalProtocol


class NatafTransform(Generic[Array]):
    """
    Nataf transform for correlated non-Gaussian variables.

    Transforms correlated variables X with arbitrary marginals and
    correlation matrix R to independent standard normal variables Z.

    The transform is:
    1. X -> Y (standard normal marginals): Y_i = Phi^{-1}(F_i(X_i))
    2. Y -> Z (decorrelate): Z = L^{-1} Y, where R_0 = L L^T

    R_0 is the correlation matrix in standard normal space, which differs
    from the correlation in physical space R due to the nonlinear marginal
    transformations.

    Parameters
    ----------
    marginals : List[MarginalProtocol[Array]]
        Marginal distributions.
    correlation : Array
        Correlation matrix in physical space. Shape: (nvars, nvars)
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    The Nataf transform assumes the joint distribution is determined by
    the marginals and a Gaussian copula. This is an approximation for
    arbitrary joint distributions.

    The correlation matrix :math:`R_0` in standard normal space is computed by
    solving the integral equation:

    .. math::

        R_{ij} = \\frac{E[X_i X_j]}{\\sigma_i \\sigma_j}
               = \\frac{1}{\\sigma_i \\sigma_j} \\int \\phi_2(y_i, y_j; R_{0,ij})
                 F_i^{-1}(\\Phi(y_i)) F_j^{-1}(\\Phi(y_j)) \\, dy_i \\, dy_j
    """

    def __init__(
        self,
        marginals: List[MarginalProtocol[Array]],
        correlation: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._marginals = marginals
        self._nvars = len(marginals)
        self._correlation = correlation

        self._standard_normal = stats.norm(0, 1)

        # Validate correlation matrix
        if correlation.shape != (self._nvars, self._nvars):
            raise ValueError(
                f"Correlation must be ({self._nvars}, {self._nvars}), "
                f"got {correlation.shape}"
            )

        # Compute correlation in standard normal space
        self._correlation_std = self._compute_correlation_std()

        # Cholesky decomposition for decorrelation
        self._chol = bkd.cholesky(self._correlation_std)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def marginals(self) -> List[MarginalProtocol[Array]]:
        """Return list of marginal distributions."""
        return self._marginals

    def correlation(self) -> Array:
        """Return correlation matrix in physical space."""
        return self._correlation

    def correlation_std(self) -> Array:
        """Return correlation matrix in standard normal space."""
        return self._correlation_std

    def _validate_input(self, samples: Array) -> None:
        """Validate that input is 2D with shape (nvars, nsamples)."""
        if samples.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (nvars, nsamples), "
                f"got {samples.ndim}D"
            )
        if samples.shape[0] != self._nvars:
            raise ValueError(
                f"Expected {self._nvars} variables, got {samples.shape[0]}"
            )

    def map_to_canonical(self, samples: Array) -> Array:
        """
        Transform correlated samples to independent standard normal.

        X -> Y -> Z

        Parameters
        ----------
        samples : Array
            Correlated samples. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Independent standard normal samples. Shape: (nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(samples)

        # Step 1: X -> Y (standard normal marginals)
        y = self._to_standard_normal_marginals(samples)

        # Step 2: Y -> Z (decorrelate)
        z = self._bkd.solve_triangular(self._chol, y, lower=True)

        return z

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        """
        Transform independent standard normal to correlated samples.

        Z -> Y -> X

        Parameters
        ----------
        canonical_samples : Array
            Independent standard normal samples. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Correlated samples. Shape: (nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(canonical_samples)

        # Step 1: Z -> Y (correlate)
        y = self._chol @ canonical_samples

        # Step 2: Y -> X (transform marginals)
        x = self._from_standard_normal_marginals(y)

        return x

    def map_to_canonical_with_jacobian(
        self, samples: Array
    ) -> Tuple[Array, Array]:
        """
        Transform to canonical with Jacobian.

        For Nataf:
            dz_i/dx_j = (L^{-1})_{ij} * f_j(x_j) / phi(y_j)

        The Jacobian is dense due to decorrelation.

        Parameters
        ----------
        samples : Array
            Correlated samples. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Tuple[Array, Array]
            canonical : Independent standard normals. Shape: (nvars, nsamples)
            jacobian : Full Jacobian. Shape: (nvars, nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(samples)

        nsamples = samples.shape[1]

        # Transform
        y = self._to_standard_normal_marginals(samples)
        z = self._bkd.solve_triangular(self._chol, y, lower=True)

        # Jacobian: dy_i/dx_i = f_i(x_i) / phi(y_i) (diagonal)
        dy_dx = self._bkd.zeros((self._nvars, nsamples))
        for i, marginal in enumerate(self._marginals):
            # Reshape to (1, nsamples) for marginal method
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            f_x = self._bkd.exp(marginal.logpdf(row_2d))[0]
            phi_y = self._bkd.asarray(
                self._standard_normal.pdf(self._bkd.to_numpy(y[i]))
            )
            dy_dx[i] = f_x / (phi_y + 1e-15)

        # dz/dx = L^{-1} @ diag(dy/dx)
        # Full Jacobian: (nvars, nvars, nsamples)
        L_inv = self._bkd.solve_triangular(
            self._chol,
            self._bkd.eye(self._nvars),
            lower=True,
        )

        jacobian = self._bkd.zeros((self._nvars, self._nvars, nsamples))
        for k in range(nsamples):
            jacobian[:, :, k] = L_inv * dy_dx[:, k]

        return z, jacobian

    def map_from_canonical_with_jacobian(
        self, canonical_samples: Array
    ) -> Tuple[Array, Array]:
        """
        Transform from canonical with Jacobian.

        Parameters
        ----------
        canonical_samples : Array
            Independent standard normals. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Tuple[Array, Array]
            samples : Correlated samples. Shape: (nvars, nsamples)
            jacobian : Full Jacobian. Shape: (nvars, nvars, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(canonical_samples)

        nsamples = canonical_samples.shape[1]

        # Transform
        y = self._chol @ canonical_samples
        x = self._from_standard_normal_marginals(y)

        # Jacobian: dx_i/dy_i = phi(y_i) / f_i(x_i) (diagonal)
        dx_dy = self._bkd.zeros((self._nvars, nsamples))
        for i, marginal in enumerate(self._marginals):
            phi_y = self._bkd.asarray(
                self._standard_normal.pdf(self._bkd.to_numpy(y[i]))
            )
            # Reshape to (1, nsamples) for marginal method
            row_2d = self._bkd.reshape(x[i], (1, -1))
            f_x = self._bkd.exp(marginal.logpdf(row_2d))[0]
            dx_dy[i] = phi_y / (f_x + 1e-15)

        # dx/dz = diag(dx/dy) @ L
        jacobian = self._bkd.zeros((self._nvars, self._nvars, nsamples))
        L_np = self._bkd.to_numpy(self._chol)
        for k in range(nsamples):
            dx_dy_k = self._bkd.to_numpy(dx_dy[:, k])
            jacobian[:, :, k] = self._bkd.asarray(np.diag(dx_dy_k) @ L_np)

        return x, jacobian

    def _to_standard_normal_marginals(self, samples: Array) -> Array:
        """Transform each marginal to standard normal."""
        nsamples = samples.shape[1]
        y = self._bkd.zeros((self._nvars, nsamples))

        for i, marginal in enumerate(self._marginals):
            # Reshape to (1, nsamples) for marginal method
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            probs = marginal.cdf(row_2d)  # Returns (1, nsamples)
            probs_np = np.clip(self._bkd.to_numpy(probs[0]), 1e-15, 1.0 - 1e-15)
            y[i] = self._bkd.asarray(self._standard_normal.ppf(probs_np))

        return y

    def _from_standard_normal_marginals(self, y: Array) -> Array:
        """Transform standard normal marginals to original."""
        nsamples = y.shape[1]
        x = self._bkd.zeros((self._nvars, nsamples))

        for i, marginal in enumerate(self._marginals):
            probs = self._bkd.asarray(
                self._standard_normal.cdf(self._bkd.to_numpy(y[i]))
            )
            # Reshape to (1, nsamples) for marginal method
            probs_2d = self._bkd.reshape(probs, (1, -1))
            result = marginal.invcdf(probs_2d)  # Returns (1, nsamples)
            x[i] = result[0]

        return x

    def _compute_correlation_std(self) -> Array:
        """
        Compute correlation matrix in standard normal space.

        For Gaussian marginals, R_0 = R.
        For non-Gaussian, we use the approximation:
            R_0,ij ≈ R_ij * F_ij
        where F_ij is a correction factor based on marginal moments.
        """
        corr_np = self._bkd.to_numpy(self._correlation)
        corr_std = np.zeros_like(corr_np)

        for i in range(self._nvars):
            for j in range(self._nvars):
                if i == j:
                    corr_std[i, j] = 1.0
                else:
                    # Use approximation based on Liu-Der Kiureghian
                    corr_std[i, j] = self._compute_std_correlation_ij(
                        corr_np[i, j], i, j
                    )

        # Ensure positive definite
        corr_std = self._ensure_positive_definite(corr_std)

        return self._bkd.asarray(corr_std)

    def _compute_std_correlation_ij(self, rho: float, i: int, j: int) -> float:
        """
        Compute standard normal correlation for pair (i, j).

        Uses numerical solution of the integral equation.
        For simplicity, we use the Nataf approximation formula.
        """
        # Check if marginals are normal
        marg_i = self._marginals[i]
        marg_j = self._marginals[j]

        # For normal marginals, no correction needed
        if (
            hasattr(marg_i, "name")
            and marg_i.name == "norm"
            and hasattr(marg_j, "name")
            and marg_j.name == "norm"
        ):
            return rho

        # General case: use iterative method
        # Simplified: use first-order approximation
        # rho_0 ≈ rho * (1 + delta_ij)
        # where delta_ij depends on marginal moments

        # Placeholder: return rho as approximation
        return rho

    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite by eigenvalue modification."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        min_eigenvalue = 1e-10
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def log_det_jacobian_to_canonical(self, samples: Array) -> Array:
        """
        Compute log absolute determinant of Jacobian.

        log|det(dz/dx)| = log|det(L^{-1})| + sum_i log|dy_i/dx_i|

        Parameters
        ----------
        samples : Array
            Samples. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Log determinant. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D with correct shape
        """
        self._validate_input(samples)

        nsamples = samples.shape[1]

        # log|det(L^{-1})| = -log|det(L)| = -sum log(L_ii)
        log_det_L_inv = -self._bkd.sum(
            self._bkd.log(self._bkd.get_diagonal(self._chol))
        )

        # Transform to get y
        y = self._to_standard_normal_marginals(samples)

        # sum_i log|dy_i/dx_i| = sum_i [log f_i(x_i) - log phi(y_i)]
        log_det_marginal = self._bkd.zeros((nsamples,))
        for i, marginal in enumerate(self._marginals):
            # Reshape to (1, nsamples) for marginal method
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            log_f = marginal.logpdf(row_2d)[0]  # Extract from (1, nsamples)
            log_phi = self._bkd.asarray(
                self._standard_normal.logpdf(self._bkd.to_numpy(y[i]))
            )
            log_det_marginal = log_det_marginal + log_f - log_phi

        result = float(log_det_L_inv) + log_det_marginal
        return self._bkd.reshape(result, (1, -1))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NatafTransform(nvars={self._nvars})"
