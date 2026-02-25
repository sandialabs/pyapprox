"""
Low-rank Laplace posterior approximation.

For high-dimensional problems where only a few directions are informed
by the data, a low-rank approximation of the posterior covariance
can be much more efficient than a full-rank approximation.
"""

from typing import Callable, Generic, Optional

import numpy as np

from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.probability.protocols import SqrtCovarianceOperatorProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg import randomized_symmetric_eigendecomposition


class LowRankLaplacePosterior(Generic[Array]):
    r"""
    Low-rank Laplace approximation to the posterior.

    For high-dimensional problems, the data often only informs a
    low-dimensional subspace of parameter space. This class computes
    a low-rank approximation of the posterior covariance using
    randomized eigenvalue decomposition.

    The posterior covariance is approximated as:

    .. math::
        \Sigma_{post} \approx L V_r D_r V_r^T L^T

    where:
    - L is the prior Cholesky factor
    - V_r are the top r eigenvectors of the prior-conditioned Hessian
    - D_r = diag(1/(1 + lambda_i)) with lambda_i the eigenvalues

    Parameters
    ----------
    map_point : Array
        The MAP point. Shape: (nvars, 1)
    prior_sqrt : SqrtCovarianceOperatorProtocol
        Square-root covariance operator for the prior.
    apply_conditioned_hessian : Callable[[Array], Array]
        Function that applies the prior-conditioned misfit Hessian.
        Typically from PriorConditionedHessianMatVec.apply.
    rank : int
        Rank of the low-rank approximation.
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    The low-rank approximation is most accurate when:
    1. The data only constrains a few directions in parameter space
    2. The eigenvalues of the prior-conditioned Hessian decay rapidly

    The rank should be chosen such that the eigenvalues beyond the
    retained ones are close to zero.
    """

    def __init__(
        self,
        map_point: Array,
        prior_sqrt: SqrtCovarianceOperatorProtocol[Array],
        apply_conditioned_hessian: Callable[[Array], Array],
        rank: int,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._nvars = prior_sqrt.nvars()
        self._rank = rank

        if map_point.shape != (self._nvars, 1):
            raise ValueError(
                f"map_point has wrong shape {map_point.shape}, "
                f"expected ({self._nvars}, 1)"
            )
        self._map_point = map_point

        if rank > self._nvars:
            raise ValueError(f"rank ({rank}) cannot exceed nvars ({self._nvars})")

        self._prior_sqrt = prior_sqrt
        self._apply_conditioned_hessian = apply_conditioned_hessian

        # State
        self._Ur: Optional[Array] = None  # Eigenvectors
        self._Sr: Optional[Array] = None  # Eigenvalues
        self._post_cov_sqrt: Optional[Array] = None
        self._computed = False

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def rank(self) -> int:
        """Return the rank of the approximation."""
        return self._rank

    def compute(
        self,
        noversampling: int = 10,
        npower_iters: int = 1,
    ) -> None:
        """
        Compute the low-rank Laplace approximation using randomized SVD.

        Parameters
        ----------
        noversampling : int, default=10
            Number of additional random samples beyond the rank for
            improved accuracy.
        npower_iters : int, default=1
            Number of power iterations for improved accuracy when
            eigenvalues decay slowly.
        """
        # Randomized eigenvalue decomposition of the prior-conditioned Hessian
        self._Sr, self._Ur = randomized_symmetric_eigendecomposition(
            apply_operator=self._apply_conditioned_hessian,
            nvars=self._nvars,
            rank=self._rank,
            bkd=self._bkd,
            noversampling=noversampling,
            npower_iters=npower_iters,
        )

        # Compute posterior covariance sqrt factor
        # Post cov = L @ U_r @ diag(1/(1+lambda)) @ U_r^T @ L^T
        # Sqrt factor = L @ U_r @ diag(1/sqrt(1+lambda))
        P = 1.0 / self._bkd.sqrt(self._Sr + 1.0)
        self._post_cov_sqrt = self._prior_sqrt.apply(
            self._Ur @ (P[:, None] * self._Ur.T)
        )
        self._computed = True

    def posterior_mean(self) -> Array:
        """
        Return the posterior mean (MAP point).

        Returns
        -------
        Array
            Posterior mean. Shape: (nvars, 1)
        """
        return self._map_point

    def posterior_covariance(self) -> Array:
        """
        Return the posterior covariance (full matrix reconstruction).

        Note: This reconstructs the full dense covariance matrix.
        For high-dimensional problems, prefer using covariance_diagonal()
        or sampling with rvs().

        Returns
        -------
        Array
            Posterior covariance. Shape: (nvars, nvars)

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if not self._computed:
            raise RuntimeError("Must call compute() first")
        return self._post_cov_sqrt @ self._post_cov_sqrt.T

    def covariance_diagonal(self) -> Array:
        """
        Return the diagonal of the posterior covariance (marginal variances).

        This is computed efficiently without forming the full covariance matrix.

        Returns
        -------
        Array
            Marginal variances. Shape: (nvars,)

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if not self._computed:
            raise RuntimeError("Must call compute() first")

        # Prior variance diagonal
        # For Cholesky L: var_i = sum_j L_{ij}^2
        # We approximate by computing L @ U_r
        LUr = self._prior_sqrt.apply(self._Ur)

        # Scaling factor: lambda / (1 + lambda)
        D = self._Sr / (1.0 + self._Sr)

        # Variance reduction from low-rank update
        var_reduction = self._bkd.sum(LUr**2 * D, axis=1)

        # Need prior diagonal - we compute it via identity samples
        # For now, return just the low-rank contribution
        # This is an approximation for the diagonal
        identity_samples = self._bkd.eye(self._nvars)
        L_full = self._prior_sqrt.apply(identity_samples)
        prior_var = self._bkd.sum(L_full**2, axis=1)

        return prior_var - var_reduction

    def eigenvalues(self) -> Array:
        """
        Return the eigenvalues of the prior-conditioned Hessian.

        These indicate how much the data informs each direction:
        - Large eigenvalue: Data strongly constrains this direction
        - Small eigenvalue: Data provides little information

        Returns
        -------
        Array
            Top eigenvalues. Shape: (rank,)

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if not self._computed:
            raise RuntimeError("Must call compute() first")
        return self._Sr

    def eigenvectors(self) -> Array:
        """
        Return the eigenvectors of the prior-conditioned Hessian.

        These are the data-informed directions in parameter space,
        in the original (not prior-whitened) coordinates.

        Returns
        -------
        Array
            Top eigenvectors. Shape: (nvars, rank)

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if not self._computed:
            raise RuntimeError("Must call compute() first")
        return self._Ur

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples from the approximate posterior.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Samples from the posterior. Shape: (nvars, nsamples)

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if not self._computed:
            raise RuntimeError("Must call compute() first")

        std_normal = self._bkd.asarray(
            np.random.normal(0, 1, (self._nvars, nsamples)).astype(np.float64)
        )
        return self._post_cov_sqrt @ std_normal + self._map_point

    def posterior_variable(self) -> DenseCholeskyMultivariateGaussian[Array]:
        """
        Return the posterior as a Gaussian distribution object.

        Note: This constructs the full covariance matrix.

        Returns
        -------
        DenseCholeskyMultivariateGaussian
            Posterior Gaussian distribution.

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if not self._computed:
            raise RuntimeError("Must call compute() first")
        return DenseCholeskyMultivariateGaussian(
            self.posterior_mean(),
            self.posterior_covariance(),
            self._bkd,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"LowRankLaplacePosterior(nvars={self._nvars}, rank={self._rank})"
