"""
Gaussian posterior prediction through linear models.

Compute the posterior pushforward (predictive distribution) for linear
observation and prediction models with Gaussian conjugate priors.
"""

from typing import Generic, Optional

from scipy.linalg import eigh as generalized_eigenvalue_decomp

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian


class DenseGaussianPrediction(Generic[Array]):
    """
    Compute the Gaussian posterior pushforward through a prediction model.

    This class computes the predictive distribution for a linear prediction
    model given observations from a linear observation model with Gaussian
    conjugate prior. It uses a generalized eigenvalue decomposition for
    efficient computation.

    For linear models:
        observations = obs_matrix @ parameters + noise
        predictions = pred_matrix @ parameters

    With:
        noise ~ N(0, obs_noise_cov)
        parameters ~ N(prior_mean, prior_cov)

    The predictive distribution is Gaussian:
        predictions | observations ~ N(pred_mean, pred_cov)

    Parameters
    ----------
    obs_matrix : Array
        The observation model matrix. Shape: (nobs, nvars)
    pred_matrix : Array
        The prediction model matrix. Shape: (npred, nvars)
    prior_mean : Array
        The mean of the Gaussian prior. Shape: (nvars, 1)
    prior_covariance : Array
        The covariance of the Gaussian prior. Shape: (nvars, nvars)
    obs_noise_covariance : Array
        The covariance of observation noise. Shape: (nobs, nobs)
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Observation model
    >>> A_obs = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> # Prediction model
    >>> A_pred = np.array([[1.0, 1.0]])
    >>> prior_mean = np.zeros((2, 1))
    >>> prior_cov = np.eye(2)
    >>> noise_cov = 0.1 * np.eye(2)
    >>> predictor = DenseGaussianPrediction(
    ...     A_obs, A_pred, prior_mean, prior_cov, noise_cov, bkd
    ... )
    >>> obs = np.array([[1.0], [1.5]])
    >>> predictor.compute(obs)
    >>> pred_mean = predictor.mean()
    """

    def __init__(
        self,
        obs_matrix: Array,
        pred_matrix: Array,
        prior_mean: Array,
        prior_covariance: Array,
        obs_noise_covariance: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._obs_matrix = obs_matrix
        self._pred_matrix = pred_matrix

        self._nobs, self._nvars = obs_matrix.shape
        self._npred = pred_matrix.shape[0]

        if pred_matrix.shape[1] != self._nvars:
            raise ValueError(
                f"pred_matrix has {pred_matrix.shape[1]} columns, "
                f"expected {self._nvars}"
            )

        if prior_mean.shape != (self._nvars, 1):
            raise ValueError(
                f"prior_mean has wrong shape {prior_mean.shape}, "
                f"expected ({self._nvars}, 1)"
            )
        self._prior_mean = prior_mean

        if prior_covariance.shape != (self._nvars, self._nvars):
            raise ValueError(
                f"prior_covariance has wrong shape {prior_covariance.shape}, "
                f"expected ({self._nvars}, {self._nvars})"
            )
        self._prior_cov = prior_covariance

        if obs_noise_covariance.shape != (self._nobs, self._nobs):
            raise ValueError(
                f"obs_noise_covariance has wrong shape {obs_noise_covariance.shape}, "
                f"expected ({self._nobs}, {self._nobs})"
            )
        self._obs_noise_cov = obs_noise_covariance

        # State
        self._pred_mean: Optional[Array] = None
        self._pred_cov: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def nobs(self) -> int:
        """Return the number of observations."""
        return self._nobs

    def npred(self) -> int:
        """Return the number of predictions."""
        return self._npred

    def compute(self, obs: Array) -> None:
        """
        Compute the predictive mean and covariance given observations.

        This method uses a generalized eigenvalue decomposition to efficiently
        compute the optimal predictive distribution.

        Parameters
        ----------
        obs : Array
            Observations. Shape: (nobs, 1)
        """
        if obs.shape != (self._nobs, 1):
            raise ValueError(
                f"obs has wrong shape {obs.shape}, expected ({self._nobs}, 1)"
            )

        # Step 1: O @ P where P = prior_cov
        OP = self._pred_matrix @ self._prior_cov

        # Step 2: C = O @ P @ A^T (cross-covariance term)
        C = OP @ self._obs_matrix.T

        # Step 3: Pz = O @ P @ O^T (prior pushforward covariance)
        Pz = OP @ self._pred_matrix.T

        # Step 4: Inverse of prior pushforward covariance
        Pz_inv = self._bkd.inv(Pz)

        # Step 5: A = C^T @ Pz^{-1} @ C
        A_mat = C.T @ Pz_inv @ C

        # Step 6: Data covariance = A @ P @ A^T + noise_cov
        data_cov = (
            self._obs_matrix @ self._prior_cov @ self._obs_matrix.T
            + self._obs_noise_cov
        )

        # Step 7: Solve generalized eigenvalue problem A @ v = lambda @ B @ v
        # where B = data_cov
        # Convert to numpy for scipy's generalized eigenvalue decomposition
        A_np = self._bkd.to_numpy(A_mat)
        B_np = self._bkd.to_numpy(data_cov)
        evals, evecs = generalized_eigenvalue_decomp(A_np, B_np)

        # Sort in descending order
        evecs = self._bkd.flip(self._bkd.asarray(evecs), axis=(1,))
        evals = self._bkd.flip(self._bkd.asarray(evals))

        # Truncate to effective rank
        rank = min(self._npred, self._nobs)
        evecs = evecs[:, :rank]
        evals = evals[:rank]

        # Step 8: Compute prediction covariance eigenvectors
        ppf_cov_evecs = C @ evecs

        # Compute residual from prior prediction
        residual = obs - self._obs_matrix @ self._prior_mean

        # Optimal predictive covariance
        self._pred_cov = Pz - ppf_cov_evecs @ ppf_cov_evecs.T

        # Optimal predictive mean
        self._pred_mean = (
            ppf_cov_evecs @ (evecs.T @ residual)
            + self._pred_matrix @ self._prior_mean
        )

    def mean(self) -> Array:
        """
        Return the posterior pushforward mean.

        Returns
        -------
        Array
            Predictive mean. Shape: (npred, 1)

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if self._pred_mean is None:
            raise RuntimeError("Must call compute() first")
        return self._pred_mean

    def covariance(self) -> Array:
        """
        Return the posterior pushforward covariance.

        Returns
        -------
        Array
            Predictive covariance. Shape: (npred, npred)

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if self._pred_cov is None:
            raise RuntimeError("Must call compute() first")
        return self._pred_cov

    def pushforward_variable(self) -> DenseCholeskyMultivariateGaussian[Array]:
        """
        Return the predictive distribution as a Gaussian object.

        Returns
        -------
        DenseCholeskyMultivariateGaussian
            Predictive Gaussian distribution.
        """
        return DenseCholeskyMultivariateGaussian(
            self.mean(), self.covariance(), self._bkd
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DenseGaussianPrediction(nobs={self._nobs}, "
            f"npred={self._npred}, nvars={self._nvars})"
        )
