"""
Parallel Gaussian OED likelihood implementations.

Provides parallelized versions of OED likelihood computations by chunking
over outer samples (observations).
"""

from typing import Generic, List, Optional

import numpy as np

from pyapprox.expdesign.likelihood.gaussian import (
    GaussianOEDOuterLoopLikelihood,
)
from pyapprox.interface.parallel import ParallelConfig
from pyapprox.util.backends.protocols import Array, Backend


def _compute_outer_chunk_logpdf(
    shapes: np.ndarray,
    obs_chunk: np.ndarray,
    base_variances: np.ndarray,
    design_weights: np.ndarray,
) -> np.ndarray:
    """Compute logpdf for a chunk of outer samples.

    Worker function for parallel execution.

    Parameters
    ----------
    shapes : np.ndarray
        Model outputs (inner samples). Shape: (nobs, ninner)
    obs_chunk : np.ndarray
        Observations chunk. Shape: (nobs, chunk_nouter)
    base_variances : np.ndarray
        Base noise variances. Shape: (nobs,)
    design_weights : np.ndarray
        Design weights. Shape: (nobs, 1)

    Returns
    -------
    np.ndarray
        Log-likelihood values. Shape: (ninner, chunk_nouter)
    """
    nobs = shapes.shape[0]
    shapes.shape[1]
    obs_chunk.shape[1]

    # Compute residuals: (nobs, ninner, chunk_nouter)
    residuals = obs_chunk[:, None, :] - shapes[:, :, None]

    # Compute inverse variance
    inv_var = design_weights[:, 0] / base_variances

    # Compute log normalization
    log_det = np.sum(np.log(base_variances)) - np.sum(np.log(design_weights))
    log_norm = -0.5 * nobs * np.log(2 * np.pi) - 0.5 * log_det

    # Compute squared Mahalanobis using einsum
    sqrt_inv_var = np.sqrt(inv_var)
    scaled_res = sqrt_inv_var[:, None, None] * residuals
    squared_dist = np.einsum("ijk,ijk->jk", scaled_res, scaled_res)

    return log_norm - 0.5 * squared_dist  # (ninner, chunk_nouter)


def _compute_outer_chunk_jacobian(
    shapes: np.ndarray,
    obs_chunk: np.ndarray,
    latent_chunk: np.ndarray,
    base_variances: np.ndarray,
    design_weights: np.ndarray,
) -> np.ndarray:
    """Compute jacobian for a chunk of outer samples.

    Worker function for parallel execution.

    Parameters
    ----------
    shapes : np.ndarray
        Model outputs (inner samples). Shape: (nobs, ninner)
    obs_chunk : np.ndarray
        Observations chunk. Shape: (nobs, chunk_nouter)
    latent_chunk : np.ndarray or None
        Latent samples chunk. Shape: (nobs, chunk_nouter) or None
    base_variances : np.ndarray
        Base noise variances. Shape: (nobs,)
    design_weights : np.ndarray
        Design weights. Shape: (nobs, 1)

    Returns
    -------
    np.ndarray
        Jacobian values. Shape: (ninner, chunk_nouter, nobs)
    """
    shapes.shape[0]
    shapes.shape[1]
    obs_chunk.shape[1]

    # Compute residuals: (nobs, ninner, chunk_nouter)
    residuals = obs_chunk[:, None, :] - shapes[:, :, None]

    # Component 1: Quadratic term
    jac = -0.5 * residuals**2 / base_variances[:, None, None]

    # Component 2: Determinant term
    det_term = 0.5 / design_weights[:, :, None]  # (nobs, 1, 1)
    jac = jac + det_term

    # Component 3: Reparameterization term
    if latent_chunk is not None:
        reparam_term = 0.5 * (
            residuals
            * latent_chunk[:, None, :]
            / (
                np.sqrt(base_variances[:, None, None])
                * np.sqrt(design_weights[:, :, None])
            )
        )
        jac = jac + reparam_term

    # Transpose from (nobs, ninner, chunk_nouter) to (ninner, chunk_nouter, nobs)
    return np.transpose(jac, (1, 2, 0))


class ParallelGaussianOEDInnerLoopLikelihood(Generic[Array]):
    """
    Parallel inner loop likelihood for Gaussian OED.

    Parallelizes computation over outer samples (observations) by chunking.
    Each worker computes the likelihood matrix for a subset of outer samples.

    Parameters
    ----------
    noise_variances : Array
        Base noise variances. Shape: (nobs,)
    bkd : Backend[Array]
        Computational backend.
    parallel_config : ParallelConfig, optional
        Parallel execution configuration.
    """

    def __init__(
        self,
        noise_variances: Array,
        bkd: Backend[Array],
        parallel_config: Optional[ParallelConfig] = None,
    ):
        self._bkd = bkd
        self._base_variances = noise_variances
        self._nobs = noise_variances.shape[0]

        # Parallel config
        self._parallel_config = parallel_config or ParallelConfig(
            backend="sequential", n_jobs=1
        )

        # State
        self._shapes: Optional[Array] = None
        self._observations: Optional[Array] = None
        self._latent_samples: Optional[Array] = None
        self._design_weights: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observation locations."""
        return self._nobs

    def ninner(self) -> int:
        """Number of inner (prior) samples."""
        if self._shapes is None:
            raise ValueError("Must call set_shapes first")
        return self._shapes.shape[1]

    def nouter(self) -> int:
        """Number of outer (observation) samples."""
        if self._observations is None:
            raise ValueError("Must call set_observations first")
        return self._observations.shape[1]

    def set_parallel_config(self, config: ParallelConfig) -> None:
        """Set parallel execution configuration."""
        self._parallel_config = config

    def set_shapes(self, shapes: Array) -> None:
        """Set model outputs for inner samples."""
        if shapes.ndim != 2:
            raise ValueError(f"shapes must be 2D, got {shapes.ndim}D")
        if shapes.shape[0] != self._nobs:
            raise ValueError(
                f"shapes has wrong number of observations: "
                f"expected {self._nobs}, got {shapes.shape[0]}"
            )
        self._shapes = shapes

    def set_observations(self, obs: Array) -> None:
        """Set artificial observations for outer samples."""
        if obs.ndim != 2:
            raise ValueError(f"obs must be 2D, got {obs.ndim}D")
        if obs.shape[0] != self._nobs:
            raise ValueError(
                f"obs has wrong number of observations: "
                f"expected {self._nobs}, got {obs.shape[0]}"
            )
        self._observations = obs

    def set_latent_samples(self, latent_samples: Array) -> None:
        """Set latent samples for reparameterization trick."""
        if self._observations is None:
            raise ValueError("Must call set_observations first")
        if latent_samples.shape != self._observations.shape:
            raise ValueError(
                f"latent_samples shape {latent_samples.shape} must match "
                f"observations {self._observations.shape}"
            )
        self._latent_samples = latent_samples

    def set_design_weights(self, weights: Array) -> None:
        """Set design weights."""
        if weights.shape != (self._nobs, 1):
            raise ValueError(
                f"weights must have shape ({self._nobs}, 1), got {weights.shape}"
            )
        self._design_weights = weights

    def _get_n_chunks(self) -> int:
        """Get number of chunks for parallel processing."""
        n_jobs = self._parallel_config.n_jobs
        if n_jobs == -1:
            import os

            n_jobs = os.cpu_count() or 1
        return max(1, n_jobs)

    def _split_outer(self, n_chunks: int) -> List[tuple]:
        """Split observations and latent samples into chunks along outer dimension.

        Returns list of (obs_chunk, latent_chunk) tuples.
        """
        obs_np = self._bkd.to_numpy(self._observations)
        nouter = obs_np.shape[1]
        chunk_size = (nouter + n_chunks - 1) // n_chunks

        latent_np = (
            self._bkd.to_numpy(self._latent_samples)
            if self._latent_samples is not None
            else None
        )

        chunks = []
        for i in range(0, nouter, chunk_size):
            end = min(i + chunk_size, nouter)
            obs_chunk = obs_np[:, i:end]
            latent_chunk = latent_np[:, i:end] if latent_np is not None else None
            chunks.append((obs_chunk, latent_chunk))
        return chunks

    def logpdf_matrix(self, design_weights: Array) -> Array:
        """
        Compute full log-likelihood matrix in parallel over outer samples.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Log-likelihood matrix. Shape: (ninner, nouter)
        """
        self.set_design_weights(design_weights)

        if self._shapes is None or self._observations is None:
            raise ValueError("Must call set_shapes and set_observations first")

        n_chunks = self._get_n_chunks()

        # If sequential or single chunk, use non-parallel version
        if n_chunks <= 1 or self._parallel_config.backend == "sequential":
            return self._logpdf_matrix_sequential(design_weights)

        # Split outer samples into chunks
        outer_chunks = self._split_outer(n_chunks)

        # Convert to numpy for parallel workers
        shapes_np = self._bkd.to_numpy(self._shapes)
        base_var_np = self._bkd.to_numpy(self._base_variances)
        weights_np = self._bkd.to_numpy(design_weights)

        # Create argument tuples for each chunk
        args_list = [
            (shapes_np, obs_chunk, base_var_np, weights_np)
            for obs_chunk, _ in outer_chunks
        ]

        # Execute in parallel using starmap for proper argument unpacking
        backend = self._parallel_config.get_parallel_backend()
        results = backend.starmap(_compute_outer_chunk_logpdf, args_list)

        # Combine results along outer dimension (axis=1)
        combined = np.concatenate(results, axis=1)
        return self._bkd.asarray(combined)

    def _logpdf_matrix_sequential(self, design_weights: Array) -> Array:
        """Sequential implementation for comparison."""
        # Compute residuals
        residuals = self._observations[:, None, :] - self._shapes[:, :, None]

        inv_var = self._design_weights[:, 0] / self._base_variances
        log_det = float(
            self._bkd.sum(self._bkd.log(self._base_variances))
            - self._bkd.sum(self._bkd.log(self._design_weights))
        )
        log_norm = float(-0.5 * self._nobs * np.log(2 * np.pi) - 0.5 * log_det)

        sqrt_inv_var = self._bkd.sqrt(inv_var)
        scaled_res = sqrt_inv_var[:, None, None] * residuals
        squared_dist = self._bkd.einsum("ijk,ijk->jk", scaled_res, scaled_res)

        return log_norm - 0.5 * squared_dist

    def jacobian_matrix(self, design_weights: Array) -> Array:
        """
        Jacobian of log-likelihood matrix w.r.t. design weights (parallel).

        Parallelizes over outer samples.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian tensor. Shape: (ninner, nouter, nobs)
        """
        self.set_design_weights(design_weights)

        if self._shapes is None or self._observations is None:
            raise ValueError("Must call set_shapes and set_observations first")

        n_chunks = self._get_n_chunks()

        # If sequential or single chunk, use non-parallel version
        if n_chunks <= 1 or self._parallel_config.backend == "sequential":
            return self._jacobian_matrix_sequential(design_weights)

        # Split outer samples into chunks
        outer_chunks = self._split_outer(n_chunks)

        # Convert to numpy for parallel workers
        shapes_np = self._bkd.to_numpy(self._shapes)
        base_var_np = self._bkd.to_numpy(self._base_variances)
        weights_np = self._bkd.to_numpy(design_weights)

        # Create argument tuples for each chunk
        args_list = [
            (shapes_np, obs_chunk, latent_chunk, base_var_np, weights_np)
            for obs_chunk, latent_chunk in outer_chunks
        ]

        # Execute in parallel using starmap for proper argument unpacking
        backend = self._parallel_config.get_parallel_backend()
        results = backend.starmap(_compute_outer_chunk_jacobian, args_list)

        # Combine results along outer dimension (axis=1)
        combined = np.concatenate(results, axis=1)
        return self._bkd.asarray(combined)

    def _jacobian_matrix_sequential(self, design_weights: Array) -> Array:
        """Sequential implementation for comparison."""
        # Compute residuals
        residuals = self._observations[:, None, :] - self._shapes[:, :, None]

        # Component 1: Quadratic term
        jac = -0.5 * residuals**2 / self._base_variances[:, None, None]

        # Component 2: Determinant term
        det_term = 0.5 / self._design_weights[:, :, None]
        jac = jac + det_term

        # Component 3: Reparameterization term
        if self._latent_samples is not None:
            reparam_term = 0.5 * (
                residuals
                * self._latent_samples[:, None, :]
                / (
                    self._bkd.sqrt(self._base_variances[:, None, None])
                    * self._bkd.sqrt(self._design_weights[:, :, None])
                )
            )
            jac = jac + reparam_term

        return self._bkd.transpose(jac, (1, 2, 0))

    def create_outer_loop_likelihood(self):
        """Create a paired outer loop likelihood."""
        return GaussianOEDOuterLoopLikelihood(self._base_variances, self._bkd)
