"""
Parallel Gaussian log-likelihood with diagonal covariance.

Provides a parallelized version of DiagonalGaussianLogLikelihood
that uses parallel processing when nprocs > 1.
"""

from typing import Generic, Optional

from numpy.typing import NDArray

from pyapprox.probability.likelihood.gaussian import (
    DiagonalGaussianLogLikelihood,
)
from pyapprox.util.backends.protocols import Array, Backend


class ParallelDiagonalGaussianLogLikelihood(
    DiagonalGaussianLogLikelihood[Array], Generic[Array]
):
    """
    Diagonal Gaussian likelihood with parallel observation processing.

    Inherits all functionality from DiagonalGaussianLogLikelihood,
    adds parallel support for logpdf_vectorized when nprocs > 1.

    Parameters
    ----------
    noise_variances : Array
        Noise variances for each observation. Shape: (nobs,)
    bkd : Backend[Array]
        Computational backend.
    nprocs : int, optional
        Number of parallel workers. Default is 1 (sequential).

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> noise_var = np.array([0.01, 0.02, 0.01])
    >>> # Sequential execution
    >>> lik_seq = ParallelDiagonalGaussianLogLikelihood(noise_var, bkd, nprocs=1)
    >>> # Parallel execution with 4 workers
    >>> lik_par = ParallelDiagonalGaussianLogLikelihood(noise_var, bkd, nprocs=4)
    """

    def __init__(
        self,
        noise_variances: Array,
        bkd: Backend[Array],
        nprocs: int = 1,
    ):
        super().__init__(noise_variances, bkd)
        self._nprocs = nprocs
        self._parallel_backend = None

        if nprocs > 1:
            from pyapprox.interface.parallel.config import (
                ParallelConfig,
            )

            config = ParallelConfig(backend="mpire", n_jobs=nprocs)
            self._parallel_backend = config.get_parallel_backend()

    def nprocs(self) -> int:
        """Return number of parallel workers."""
        return self._nprocs

    def logpdf_vectorized(
        self,
        model_outputs: Array,
        observations: Optional[Array] = None,
    ) -> Array:
        """
        Compute log-likelihood for all combinations.

        When nprocs > 1, splits observation samples into chunks
        and evaluates in parallel using mpire.

        Parameters
        ----------
        model_outputs : Array
            Shape: (nobs, n_model_samples) - must be 2D
        observations : Array, optional
            Shape: (nobs, n_obs_samples) - must be 2D.
            Uses stored observations if None.

        Returns
        -------
        Array
            Shape: (n_model_samples, n_obs_samples)

        Raises
        ------
        ValueError
            If inputs are not 2D
        """
        if observations is None:
            if self._observations is None:
                raise ValueError("Observations not set.")
            observations = self._observations

        self._validate_model_outputs(model_outputs)
        if observations.ndim != 2:
            raise ValueError(
                f"Expected 2D observations array, got {observations.ndim}D"
            )

        if self._nprocs == 1 or self._parallel_backend is None:
            return self._logpdf_vectorized_serial(model_outputs, observations)

        return self._logpdf_vectorized_parallel(model_outputs, observations)

    def _logpdf_vectorized_serial(
        self, model_outputs: Array, observations: Array
    ) -> Array:
        """Serial implementation — delegates to base class."""
        return super().logpdf_vectorized(model_outputs, observations)

    def _logpdf_vectorized_parallel(
        self, model_outputs: Array, observations: Array
    ) -> Array:
        """Parallel implementation of vectorized log-likelihood."""
        from pyapprox.interface.parallel.tensor_utils import (
            TensorTransfer,
        )

        n_obs_samples = observations.shape[1]
        chunk_size = (n_obs_samples + self._nprocs - 1) // self._nprocs

        # Split observations into chunks
        obs_chunks = [
            observations[:, i : i + chunk_size]
            for i in range(0, n_obs_samples, chunk_size)
        ]

        transfer = TensorTransfer(self._bkd)
        model_outputs_np = transfer.to_numpy(model_outputs)

        # Closure that captures model_outputs for parallel execution
        def eval_chunk(obs_chunk_np: NDArray) -> NDArray:
            obs_chunk = transfer.from_numpy(obs_chunk_np)
            model = transfer.from_numpy(model_outputs_np)
            result = self._logpdf_vectorized_serial(model, obs_chunk)
            return transfer.to_numpy(result)

        obs_chunks_np = [transfer.to_numpy(c) for c in obs_chunks]
        results_np = self._parallel_backend.map(eval_chunk, obs_chunks_np)

        # Combine results along observation axis
        results = [transfer.from_numpy(r) for r in results_np]
        return self._bkd.concatenate(results, axis=1)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ParallelDiagonalGaussianLogLikelihood("
            f"nobs={self._nobs}, nprocs={self._nprocs})"
        )
