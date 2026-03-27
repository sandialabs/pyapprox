"""Factory pattern for parallel function wrapping.

This module provides ParallelFunctionWrapper for wrapping functions
with parallel batch execution capabilities, and a make_parallel
convenience function.
"""

from typing import Generic, Optional, Union

from pyapprox.interface.parallel.batch_utils import BatchSplitter
from pyapprox.interface.parallel.config import (
    ParallelConfig,
    SequentialBackend,
)
from pyapprox.interface.parallel.protocols import (
    ParallelBackendProtocol,
)
from pyapprox.interface.parallel.tensor_utils import TensorTransfer
from pyapprox.util.backends.protocols import Array, Backend


class ParallelFunctionWrapper(Generic[Array]):
    """Wrapper that adds parallel batch methods to functions.

    Wraps a function and auto-detects available derivative methods
    (jacobian, hvp, whvp) via hasattr, adding parallel batch versions.

    Parameters
    ----------
    function : object
        Function object with bkd(), nvars(), nqoi(), __call__().
        May optionally have jacobian(), hvp(), whvp() methods.
    config : ParallelConfig, optional
        Parallel execution configuration. Default uses joblib with -1 jobs.

    Examples
    --------
    >>> from pyapprox.interface.parallel import make_parallel
    >>> # Wrap a GP with parallel support
    >>> parallel_gp = make_parallel(gp, backend="joblib_processes", n_jobs=4)
    >>> jacobians = parallel_gp.jacobian_batch(samples)
    """

    def __init__(
        self,
        function: object,
        config: Optional[ParallelConfig] = None,
    ) -> None:
        self._function = function
        self._config = config or ParallelConfig()
        self._backend: Union[ParallelBackendProtocol, SequentialBackend] = (
            self._config.get_parallel_backend()
        )
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        """Auto-detect and set up derivative methods via hasattr."""
        if hasattr(self._function, "jacobian"):
            self.jacobian = self._function.jacobian
            self.jacobian_batch = self._jacobian_batch
        if hasattr(self._function, "hvp"):
            self.hvp = self._function.hvp
            self.hvp_batch = self._hvp_batch
        if hasattr(self._function, "whvp"):
            self.whvp = self._function.whvp
            self.whvp_batch = self._whvp_batch
        if hasattr(self._function, "hessian"):
            self.hessian = self._function.hessian
            self.hessian_batch = self._hessian_batch

    def bkd(self) -> Backend[Array]:
        """Return the array backend."""
        return self._function.bkd()  # type: ignore

    def nvars(self) -> int:
        """Return number of input variables."""
        return self._function.nvars()  # type: ignore

    def nqoi(self) -> int:
        """Return number of outputs."""
        return self._function.nqoi()  # type: ignore

    def __call__(self, samples: Array) -> Array:
        """Evaluate function at samples, optionally in parallel.

        Splits samples into chunks, dispatches each chunk to the
        wrapped function's __call__ (preserving vectorization within
        each chunk), and combines results.

        Parameters
        ----------
        samples : Array
            Input samples, shape (nvars, nsamples).

        Returns
        -------
        Array
            Function values, shape (nqoi, nsamples).
        """
        nsamples = samples.shape[1]
        n_workers = self._effective_n_workers()

        # Short-circuit: no parallelism needed
        if n_workers <= 1 or nsamples <= 1:
            return self._function(samples)  # type: ignore

        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        n_chunks = min(n_workers, nsamples)
        chunks = splitter.split_samples(samples, n_chunks)

        # Wrap __call__ for numpy conversion (multiprocessing serialization)
        wrapped_call = transfer.wrap_function(
            self._function.__call__  # type: ignore
        )

        # Convert chunks to numpy for parallel execution
        chunks_np = [transfer.to_numpy(chunk) for chunk in chunks]

        # Execute in parallel
        results_np = self._backend.map(wrapped_call, chunks_np)

        # Convert back and combine along samples axis
        results = [transfer.from_numpy(r) for r in results_np]
        return splitter.combine_outputs(results, axis=1)

    def parallel_backend(self) -> Optional[str]:
        """Return name of parallel backend, or None if sequential."""
        return self._backend.backend_name()

    def n_workers(self) -> int:
        """Return number of parallel workers."""
        return self._config.n_jobs

    def _effective_n_workers(self) -> int:
        """Resolve the effective number of workers.

        Returns
        -------
        int
            Resolved worker count. Returns 1 for SequentialBackend.
            For -1, resolves to os.cpu_count().
        """
        if isinstance(self._backend, SequentialBackend):
            return 1
        n = self._config.n_jobs
        if n == -1:
            import os

            return os.cpu_count() or 1
        return n

    def _jacobian_batch(self, samples: Array) -> Array:
        """Compute jacobians at multiple samples in parallel.

        Parameters
        ----------
        samples : Array
            Samples, shape (nvars, nsamples).

        Returns
        -------
        Array
            Jacobians, shape (nsamples, nqoi, nvars).
        """
        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)

        # Wrap jacobian for numpy conversion
        wrapped_jac = transfer.wrap_function(self._function.jacobian)  # type: ignore

        # Convert samples to numpy for parallel execution
        singles_np = [transfer.to_numpy(s) for s in singles]

        # Execute in parallel
        jacobians_np = self._backend.map(wrapped_jac, singles_np)

        # Convert back and combine
        jacobians = [transfer.from_numpy(j) for j in jacobians_np]
        return splitter.combine_jacobians(jacobians)

    def _hessian_batch(self, samples: Array) -> Array:
        """Compute hessians at multiple samples in parallel.

        Only valid for nqoi == 1.

        Parameters
        ----------
        samples : Array
            Samples, shape (nvars, nsamples).

        Returns
        -------
        Array
            Hessians, shape (nsamples, nvars, nvars).
        """
        if self.nqoi() != 1:
            raise ValueError("hessian_batch only valid for nqoi == 1")

        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)
        wrapped_hess = transfer.wrap_function(self._function.hessian)  # type: ignore
        singles_np = [transfer.to_numpy(s) for s in singles]

        hessians_np = self._backend.map(wrapped_hess, singles_np)

        hessians = [transfer.from_numpy(h) for h in hessians_np]
        return splitter.combine_hessians(hessians)

    def _hvp_batch(self, samples: Array, vecs: Array) -> Array:
        """Compute HVPs at multiple samples in parallel.

        Only valid for nqoi == 1.

        Parameters
        ----------
        samples : Array
            Samples, shape (nvars, nsamples).
        vecs : Array
            Direction vectors, shape (nvars, nsamples).

        Returns
        -------
        Array
            HVP results, shape (nsamples, nvars).
        """
        if self.nqoi() != 1:
            raise ValueError("hvp_batch only valid for nqoi == 1")

        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)
        vec_singles = splitter.split_to_singles(vecs)

        wrapped_hvp = transfer.wrap_starmap_function(self._function.hvp)  # type: ignore

        # Create (sample, vec) pairs as numpy
        pairs_np = [
            (transfer.to_numpy(s), transfer.to_numpy(v))
            for s, v in zip(singles, vec_singles)
        ]

        hvps_np = self._backend.starmap(wrapped_hvp, pairs_np)

        hvps = [transfer.from_numpy(h) for h in hvps_np]
        return splitter.combine_hvps(hvps)

    def _whvp_batch(self, samples: Array, vecs: Array, weights: Array) -> Array:
        """Compute weighted HVPs at multiple samples in parallel.

        Parameters
        ----------
        samples : Array
            Samples, shape (nvars, nsamples).
        vecs : Array
            Direction vectors, shape (nvars, nsamples).
        weights : Array
            Weights for each QoI, shape (nqoi, 1).

        Returns
        -------
        Array
            Weighted HVP results, shape (nsamples, nvars).
        """
        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)
        vec_singles = splitter.split_to_singles(vecs)
        weights_np = transfer.to_numpy(weights)

        def whvp_with_weights(sample_np, vec_np):
            sample = transfer.from_numpy(sample_np)
            vec = transfer.from_numpy(vec_np)
            w = transfer.from_numpy(weights_np)
            result = self._function.whvp(sample, vec, w)  # type: ignore
            return transfer.to_numpy(result)

        pairs_np = [
            (transfer.to_numpy(s), transfer.to_numpy(v))
            for s, v in zip(singles, vec_singles)
        ]

        whvps_np = self._backend.starmap(whvp_with_weights, pairs_np)

        whvps = [transfer.from_numpy(h) for h in whvps_np]
        return splitter.combine_hvps(whvps)


def make_parallel(
    function: object,
    backend: str = "joblib_processes",
    n_jobs: int = -1,
) -> ParallelFunctionWrapper[Array]:
    """Create parallel wrapper for a function.

    Auto-detects jacobian, hvp, whvp methods and adds batch versions.

    Parameters
    ----------
    function : object
        Function object with bkd(), nvars(), nqoi(), __call__().
        May optionally have jacobian(), hvp(), whvp() methods.
    backend : {"joblib_processes", "joblib_threads", "futures", "mpire", "sequential"}
        Parallel execution backend.
    n_jobs : int
        Number of parallel workers. -1 means use all CPUs.

    Returns
    -------
    ParallelFunctionWrapper
        Wrapped function with parallel batch methods.

    Examples
    --------
    >>> from pyapprox.interface.parallel import make_parallel
    >>> parallel_gp = make_parallel(gp, backend="joblib_processes", n_jobs=4)
    >>> jacobians = parallel_gp.jacobian_batch(samples)

    >>> # Or with mpire for progress bars
    >>> parallel_gp = make_parallel(gp, backend="mpire", n_jobs=4)
    """
    config = ParallelConfig(
        backend=backend,  # type: ignore
        n_jobs=n_jobs,
    )
    return ParallelFunctionWrapper(function, config)
