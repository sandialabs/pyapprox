"""Factory pattern for parallel function wrapping.

This module provides ParallelFunctionWrapper for wrapping functions
with parallel batch execution capabilities, and a make_parallel
convenience function.

Both are deprecated and scheduled for removal: an evaluator built from
``pyapprox.interface.evaluation`` parallelizes the same work and can
also report progress, return per-sample failures as data, and account
for measured cost. See :class:`ParallelFunctionWrapper` for the
equivalent composition. The rest of this subpackage is not deprecated --
``ParallelConfig`` and the backends it constructs are used by the
expdesign likelihoods and objectives independently of this module.
"""

from typing import Generic, Literal, Optional, Union

from pyapprox.interface.functions.derivatives import (
    Derivatives,
    HessianFn,
    HVPFn,
    JacobianFn,
    WHVPFn,
)
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
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

    .. deprecated::
        Scheduled for removal. Compose an evaluator instead::

            from pyapprox.interface.evaluation import (
                CallableMarshaller, Evaluator, blocking, process_dispatcher,
            )

            marshaller = CallableMarshaller(
                fn=fn, bkd=bkd, nvars=nvars, nqoi=nqoi, samples_per_task=1
            )
            model = blocking(
                Evaluator(marshaller, process_dispatcher(marshaller.run, 4))
            )

        The result satisfies the same protocol and evaluates in
        parallel, and additionally reports progress while work is in
        flight, returns per-sample failures as data rather than raising,
        and accumulates measured cost. This wrapper can do none of
        those: it blocks until the whole batch finishes, so a failure
        anywhere loses the batch and there is nothing to ask about
        meanwhile.

    Wraps a function and reads its derivative capability from its
    ``Derivatives`` bundle: single-sample fields are forwarded
    unchanged, and a parallel batch version is added for each populated
    field.

    Parameters
    ----------
    function : ObjectiveProtocol[Array]
        Function object with bkd(), nvars(), nqoi(), __call__() and
        derivatives(). Its derivative capability is read from its
        ``Derivatives`` bundle; a derivative-free function participates
        by returning ``Derivatives.none()``.
    config : ParallelConfig, optional
        Parallel execution configuration. Default uses joblib with -1 jobs.

    Examples
    --------
    >>> from pyapprox.interface.parallel import make_parallel
    >>> # Wrap a GP with parallel support
    >>> parallel_gp = make_parallel(gp, backend="joblib_processes", n_jobs=4)
    >>> jacobians = parallel_gp.derivatives().jacobian_batch(samples)
    """

    def __init__(
        self,
        function: ObjectiveProtocol[Array],
        config: Optional[ParallelConfig] = None,
    ) -> None:
        if not isinstance(function, ObjectiveProtocol):
            raise TypeError(
                "function must satisfy ObjectiveProtocol (a "
                "FunctionProtocol exposing derivatives()), got "
                f"{type(function).__name__}. If the function has no "
                "derivative capability, add a derivatives() method "
                "returning Derivatives.none()."
            )
        self._function = function
        self._config = config or ParallelConfig()
        self._backend: Union[ParallelBackendProtocol, SequentialBackend] = (
            self._config.get_parallel_backend()
        )
        # Mirror the wrapped function's capability: forward the
        # single-sample fields unchanged and add a parallel batch form
        # for each populated field.
        fd = function.derivatives()
        self._function_jac: Optional[JacobianFn[Array]] = fd.jacobian
        self._function_hvp: Optional[HVPFn[Array]] = fd.hvp
        self._function_whvp: Optional[WHVPFn[Array]] = fd.whvp
        self._function_hessian: Optional[HessianFn[Array]] = fd.hessian
        self._derivs: Derivatives[Array] = Derivatives(
            jacobian=fd.jacobian,
            jacobian_batch=None
            if fd.jacobian is None
            else self._jacobian_batch,
            hvp=fd.hvp,
            hvp_batch=None if fd.hvp is None else self._hvp_batch,
            whvp=fd.whvp,
            whvp_batch=None if fd.whvp is None else self._whvp_batch,
            hessian=fd.hessian,
            hessian_batch=None
            if fd.hessian is None
            else self._hessian_batch,
            inexact=fd.inexact,
        )

    def derivatives(self) -> Derivatives[Array]:
        """Return the bundle with parallel batch fields added."""
        return self._derivs

    def bkd(self) -> Backend[Array]:
        """Return the array backend."""
        return self._function.bkd()

    def nvars(self) -> int:
        """Return number of input variables."""
        return self._function.nvars()

    def nqoi(self) -> int:
        """Return number of outputs."""
        return self._function.nqoi()

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
            return self._function(samples)

        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        n_chunks = min(n_workers, nsamples)
        chunks = splitter.split_samples(samples, n_chunks)

        # Wrap __call__ for numpy conversion (multiprocessing serialization)
        wrapped_call = transfer.wrap_function(self._function.__call__)

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
        function_jac = self._function_jac
        if function_jac is None:
            raise RuntimeError(
                "jacobian is unavailable; check derivatives() before calling"
            )
        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)

        # Wrap jacobian for numpy conversion
        wrapped_jac = transfer.wrap_function(function_jac)

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
        function_hessian = self._function_hessian
        if function_hessian is None:
            raise RuntimeError(
                "hessian is unavailable; check derivatives() before calling"
            )

        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)
        wrapped_hess = transfer.wrap_function(function_hessian)
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
        function_hvp = self._function_hvp
        if function_hvp is None:
            raise RuntimeError(
                "hvp is unavailable; check derivatives() before calling"
            )

        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)
        vec_singles = splitter.split_to_singles(vecs)

        wrapped_hvp = transfer.wrap_starmap_function(function_hvp)

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
        function_whvp = self._function_whvp
        if function_whvp is None:
            raise RuntimeError(
                "whvp is unavailable; check derivatives() before calling"
            )
        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)
        vec_singles = splitter.split_to_singles(vecs)
        weights_np = transfer.to_numpy(weights)

        # TensorTransfer types the numpy side with the same Array
        # variable, so the closure is annotated to match.
        def whvp_with_weights(sample_np: Array, vec_np: Array) -> Array:
            sample = transfer.from_numpy(sample_np)
            vec = transfer.from_numpy(vec_np)
            w = transfer.from_numpy(weights_np)
            result = function_whvp(sample, vec, w)
            return transfer.to_numpy(result)

        pairs_np = [
            (transfer.to_numpy(s), transfer.to_numpy(v))
            for s, v in zip(singles, vec_singles)
        ]

        whvps_np = self._backend.starmap(whvp_with_weights, pairs_np)

        whvps = [transfer.from_numpy(h) for h in whvps_np]
        return splitter.combine_hvps(whvps)


def make_parallel(
    function: ObjectiveProtocol[Array],
    backend: Literal[
        "joblib_processes",
        "joblib_threads",
        "futures",
        "mpire",
        "sequential",
    ] = "joblib_processes",
    n_jobs: int = -1,
) -> ParallelFunctionWrapper[Array]:
    """Create parallel wrapper for a function.

    .. deprecated::
        Scheduled for removal; see :class:`ParallelFunctionWrapper` for
        the evaluator composition that replaces it and what that buys.

    Reads jacobian, hvp, whvp, hessian capability from the function's
    ``Derivatives`` bundle and adds parallel batch versions.

    Parameters
    ----------
    function : ObjectiveProtocol[Array]
        Function object with bkd(), nvars(), nqoi(), __call__() and
        derivatives(). Its derivative capability is read from its
        ``Derivatives`` bundle; a derivative-free function participates
        by returning ``Derivatives.none()``.
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
    >>> jacobians = parallel_gp.derivatives().jacobian_batch(samples)

    >>> # Or with mpire for progress bars
    >>> parallel_gp = make_parallel(gp, backend="mpire", n_jobs=4)
    """
    config = ParallelConfig(
        backend=backend,
        n_jobs=n_jobs,
    )
    return ParallelFunctionWrapper(function, config)
