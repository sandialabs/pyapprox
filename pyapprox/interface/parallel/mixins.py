"""Mixin classes for adding parallel batch methods.

This module provides mixins that add parallel batch execution
capabilities to classes that implement single-sample methods.
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


class ParallelJacobianMixin(Generic[Array]):
    """Mixin that adds parallel jacobian_batch from single-sample jacobian.

    Requires the class to implement:
    - bkd() -> Backend[Array]
    - nvars() -> int
    - nqoi() -> int
    - jacobian(sample: Array) -> Array

    Usage
    -----
    class MyFunction(ParallelJacobianMixin[Array]):
        def __init__(self, bkd):
            self._bkd = bkd

        def bkd(self): return self._bkd
        def nvars(self): return 3
        def nqoi(self): return 1

        def jacobian(self, sample):
            # Single sample jacobian implementation
            return self._bkd.ones((1, 3))

    func = MyFunction(bkd)
    func.set_parallel_config(ParallelConfig(backend="mpire", n_jobs=4))
    jacobians = func.jacobian_batch(samples)
    """

    _parallel_config: Optional[ParallelConfig] = None
    _parallel_backend: Optional[Union[ParallelBackendProtocol, SequentialBackend]] = (
        None
    )

    def set_parallel_config(self, config: ParallelConfig) -> None:
        """Set the parallel execution configuration.

        Parameters
        ----------
        config : ParallelConfig
            Configuration for parallel execution.
        """
        self._parallel_config = config
        self._parallel_backend = config.get_parallel_backend()

    def jacobian_batch(self, samples: Array) -> Array:
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
        bkd: Backend[Array] = self.bkd()  # type: ignore
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)

        if self._parallel_backend is None:
            # Sequential execution
            jacobians = [self.jacobian(s) for s in singles]  # type: ignore
            return splitter.combine_jacobians(jacobians)

        # Parallel execution with numpy conversion
        wrapped_jac = transfer.wrap_function(self.jacobian)  # type: ignore
        singles_np = [transfer.to_numpy(s) for s in singles]
        jacobians_np = self._parallel_backend.map(wrapped_jac, singles_np)
        jacobians = [transfer.from_numpy(j) for j in jacobians_np]
        return splitter.combine_jacobians(jacobians)


class ParallelHessianMixin(Generic[Array]):
    """Mixin that adds parallel hessian_batch from single-sample hessian.

    Only valid for nqoi == 1.

    Requires the class to implement:
    - bkd() -> Backend[Array]
    - nvars() -> int
    - nqoi() -> int
    - hessian(sample: Array) -> Array
    """

    _parallel_config: Optional[ParallelConfig] = None
    _parallel_backend: Optional[Union[ParallelBackendProtocol, SequentialBackend]] = (
        None
    )

    def set_parallel_config(self, config: ParallelConfig) -> None:
        """Set the parallel execution configuration."""
        self._parallel_config = config
        self._parallel_backend = config.get_parallel_backend()

    def hessian_batch(self, samples: Array) -> Array:
        """Compute hessians at multiple samples in parallel.

        Parameters
        ----------
        samples : Array
            Samples, shape (nvars, nsamples).

        Returns
        -------
        Array
            Hessians, shape (nsamples, nvars, nvars).
        """
        if self.nqoi() != 1:  # type: ignore
            raise ValueError("hessian_batch only valid for nqoi == 1")

        bkd: Backend[Array] = self.bkd()  # type: ignore
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)

        if self._parallel_backend is None:
            hessians = [self.hessian(s) for s in singles]  # type: ignore
            return splitter.combine_hessians(hessians)

        wrapped_hess = transfer.wrap_function(self.hessian)  # type: ignore
        singles_np = [transfer.to_numpy(s) for s in singles]
        hessians_np = self._parallel_backend.map(wrapped_hess, singles_np)
        hessians = [transfer.from_numpy(h) for h in hessians_np]
        return splitter.combine_hessians(hessians)


class ParallelHVPMixin(Generic[Array]):
    """Mixin that adds parallel hvp_batch from single-sample hvp.

    Only valid for nqoi == 1.

    Requires the class to implement:
    - bkd() -> Backend[Array]
    - nvars() -> int
    - nqoi() -> int
    - hvp(sample: Array, vec: Array) -> Array
    """

    _parallel_config: Optional[ParallelConfig] = None
    _parallel_backend: Optional[Union[ParallelBackendProtocol, SequentialBackend]] = (
        None
    )

    def set_parallel_config(self, config: ParallelConfig) -> None:
        """Set the parallel execution configuration."""
        self._parallel_config = config
        self._parallel_backend = config.get_parallel_backend()

    def hvp_batch(self, samples: Array, vecs: Array) -> Array:
        """Compute HVPs at multiple samples in parallel.

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
        if self.nqoi() != 1:  # type: ignore
            raise ValueError("hvp_batch only valid for nqoi == 1")

        bkd: Backend[Array] = self.bkd()  # type: ignore
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)
        vec_singles = splitter.split_to_singles(vecs)

        if self._parallel_backend is None:
            hvps = [
                self.hvp(s, v)  # type: ignore
                for s, v in zip(singles, vec_singles)
            ]
            return splitter.combine_hvps(hvps)

        wrapped_hvp = transfer.wrap_starmap_function(self.hvp)  # type: ignore
        pairs_np = [
            (transfer.to_numpy(s), transfer.to_numpy(v))
            for s, v in zip(singles, vec_singles)
        ]
        hvps_np = self._parallel_backend.starmap(wrapped_hvp, pairs_np)
        hvps = [transfer.from_numpy(h) for h in hvps_np]
        return splitter.combine_hvps(hvps)


class ParallelWHVPMixin(Generic[Array]):
    """Mixin that adds parallel whvp_batch from single-sample whvp.

    Requires the class to implement:
    - bkd() -> Backend[Array]
    - nvars() -> int
    - nqoi() -> int
    - whvp(sample: Array, vec: Array, weights: Array) -> Array
    """

    _parallel_config: Optional[ParallelConfig] = None
    _parallel_backend: Optional[Union[ParallelBackendProtocol, SequentialBackend]] = (
        None
    )

    def set_parallel_config(self, config: ParallelConfig) -> None:
        """Set the parallel execution configuration."""
        self._parallel_config = config
        self._parallel_backend = config.get_parallel_backend()

    def whvp_batch(self, samples: Array, vecs: Array, weights: Array) -> Array:
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
        bkd: Backend[Array] = self.bkd()  # type: ignore
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)
        vec_singles = splitter.split_to_singles(vecs)

        if self._parallel_backend is None:
            whvps = [
                self.whvp(s, v, weights)  # type: ignore
                for s, v in zip(singles, vec_singles)
            ]
            return splitter.combine_hvps(whvps)

        weights_np = transfer.to_numpy(weights)

        def whvp_with_weights(sample_np, vec_np):
            sample = transfer.from_numpy(sample_np)
            vec = transfer.from_numpy(vec_np)
            w = transfer.from_numpy(weights_np)
            result = self.whvp(sample, vec, w)  # type: ignore
            return transfer.to_numpy(result)

        pairs_np = [
            (transfer.to_numpy(s), transfer.to_numpy(v))
            for s, v in zip(singles, vec_singles)
        ]
        whvps_np = self._parallel_backend.starmap(whvp_with_weights, pairs_np)
        whvps = [transfer.from_numpy(h) for h in whvps_np]
        return splitter.combine_hvps(whvps)
