"""Parallel mixin test functions for benchmarking.

Sum-of-squares functions with manual parallel config wiring,
useful for verifying the parallel mixin pattern
(set_parallel_config / jacobian_batch / hvp_batch).
"""

from typing import Any, Generic

from pyapprox.util.backends.protocols import Array, Backend


class JacobianMixinFunction(Generic[Array]):
    """Sum-of-squares with manual parallel jacobian_batch."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._parallel_config: Any = None
        self._parallel_backend: Any = None

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        return self._bkd.sum(samples**2, axis=0, keepdims=True)

    def jacobian(self, sample: Array) -> Array:
        return 2 * sample.T

    def set_parallel_config(self, config: Any) -> None:
        self._parallel_config = config
        self._parallel_backend = config.get_parallel_backend()

    def jacobian_batch(self, samples: Array) -> Array:
        from pyapprox.interface.parallel.batch_utils import BatchSplitter
        from pyapprox.interface.parallel.tensor_utils import TensorTransfer

        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)

        if self._parallel_backend is None:
            jacobians = [self.jacobian(s) for s in singles]
            return splitter.combine_jacobians(jacobians)

        wrapped_jac = transfer.wrap_function(self.jacobian)
        singles_np = [transfer.to_numpy(s) for s in singles]
        jacobians_np = self._parallel_backend.map(wrapped_jac, singles_np)
        jacobians = [transfer.from_numpy(j) for j in jacobians_np]
        return splitter.combine_jacobians(jacobians)


class HVPMixinFunction(Generic[Array]):
    """Sum-of-squares with manual parallel hvp_batch."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._parallel_config: Any = None
        self._parallel_backend: Any = None

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        return self._bkd.sum(samples**2, axis=0, keepdims=True)

    def hvp(self, sample: Array, vec: Array) -> Array:
        return 2 * vec

    def set_parallel_config(self, config: Any) -> None:
        self._parallel_config = config
        self._parallel_backend = config.get_parallel_backend()

    def hvp_batch(self, samples: Array, vecs: Array) -> Array:
        from pyapprox.interface.parallel.batch_utils import BatchSplitter
        from pyapprox.interface.parallel.tensor_utils import TensorTransfer

        if self.nqoi() != 1:
            raise ValueError("hvp_batch only valid for nqoi == 1")

        bkd = self.bkd()
        splitter = BatchSplitter(bkd)
        transfer = TensorTransfer(bkd)

        singles = splitter.split_to_singles(samples)
        vec_singles = splitter.split_to_singles(vecs)

        if self._parallel_backend is None:
            hvps = [self.hvp(s, v) for s, v in zip(singles, vec_singles)]
            return splitter.combine_hvps(hvps)

        wrapped_hvp = transfer.wrap_starmap_function(self.hvp)
        pairs_np = [
            (transfer.to_numpy(s), transfer.to_numpy(v))
            for s, v in zip(singles, vec_singles)
        ]
        hvps_np = self._parallel_backend.starmap(wrapped_hvp, pairs_np)
        hvps = [transfer.from_numpy(h) for h in hvps_np]
        return splitter.combine_hvps(hvps)
