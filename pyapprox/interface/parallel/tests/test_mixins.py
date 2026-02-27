"""Tests for parallel mixins."""

from typing import Generic

import pytest

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.optional_deps import package_available

HAS_JOBLIB = package_available("joblib")
HAS_MPIRE = package_available("mpire")


class JacobianMixinFunction(Generic[Array]):
    """Test function using ParallelJacobianMixin."""

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        # Add mixin methods dynamically
        self._parallel_config = None
        self._parallel_backend = None

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

    def set_parallel_config(self, config):
        self._parallel_config = config
        self._parallel_backend = config.get_parallel_backend()

    def jacobian_batch(self, samples: Array) -> Array:
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )
        from pyapprox.interface.parallel.tensor_utils import (
            TensorTransfer,
        )

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
    """Test function using ParallelHVPMixin."""

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self._parallel_config = None
        self._parallel_backend = None

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

    def set_parallel_config(self, config):
        self._parallel_config = config
        self._parallel_backend = config.get_parallel_backend()

    def hvp_batch(self, samples: Array, vecs: Array) -> Array:
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )
        from pyapprox.interface.parallel.tensor_utils import (
            TensorTransfer,
        )

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


class TestParallelMixins:
    """Tests for parallel mixins."""

    def test_jacobian_mixin_sequential(self, bkd):
        """Test jacobian mixin without parallel config (sequential)."""
        func = JacobianMixinFunction(bkd)

        samples = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        jacobians = func.jacobian_batch(samples)

        assert jacobians.shape == (3, 1, 2)

        # Verify each jacobian
        for i in range(3):
            sample = samples[:, i : i + 1]
            expected = func.jacobian(sample)
            assert bkd.allclose(jacobians[i], expected, rtol=1e-10)

    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not installed")
    def test_jacobian_mixin_parallel(self, bkd):
        """Test jacobian mixin with parallel config."""
        from pyapprox.interface.parallel.config import ParallelConfig

        func = JacobianMixinFunction(bkd)
        func.set_parallel_config(ParallelConfig(backend="joblib_processes", n_jobs=2))

        samples = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        jacobians = func.jacobian_batch(samples)

        assert jacobians.shape == (3, 1, 2)

        # Verify results match sequential
        func_seq = JacobianMixinFunction(bkd)
        seq_jacobians = func_seq.jacobian_batch(samples)

        assert bkd.allclose(jacobians, seq_jacobians)

    def test_hvp_mixin_sequential(self, bkd):
        """Test HVP mixin without parallel config."""
        func = HVPMixinFunction(bkd)

        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        vecs = bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        hvps = func.hvp_batch(samples, vecs)

        assert hvps.shape == (2, 2)

        # HVP is 2*v
        expected = bkd.asarray([[2.0, 0.0], [0.0, 2.0]])
        assert bkd.allclose(hvps, expected)

    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib not installed")
    def test_hvp_mixin_parallel(self, bkd):
        """Test HVP mixin with parallel config."""
        from pyapprox.interface.parallel.config import ParallelConfig

        func = HVPMixinFunction(bkd)
        func.set_parallel_config(ParallelConfig(backend="joblib_processes", n_jobs=2))

        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        vecs = bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        hvps = func.hvp_batch(samples, vecs)

        # Verify results match sequential
        func_seq = HVPMixinFunction(bkd)
        seq_hvps = func_seq.hvp_batch(samples, vecs)

        assert bkd.allclose(hvps, seq_hvps)

    @pytest.mark.skipif(not HAS_MPIRE, reason="mpire not installed")
    def test_jacobian_mixin_with_mpire(self, bkd):
        """Test jacobian mixin with mpire backend."""

        from pyapprox.interface.parallel.config import ParallelConfig

        func = JacobianMixinFunction(bkd)
        func.set_parallel_config(ParallelConfig(backend="mpire", n_jobs=2))

        samples = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        jacobians = func.jacobian_batch(samples)

        # Verify results match sequential
        func_seq = JacobianMixinFunction(bkd)
        seq_jacobians = func_seq.jacobian_batch(samples)

        assert bkd.allclose(jacobians, seq_jacobians)
