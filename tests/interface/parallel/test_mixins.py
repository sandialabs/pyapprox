"""Tests for parallel mixins."""

import pytest

from pyapprox.util.optional_deps import package_available
from pyapprox_benchmarks.functions.algebraic.parallel_mixin_fixtures import (
    HVPMixinFunction,
    JacobianMixinFunction,
)

HAS_JOBLIB = package_available("joblib")
HAS_MPIRE = package_available("mpire")


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
