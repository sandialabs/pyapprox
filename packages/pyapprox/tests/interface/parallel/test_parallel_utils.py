"""Tests for parallel execution utilities."""

import pytest


class TestBatchSplitter:
    """Tests for BatchSplitter."""

    def test_split_samples_basic(self, bkd):
        """Test basic sample splitting."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(bkd)
        samples = bkd.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        chunks = splitter.split_samples(samples, n_chunks=2)

        assert len(chunks) == 2
        assert chunks[0].shape == (2, 2)
        assert chunks[1].shape == (2, 2)

    def test_split_samples_uneven(self, bkd):
        """Test splitting when samples don't divide evenly."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(bkd)
        samples = bkd.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
        )
        chunks = splitter.split_samples(samples, n_chunks=2)

        assert len(chunks) == 2
        # First chunk gets ceiling, second gets remainder
        assert chunks[0].shape[1] + chunks[1].shape[1] == 5

    def test_split_to_singles(self, bkd):
        """Test splitting to single samples."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(bkd)
        samples = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        singles = splitter.split_to_singles(samples)

        assert len(singles) == 3
        for single in singles:
            assert single.shape == (2, 1)

    def test_combine_outputs(self, bkd):
        """Test combining outputs."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(bkd)
        output1 = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        output2 = bkd.asarray([[5.0, 6.0], [7.0, 8.0]])
        combined = splitter.combine_outputs([output1, output2], axis=1)

        assert combined.shape == (2, 4)
        expected = bkd.asarray([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]])
        assert bkd.allclose(combined, expected)

    def test_combine_jacobians(self, bkd):
        """Test combining jacobians into batch format."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(bkd)
        # 3 samples, each jacobian is (2 qoi, 3 vars)
        jac1 = bkd.ones((2, 3))
        jac2 = bkd.ones((2, 3)) * 2
        jac3 = bkd.ones((2, 3)) * 3

        combined = splitter.combine_jacobians([jac1, jac2, jac3])

        assert combined.shape == (3, 2, 3)  # (nsamples, nqoi, nvars)
        assert bkd.allclose(combined[0], bkd.ones((2, 3)))
        assert bkd.allclose(combined[1], bkd.ones((2, 3)) * 2)

    def test_combine_hessians(self, bkd):
        """Test combining hessians into batch format."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(bkd)
        hess1 = bkd.eye(3)
        hess2 = bkd.eye(3) * 2

        combined = splitter.combine_hessians([hess1, hess2])

        assert combined.shape == (2, 3, 3)  # (nsamples, nvars, nvars)

    def test_combine_hvps(self, bkd):
        """Test combining HVP results into batch format."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(bkd)
        # Each HVP result is (nvars, 1)
        hvp1 = bkd.asarray([[1.0], [2.0], [3.0]])
        hvp2 = bkd.asarray([[4.0], [5.0], [6.0]])

        combined = splitter.combine_hvps([hvp1, hvp2])

        assert combined.shape == (2, 3)  # (nsamples, nvars)
        expected = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert bkd.allclose(combined, expected)

    def test_split_more_chunks_than_samples(self, bkd):
        """Test when n_chunks > nsamples."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(bkd)
        samples = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        chunks = splitter.split_samples(samples, n_chunks=10)

        # Should limit to 2 chunks (one per sample)
        assert len(chunks) == 2

    def test_empty_list_errors(self, bkd):
        """Test that empty lists raise errors."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(bkd)

        with pytest.raises(ValueError):
            splitter.combine_outputs([])
        with pytest.raises(ValueError):
            splitter.combine_jacobians([])
        with pytest.raises(ValueError):
            splitter.combine_hessians([])
        with pytest.raises(ValueError):
            splitter.combine_hvps([])


class TestTensorTransfer:
    """Tests for TensorTransfer."""

    def test_to_numpy_from_numpy(self, bkd):
        """Test round-trip conversion."""
        from pyapprox.interface.parallel.tensor_utils import (
            TensorTransfer,
        )

        transfer = TensorTransfer(bkd)
        original = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        numpy_arr = transfer.to_numpy(original)
        back = transfer.from_numpy(numpy_arr)

        assert bkd.allclose(back, original)

    def test_wrap_function(self, bkd):
        """Test function wrapping."""
        from pyapprox.interface.parallel.tensor_utils import (
            TensorTransfer,
        )

        transfer = TensorTransfer(bkd)

        def square(x):
            return x * x

        wrapped = transfer.wrap_function(square)

        # Create test input in numpy format
        import numpy as np

        input_np = np.array([[1.0, 2.0, 3.0]])
        result_np = wrapped(input_np)

        expected = np.array([[1.0, 4.0, 9.0]])
        assert np.allclose(result_np, expected)

    def test_wrap_starmap_function(self, bkd):
        """Test starmap function wrapping."""
        from pyapprox.interface.parallel.tensor_utils import (
            TensorTransfer,
        )

        transfer = TensorTransfer(bkd)

        def add(x, y):
            return x + y

        wrapped = transfer.wrap_starmap_function(add)

        import numpy as np

        x_np = np.array([[1.0, 2.0]])
        y_np = np.array([[3.0, 4.0]])
        result_np = wrapped(x_np, y_np)

        expected = np.array([[4.0, 6.0]])
        assert np.allclose(result_np, expected)
