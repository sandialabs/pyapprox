"""Tests for parallel execution utilities."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests


class TestBatchSplitter(Generic[Array], unittest.TestCase):
    """Tests for BatchSplitter - NOT run directly."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_split_samples_basic(self):
        """Test basic sample splitting."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(self._bkd)
        samples = self._bkd.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        chunks = splitter.split_samples(samples, n_chunks=2)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].shape, (2, 2))
        self.assertEqual(chunks[1].shape, (2, 2))

    def test_split_samples_uneven(self):
        """Test splitting when samples don't divide evenly."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(self._bkd)
        samples = self._bkd.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
        )
        chunks = splitter.split_samples(samples, n_chunks=2)

        self.assertEqual(len(chunks), 2)
        # First chunk gets ceiling, second gets remainder
        self.assertEqual(chunks[0].shape[1] + chunks[1].shape[1], 5)

    def test_split_to_singles(self):
        """Test splitting to single samples."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(self._bkd)
        samples = self._bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        singles = splitter.split_to_singles(samples)

        self.assertEqual(len(singles), 3)
        for single in singles:
            self.assertEqual(single.shape, (2, 1))

    def test_combine_outputs(self):
        """Test combining outputs."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(self._bkd)
        output1 = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        output2 = self._bkd.asarray([[5.0, 6.0], [7.0, 8.0]])
        combined = splitter.combine_outputs([output1, output2], axis=1)

        self.assertEqual(combined.shape, (2, 4))
        expected = self._bkd.asarray(
            [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]
        )
        self.assertTrue(self._bkd.allclose(combined, expected))

    def test_combine_jacobians(self):
        """Test combining jacobians into batch format."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(self._bkd)
        # 3 samples, each jacobian is (2 qoi, 3 vars)
        jac1 = self._bkd.ones((2, 3))
        jac2 = self._bkd.ones((2, 3)) * 2
        jac3 = self._bkd.ones((2, 3)) * 3

        combined = splitter.combine_jacobians([jac1, jac2, jac3])

        self.assertEqual(combined.shape, (3, 2, 3))  # (nsamples, nqoi, nvars)
        self.assertTrue(
            self._bkd.allclose(combined[0], self._bkd.ones((2, 3)))
        )
        self.assertTrue(
            self._bkd.allclose(combined[1], self._bkd.ones((2, 3)) * 2)
        )

    def test_combine_hessians(self):
        """Test combining hessians into batch format."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(self._bkd)
        hess1 = self._bkd.eye(3)
        hess2 = self._bkd.eye(3) * 2

        combined = splitter.combine_hessians([hess1, hess2])

        self.assertEqual(combined.shape, (2, 3, 3))  # (nsamples, nvars, nvars)

    def test_combine_hvps(self):
        """Test combining HVP results into batch format."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(self._bkd)
        # Each HVP result is (nvars, 1)
        hvp1 = self._bkd.asarray([[1.0], [2.0], [3.0]])
        hvp2 = self._bkd.asarray([[4.0], [5.0], [6.0]])

        combined = splitter.combine_hvps([hvp1, hvp2])

        self.assertEqual(combined.shape, (2, 3))  # (nsamples, nvars)
        expected = self._bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertTrue(self._bkd.allclose(combined, expected))

    def test_split_more_chunks_than_samples(self):
        """Test when n_chunks > nsamples."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(self._bkd)
        samples = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        chunks = splitter.split_samples(samples, n_chunks=10)

        # Should limit to 2 chunks (one per sample)
        self.assertEqual(len(chunks), 2)

    def test_empty_list_errors(self):
        """Test that empty lists raise errors."""
        from pyapprox.interface.parallel.batch_utils import (
            BatchSplitter,
        )

        splitter = BatchSplitter(self._bkd)

        with self.assertRaises(ValueError):
            splitter.combine_outputs([])
        with self.assertRaises(ValueError):
            splitter.combine_jacobians([])
        with self.assertRaises(ValueError):
            splitter.combine_hessians([])
        with self.assertRaises(ValueError):
            splitter.combine_hvps([])


class TestTensorTransfer(Generic[Array], unittest.TestCase):
    """Tests for TensorTransfer - NOT run directly."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_to_numpy_from_numpy(self):
        """Test round-trip conversion."""
        from pyapprox.interface.parallel.tensor_utils import (
            TensorTransfer,
        )

        transfer = TensorTransfer(self._bkd)
        original = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        numpy_arr = transfer.to_numpy(original)
        back = transfer.from_numpy(numpy_arr)

        self.assertTrue(self._bkd.allclose(back, original))

    def test_wrap_function(self):
        """Test function wrapping."""
        from pyapprox.interface.parallel.tensor_utils import (
            TensorTransfer,
        )

        transfer = TensorTransfer(self._bkd)

        def square(x):
            return x * x

        wrapped = transfer.wrap_function(square)

        # Create test input in numpy format
        import numpy as np

        input_np = np.array([[1.0, 2.0, 3.0]])
        result_np = wrapped(input_np)

        expected = np.array([[1.0, 4.0, 9.0]])
        self.assertTrue(np.allclose(result_np, expected))

    def test_wrap_starmap_function(self):
        """Test starmap function wrapping."""
        from pyapprox.interface.parallel.tensor_utils import (
            TensorTransfer,
        )

        transfer = TensorTransfer(self._bkd)

        def add(x, y):
            return x + y

        wrapped = transfer.wrap_starmap_function(add)

        import numpy as np

        x_np = np.array([[1.0, 2.0]])
        y_np = np.array([[3.0, 4.0]])
        result_np = wrapped(x_np, y_np)

        expected = np.array([[4.0, 6.0]])
        self.assertTrue(np.allclose(result_np, expected))


class TestBatchSplitterNumpy(TestBatchSplitter[NDArray[Any]]):
    """NumPy backend tests for BatchSplitter."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBatchSplitterTorch(TestBatchSplitter[torch.Tensor]):
    """PyTorch backend tests for BatchSplitter."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestTensorTransferNumpy(TestTensorTransfer[NDArray[Any]]):
    """NumPy backend tests for TensorTransfer."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTensorTransferTorch(TestTensorTransfer[torch.Tensor]):
    """PyTorch backend tests for TensorTransfer."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
