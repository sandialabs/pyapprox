"""Tests for UMBridgeModel client."""

import os
import unittest

import numpy as np

from pyapprox.typing.interface.umbridge.client import (
    UMBridgeModel,
    UMBRIDGE_AVAILABLE,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.test_utils import load_tests


@unittest.skipUnless(UMBRIDGE_AVAILABLE, "umbridge package not installed")
class TestUMBridgeModel(unittest.TestCase):
    """Tests for UMBridgeModel client.

    These tests spawn a real UMBridge server subprocess and test
    the client against it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Start the test server."""
        cls.bkd = NumpyBkd()
        cls.url = "http://localhost:4242"

        # Get path to test server
        server_dir = os.path.dirname(__file__)
        server_script = os.path.join(server_dir, "test_umbridge_server.py")
        run_command = f"python {server_script}"

        # Start the server
        cls.process, cls.out = UMBridgeModel.start_server(
            run_command, url=cls.url, max_wait_time=30
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Kill the test server."""
        UMBridgeModel.kill_server(cls.process, cls.out)

    def test_basic_evaluation(self) -> None:
        """Test basic model evaluation."""
        config = {"nvars": 2}
        model = UMBridgeModel(
            self.url, "quadratic", self.bkd, config=config
        )

        # f([1, 2]) = 1^2 + 2^2 = 5
        samples = self.bkd.asarray([[1.0], [2.0]])
        values = model(samples)

        self.assertEqual(values.shape, (1, 1))
        self.bkd.assert_allclose(
            values, self.bkd.asarray([[5.0]])
        )

    def test_batch_evaluation(self) -> None:
        """Test evaluation of multiple samples."""
        config = {"nvars": 2}
        model = UMBridgeModel(
            self.url, "quadratic", self.bkd, config=config
        )

        # f([1, 2]) = 5, f([2, 3]) = 13
        samples = self.bkd.asarray([[1.0, 2.0], [2.0, 3.0]])
        values = model(samples)

        self.assertEqual(values.shape, (1, 2))
        self.bkd.assert_allclose(
            values, self.bkd.asarray([[5.0, 13.0]])
        )

    def test_nvars_nqoi(self) -> None:
        """Test nvars and nqoi from model."""
        config = {"nvars": 3}
        model = UMBridgeModel(
            self.url, "quadratic", self.bkd, config=config
        )

        self.assertEqual(model.nvars(), 3)
        self.assertEqual(model.nqoi(), 1)

    def test_jacobian(self) -> None:
        """Test jacobian computation."""
        config = {"nvars": 2}
        model = UMBridgeModel(
            self.url, "quadratic", self.bkd, config=config
        )

        self.assertTrue(model.has_jacobian())

        # Jacobian of sum(x_i^2) at [1, 2] is [2*1, 2*2] = [2, 4]
        sample = self.bkd.asarray([[1.0], [2.0]])
        jacobian = model.jacobian(sample)

        self.assertEqual(jacobian.shape, (1, 2))
        self.bkd.assert_allclose(
            jacobian, self.bkd.asarray([[2.0, 4.0]])
        )

    def test_linear_model_no_gradient(self) -> None:
        """Test model without gradient support."""
        config = {"nvars": 2}
        model = UMBridgeModel(
            self.url, "linear", self.bkd, config=config
        )

        # Linear model doesn't support gradient
        self.assertFalse(model.has_jacobian())

        with self.assertRaises(RuntimeError):
            sample = self.bkd.asarray([[1.0], [2.0]])
            model.jacobian(sample)

    def test_linear_model_evaluation(self) -> None:
        """Test linear model evaluation."""
        config = {"nvars": 2}
        model = UMBridgeModel(
            self.url, "linear", self.bkd, config=config
        )

        # f([1, 2]) = 1 + 2 = 3
        samples = self.bkd.asarray([[1.0], [2.0]])
        values = model(samples)

        self.assertEqual(values.shape, (1, 1))
        self.bkd.assert_allclose(
            values, self.bkd.asarray([[3.0]])
        )

    def test_config_update(self) -> None:
        """Test updating model configuration."""
        config = {"nvars": 2}
        model = UMBridgeModel(
            self.url, "quadratic", self.bkd, config=config
        )

        self.assertEqual(model.nvars(), 2)

        # Update config
        model.set_config({"nvars": 3})
        self.assertEqual(model.nvars(), 3)

    def test_repr(self) -> None:
        """Test string representation."""
        config = {"nvars": 2}
        model = UMBridgeModel(
            self.url, "quadratic", self.bkd, config=config
        )

        repr_str = repr(model)
        self.assertIn("UMBridgeModel", repr_str)
        self.assertIn("quadratic", repr_str)


@unittest.skipIf(UMBRIDGE_AVAILABLE, "Testing import error when not installed")
class TestUMBridgeNotInstalled(unittest.TestCase):
    """Test behavior when umbridge is not installed."""

    def test_import_error(self) -> None:
        """Test that appropriate error is raised when umbridge not installed."""
        # This test only runs when umbridge is NOT installed
        bkd = NumpyBkd()
        with self.assertRaises(ImportError):
            UMBridgeModel("http://localhost:4242", "test", bkd)


if __name__ == "__main__":
    unittest.main()
