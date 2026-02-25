"""Tests for MFNet composite fitter."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.surrogates.mfnets.builders import build_chain_mfnet
from pyapprox.surrogates.mfnets.fitters.composite_fitter import (
    MFNetCompositeFitter,
)
from pyapprox.surrogates.mfnets.helpers import (
    generate_synthetic_data,
    randomize_coefficients,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import (
    load_tests,  # noqa: F401
    slow_test,
    slower_test,
    slowest_test,
)


class TestCompositeFitter(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @slower_test
    def test_composite_recovers_truth(self) -> None:
        """ALS+gradient composite recovers truth from clean data."""
        bkd = self._bkd

        # Build and randomize a true network
        true_net = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=2,
            bkd=bkd,
            leaf_level=3,
            scale_level=1,
            delta_level=3,
        )
        randomize_coefficients(true_net, bkd, seed=10)

        # Generate clean training data
        samples, values = generate_synthetic_data(
            true_net,
            bkd,
            nsamples_per_node=[30, 25],
            seed=42,
        )

        # Build blank network and fit
        fit_net = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=2,
            bkd=bkd,
            leaf_level=3,
            scale_level=1,
            delta_level=3,
        )
        fitter = MFNetCompositeFitter(
            bkd,
            als_max_sweeps=20,
            als_tol=1e-14,
        )
        result = fitter.fit(fit_net, samples, values)

        # Predictions should match truth
        np.random.seed(123)
        test_s = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        true_out = true_net(test_s)
        fit_out = result(test_s)
        bkd.assert_allclose(fit_out, true_out, atol=1e-4)

    def test_composite_als_only(self) -> None:
        """skip_gradient=True only runs ALS."""
        bkd = self._bkd

        true_net = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=2,
            bkd=bkd,
            leaf_level=3,
            scale_level=1,
            delta_level=3,
        )
        randomize_coefficients(true_net, bkd, seed=10)
        samples, values = generate_synthetic_data(
            true_net,
            bkd,
            nsamples_per_node=[30, 25],
            seed=42,
        )

        fit_net = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=2,
            bkd=bkd,
            leaf_level=3,
            scale_level=1,
            delta_level=3,
        )
        fitter = MFNetCompositeFitter(
            bkd,
            als_max_sweeps=10,
            skip_gradient=True,
        )
        result = fitter.fit(fit_net, samples, values)

        # gradient_result should be None
        self.assertIsNone(result.gradient_result())
        # als_result should be present
        self.assertIsNotNone(result.als_result())
        self.assertGreater(len(result.als_result().loss_history()), 0)

    def test_composite_result_has_both_phases(self) -> None:
        """Result exposes both als_result and gradient_result."""
        bkd = self._bkd

        true_net = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=2,
            bkd=bkd,
            leaf_level=2,
            scale_level=0,
            delta_level=2,
        )
        randomize_coefficients(true_net, bkd, seed=10)
        samples, values = generate_synthetic_data(
            true_net,
            bkd,
            nsamples_per_node=[20, 15],
            seed=42,
        )

        fit_net = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=2,
            bkd=bkd,
            leaf_level=2,
            scale_level=0,
            delta_level=2,
        )
        fitter = MFNetCompositeFitter(
            bkd,
            als_max_sweeps=5,
            als_tol=1e-10,
        )
        result = fitter.fit(fit_net, samples, values)

        # Both results should be present
        self.assertIsNotNone(result.als_result())
        self.assertIsNotNone(result.gradient_result())

        # Loss should be finite
        self.assertTrue(np.isfinite(result.loss_value()))

    @slowest_test
    def test_three_node_composite_recovery(self) -> None:
        """Composite fitter recovers 3-node chain."""
        bkd = self._bkd

        true_net = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=3,
            bkd=bkd,
            leaf_level=2,
            scale_level=0,
            delta_level=2,
        )
        randomize_coefficients(true_net, bkd, seed=30)
        samples, values = generate_synthetic_data(
            true_net,
            bkd,
            nsamples_per_node=[30, 25, 20],
            seed=42,
        )

        fit_net = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=3,
            bkd=bkd,
            leaf_level=2,
            scale_level=0,
            delta_level=2,
        )
        fitter = MFNetCompositeFitter(
            bkd,
            als_max_sweeps=30,
            als_tol=1e-14,
        )
        result = fitter.fit(fit_net, samples, values)

        np.random.seed(123)
        test_s = bkd.asarray(np.random.uniform(-1, 1, (1, 40)))
        true_out = true_net(test_s)
        fit_out = result(test_s)
        bkd.assert_allclose(fit_out, true_out, atol=1e-4)


# --- Concrete backend test classes ---


class TestCompositeFitterNumpy(TestCompositeFitter[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

    @slow_test
    def test_composite_result_has_both_phases(self) -> None:
        super().test_composite_result_has_both_phases()


class TestCompositeFitterTorch(TestCompositeFitter[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main(verbosity=2)
