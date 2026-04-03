"""Tests for MFNet composite fitter."""

import numpy as np
import pytest

from pyapprox.surrogates.mfnets.builders import build_chain_mfnet
from pyapprox.surrogates.mfnets.fitters.composite_fitter import (
    MFNetCompositeFitter,
)
from pyapprox.surrogates.mfnets.helpers import (
    generate_synthetic_data,
    randomize_coefficients,
)
from tests._helpers.markers import slower_test, slowest_test


class TestCompositeFitter:

    @slower_test
    def test_composite_recovers_truth(self, bkd) -> None:
        """ALS+gradient composite recovers truth from clean data."""
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

    def test_composite_als_only(self, bkd) -> None:
        """skip_gradient=True only runs ALS."""
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
        assert result.gradient_result() is None
        # als_result should be present
        assert result.als_result() is not None
        assert len(result.als_result().loss_history()) > 0

    @pytest.mark.slow_on("NumpyBkd")
    def test_composite_result_has_both_phases(self, bkd) -> None:
        """Result exposes both als_result and gradient_result."""
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
        assert result.als_result() is not None
        assert result.gradient_result() is not None

        # Loss should be finite
        assert np.isfinite(result.loss_value())

    @slowest_test
    def test_three_node_composite_recovery(self, bkd) -> None:
        """Composite fitter recovers 3-node chain."""
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
