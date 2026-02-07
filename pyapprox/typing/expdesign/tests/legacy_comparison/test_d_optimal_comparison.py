"""
Legacy comparison tests for DOptimalLinearModelObjective.

TODO: Delete after legacy removed.

These tests verify that the new typing module implementation produces
identical results to the legacy pyapprox.expdesign implementation.
"""

import unittest

import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin


class TestDOptimalLegacyComparison(unittest.TestCase):
    """Verify typing DOptimalLinearModelObjective matches legacy."""

    def setUp(self):
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        self._bkd = NumpyBkd()

    def test_objective_matches_legacy(self):
        """Test objective value matches legacy implementation."""
        np.random.seed(42)

        # Shared setup
        nobs, nparams = 5, 3
        design_matrix = np.random.randn(nobs, nparams)
        noise_cov = np.array(0.1)  # scalar
        prior_cov = np.array(1.0)  # scalar
        weights = np.ones((nobs, 1)) / nobs

        # Legacy
        from pyapprox.expdesign.bayesoed import DOptimalLinearModelObjective
        from pyapprox.interface.model import DenseMatrixLinearModel

        legacy_model = DenseMatrixLinearModel(design_matrix, backend=NumpyMixin)
        legacy_noise = NumpyMixin.asarray(noise_cov)
        legacy_prior = NumpyMixin.asarray(prior_cov)
        legacy_obj = DOptimalLinearModelObjective(
            legacy_model, legacy_noise, legacy_prior
        )
        legacy_result = legacy_obj(weights)

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.objective import (
            DOptimalLinearModelObjective as TypingDOptimal,
        )

        bkd = NumpyBkd()
        typing_obj = TypingDOptimal(
            bkd.asarray(design_matrix),
            bkd.asarray(noise_cov),
            bkd.asarray(prior_cov),
            bkd,
        )
        typing_result = typing_obj(bkd.asarray(weights))

        self._bkd.assert_allclose(
            typing_result,
            self._bkd.asarray(NumpyMixin.to_numpy(legacy_result)),
            rtol=1e-12,
        )

    def test_jacobian_matches_legacy(self):
        """Test Jacobian matches legacy implementation."""
        np.random.seed(42)

        nobs, nparams = 5, 3
        design_matrix = np.random.randn(nobs, nparams)
        noise_cov = np.array(0.1)
        prior_cov = np.array(1.0)
        weights = np.ones((nobs, 1)) / nobs

        # Legacy
        from pyapprox.expdesign.bayesoed import DOptimalLinearModelObjective
        from pyapprox.interface.model import DenseMatrixLinearModel

        legacy_model = DenseMatrixLinearModel(design_matrix, backend=NumpyMixin)
        legacy_obj = DOptimalLinearModelObjective(
            legacy_model,
            NumpyMixin.asarray(noise_cov),
            NumpyMixin.asarray(prior_cov),
        )
        legacy_jac = legacy_obj.jacobian(weights)

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.objective import (
            DOptimalLinearModelObjective as TypingDOptimal,
        )

        bkd = NumpyBkd()
        typing_obj = TypingDOptimal(
            bkd.asarray(design_matrix),
            bkd.asarray(noise_cov),
            bkd.asarray(prior_cov),
            bkd,
        )
        typing_jac = typing_obj.jacobian(bkd.asarray(weights))

        self._bkd.assert_allclose(
            typing_jac,
            self._bkd.asarray(NumpyMixin.to_numpy(legacy_jac)),
            rtol=1e-12,
        )

    def test_hessian_matches_legacy(self):
        """Test Hessian matches legacy implementation."""
        np.random.seed(42)

        nobs, nparams = 5, 3
        design_matrix = np.random.randn(nobs, nparams)
        noise_cov = np.array(0.1)
        prior_cov = np.array(1.0)
        weights = np.ones((nobs, 1)) / nobs

        # Legacy
        from pyapprox.expdesign.bayesoed import DOptimalLinearModelObjective
        from pyapprox.interface.model import DenseMatrixLinearModel

        legacy_model = DenseMatrixLinearModel(design_matrix, backend=NumpyMixin)
        legacy_obj = DOptimalLinearModelObjective(
            legacy_model,
            NumpyMixin.asarray(noise_cov),
            NumpyMixin.asarray(prior_cov),
        )
        legacy_hess = legacy_obj.hessian(weights)

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.objective import (
            DOptimalLinearModelObjective as TypingDOptimal,
        )

        bkd = NumpyBkd()
        typing_obj = TypingDOptimal(
            bkd.asarray(design_matrix),
            bkd.asarray(noise_cov),
            bkd.asarray(prior_cov),
            bkd,
        )
        typing_hess = typing_obj.hessian(bkd.asarray(weights))

        self._bkd.assert_allclose(
            typing_hess,
            self._bkd.asarray(NumpyMixin.to_numpy(legacy_hess)),
            rtol=1e-12,
        )

    def test_different_weights(self):
        """Test with non-uniform weights."""
        np.random.seed(123)

        nobs, nparams = 4, 2
        design_matrix = np.random.randn(nobs, nparams)
        noise_cov = np.array(0.2)
        prior_cov = np.array(0.5)
        weights = np.random.dirichlet(np.ones(nobs))[:, None]

        # Legacy
        from pyapprox.expdesign.bayesoed import DOptimalLinearModelObjective
        from pyapprox.interface.model import DenseMatrixLinearModel

        legacy_model = DenseMatrixLinearModel(design_matrix, backend=NumpyMixin)
        legacy_obj = DOptimalLinearModelObjective(
            legacy_model,
            NumpyMixin.asarray(noise_cov),
            NumpyMixin.asarray(prior_cov),
        )
        legacy_result = legacy_obj(weights)
        legacy_jac = legacy_obj.jacobian(weights)

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.objective import (
            DOptimalLinearModelObjective as TypingDOptimal,
        )

        bkd = NumpyBkd()
        typing_obj = TypingDOptimal(
            bkd.asarray(design_matrix),
            bkd.asarray(noise_cov),
            bkd.asarray(prior_cov),
            bkd,
        )
        typing_result = typing_obj(bkd.asarray(weights))
        typing_jac = typing_obj.jacobian(bkd.asarray(weights))

        self._bkd.assert_allclose(
            typing_result,
            self._bkd.asarray(NumpyMixin.to_numpy(legacy_result)),
            rtol=1e-12,
        )
        self._bkd.assert_allclose(
            typing_jac,
            self._bkd.asarray(NumpyMixin.to_numpy(legacy_jac)),
            rtol=1e-12,
        )

    def test_hvp_matches_legacy_hessian(self):
        """Test HVP matches legacy Hessian @ vec."""
        np.random.seed(42)

        nobs, nparams = 5, 3
        design_matrix = np.random.randn(nobs, nparams)
        noise_cov = np.array(0.1)
        prior_cov = np.array(1.0)
        weights = np.ones((nobs, 1)) / nobs
        vec = np.random.randn(nobs, 1)

        # Legacy - compute Hessian @ vec
        from pyapprox.expdesign.bayesoed import DOptimalLinearModelObjective
        from pyapprox.interface.model import DenseMatrixLinearModel

        legacy_model = DenseMatrixLinearModel(design_matrix, backend=NumpyMixin)
        legacy_obj = DOptimalLinearModelObjective(
            legacy_model,
            NumpyMixin.asarray(noise_cov),
            NumpyMixin.asarray(prior_cov),
        )
        legacy_hess = legacy_obj.hessian(weights)
        # hessian shape is (1, nobs, nobs), vec is (nobs, 1)
        legacy_hvp = np.squeeze(legacy_hess, axis=0) @ vec

        # Typing - use hvp method
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.objective import (
            DOptimalLinearModelObjective as TypingDOptimal,
        )

        bkd = NumpyBkd()
        typing_obj = TypingDOptimal(
            bkd.asarray(design_matrix),
            bkd.asarray(noise_cov),
            bkd.asarray(prior_cov),
            bkd,
        )
        typing_hvp = typing_obj.hvp(bkd.asarray(weights), bkd.asarray(vec))

        # HVP returns (1, nobs), need to reshape for comparison
        self._bkd.assert_allclose(
            typing_hvp.T,
            self._bkd.asarray(legacy_hvp),
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
