"""
Tests verifying torch.autograd.functional.jacobian compatibility.

These tests confirm that the entire forward pass from design_weights to
objective value preserves the PyTorch computation graph, enabling
automatic differentiation.

After fixing float() conversions in compute_log_normalization and
GaussianOEDOuterLoopLikelihood._compute_log_normalization, the full
pipeline is autograd-safe:
- No .detach(), .data, .numpy(), .item() in production code
- bkd.copy() = tensor.clone() (preserves graph)
- set_observations()/set_latent_samples() store references (no detach)
- LogEvidence uses log-sum-exp trick (max, exp, sum, log — all autograd-safe)

Note on reparameterization: The analytical jacobians include a
reparameterization term (Component 3) when latent_samples are set,
assuming obs = shapes + sqrt(var/w) * latent. For standalone component
tests where observations are fixed data (not generated from w), we do
NOT set latent_samples. The full reparameterization path is tested
through TestAutogradKLObjective, which generates observations as a
function of design_weights.
"""

import unittest

import numpy as np
import torch

from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.likelihood import (
    GaussianOEDInnerLoopLikelihood,
    GaussianOEDOuterLoopLikelihood,
)
from pyapprox.typing.expdesign.evidence import Evidence, LogEvidence
from pyapprox.typing.expdesign.objective import KLOEDObjective


class TestAutogradOuterLikelihood(unittest.TestCase):
    """Test torch.autograd through GaussianOEDOuterLoopLikelihood.

    Observations are fixed data (not generated from design_weights), so
    latent_samples are NOT set to avoid the reparameterization term mismatch.
    """

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

        self._nobs = 3
        self._nouter = 10

        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.2, 0.15])
        )
        self._shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._obs = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

    def _create_likelihood(self):
        like = GaussianOEDOuterLoopLikelihood(
            self._noise_variances, self._bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        return like

    def test_autograd_jacobian_matches_analytical(self):
        """torch.autograd.functional.jacobian matches analytical jacobian."""
        like = self._create_likelihood()
        weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

        # Analytical jacobian: (nouter, nobs)
        analytical_jac = like.jacobian(weights)

        # Autograd jacobian
        def forward(w):
            return like(w).flatten()  # (nouter,)

        w = weights.clone().requires_grad_(True)
        autograd_jac = torch.autograd.functional.jacobian(forward, w)
        # Shape: (nouter, nobs, 1) -> squeeze to (nouter, nobs)
        autograd_jac = autograd_jac.squeeze(-1)

        self._bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)


class TestAutogradEvidence(unittest.TestCase):
    """Test torch.autograd through Evidence.

    Observations are fixed data, so latent_samples are NOT set.
    """

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        # torch.compile donated buffers conflict with autograd jacobian
        import torch._functorch.config as _ftconfig
        _ftconfig.donated_buffer = False
        self._bkd = TorchBkd()
        np.random.seed(42)

        self._nobs = 3
        self._ninner = 15
        self._nouter = 10

        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.2, 0.15])
        )
        self._shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._obs = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

    def _create_evidence(self):
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        quad_weights = self._bkd.ones((self._ninner,)) / self._ninner
        return Evidence(like, quad_weights, self._bkd)

    def test_autograd_jacobian_matches_analytical(self):
        """torch.autograd.functional.jacobian matches analytical jacobian."""
        ev = self._create_evidence()
        weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

        # Analytical jacobian: (nouter, nobs)
        analytical_jac = ev.jacobian(weights)

        # Autograd jacobian
        def forward(w):
            return ev(w).flatten()  # (nouter,)

        w = weights.clone().requires_grad_(True)
        autograd_jac = torch.autograd.functional.jacobian(forward, w)
        autograd_jac = autograd_jac.squeeze(-1)

        self._bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-6)


class TestAutogradLogEvidence(unittest.TestCase):
    """Test torch.autograd through LogEvidence.

    Observations are fixed data, so latent_samples are NOT set.
    """

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        # torch.compile donated buffers conflict with autograd jacobian
        import torch._functorch.config as _ftconfig
        _ftconfig.donated_buffer = False
        self._bkd = TorchBkd()
        np.random.seed(42)

        self._nobs = 3
        self._ninner = 15
        self._nouter = 10

        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.2, 0.15])
        )
        self._shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._obs = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

    def _create_log_evidence(self):
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        quad_weights = self._bkd.ones((self._ninner,)) / self._ninner
        return LogEvidence(like, quad_weights, self._bkd)

    def test_autograd_jacobian_matches_analytical(self):
        """torch.autograd.functional.jacobian matches analytical jacobian."""
        log_ev = self._create_log_evidence()
        weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

        # Analytical jacobian: (nouter, nobs)
        analytical_jac = log_ev.jacobian(weights)

        # Autograd jacobian
        def forward(w):
            return log_ev(w).flatten()  # (nouter,)

        w = weights.clone().requires_grad_(True)
        autograd_jac = torch.autograd.functional.jacobian(forward, w)
        autograd_jac = autograd_jac.squeeze(-1)

        self._bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-6)


class TestAutogradKLObjective(unittest.TestCase):
    """Test torch.autograd through KLOEDObjective.

    This exercises the full path including reparameterization trick:
    design_weights -> obs generation (sqrt(var/w) * latent) ->
    inner likelihood -> LogEvidence (logsumexp) ->
    outer likelihood -> weighted sum

    The reparameterization trick creates implicit dependence of obs on
    design_weights, which autograd correctly tracks through the
    computation graph.
    """

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

        self._nobs = 3
        self._ninner = 15
        self._nouter = 10

    def _create_objective(self):
        bkd = self._bkd
        noise_variances = bkd.asarray(np.array([0.1, 0.2, 0.15]))
        inner_loglike = GaussianOEDInnerLoopLikelihood(
            noise_variances, bkd,
        )
        inner_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        outer_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        latent_samples = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        return KLOEDObjective(
            inner_loglike, outer_shapes, latent_samples, inner_shapes,
            None, None, bkd,
        )

    def test_autograd_jacobian_matches_analytical(self):
        """torch.autograd.functional.jacobian matches analytical jacobian."""
        obj = self._create_objective()
        np.random.seed(123)
        weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

        # Analytical jacobian: (1, nobs)
        analytical_jac = obj.jacobian(weights)

        # Autograd jacobian
        def forward(w):
            return obj(w).flatten()  # (1,)

        w = weights.clone().requires_grad_(True)
        autograd_jac = torch.autograd.functional.jacobian(forward, w)
        # Shape: (1, nobs, 1) -> squeeze to (1, nobs)
        autograd_jac = autograd_jac.squeeze(-1)

        self._bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
