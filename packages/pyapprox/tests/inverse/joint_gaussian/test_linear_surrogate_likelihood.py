"""The approximation-error likelihood does not depend on the linear surrogate.

The approximation-error approach picks a linear surrogate: a matrix ``L``
with ``G(m, a) ~ L m``. It writes the model as ``G(m, a) = L m + eps(m, a)``,
where ``eps`` is the surrogate's error, and approximates ``eps | m`` as
Gaussian using the moments of ``(m, eps)``. The resulting likelihood of
``y = G + e`` given ``m`` has

- mean ``L m + mu_eps + Gamma_eps,m Gamma_mm^{-1} (m - mu_m)``,
- covariance ``Gamma_eps,eps - Gamma_eps,m Gamma_mm^{-1} Gamma_m,eps + Gamma_e``.

Substituting ``eps = G - L m``, every term in ``L`` cancels, leaving the
best-linear-predictor likelihood: mean ``mu_g + B (m - mu_m)`` with
``B = Gamma_gm Gamma_mm^{-1}``, and covariance ``Gamma_yy|m``.

The test builds this likelihood from two different random surrogates,
``L_1`` and ``L_2``. Their errors ``eps_1`` and ``eps_2`` have very
different moments, yet both likelihoods must equal the one computed by
``JointGaussian.blp`` and ``observation_covariance_given_target``, and so
each other. Two surrogates show the result does not depend on ``L``, which
one surrogate could match by coincidence. The cancellation is algebraic, so
it holds for sampled moments; all moments here come from one shared set of
Monte Carlo samples.
"""

from typing import Tuple

import numpy as np

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.joint import InputTarget, SeparateFunctions
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import DiagonalCovarianceOperator
from pyapprox.probability.moments import (
    CovarianceBlocksProtocol,
    DenseBlocks,
    QuadratureMoments,
    SampledRule,
)
from pyapprox.util.backends.protocols import Array, Backend


class _StandardNormalSampler:
    """Sampler of N(0, I) with its own generator."""

    def __init__(self, nvars: int, bkd: Backend[Array], seed: int) -> None:
        self._nvars, self._bkd = nvars, bkd
        self._rng = np.random.default_rng(seed)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        points = self._rng.standard_normal((self._nvars, nsamples))
        return self._bkd.asarray(points), self._bkd.full((nsamples,), 1.0 / nsamples)

    def reset(self) -> None:
        pass


class TestLinearSurrogateLikelihood:
    """Parameter m in R^2, nuisance a in R^2, 4 observations, nonlinear G."""

    _nm, _na, _nobs = 2, 2, 4

    def _setup(self, bkd: Backend[Array]) -> None:
        rng = np.random.default_rng(16)
        self._amat = bkd.asarray(rng.normal(size=(self._nobs, self._nm)))
        self._bmat = bkd.asarray(rng.normal(size=(self._nobs, self._na)))
        self._surrogates = [
            bkd.asarray(rng.normal(size=(self._nobs, self._nm))) for _ in range(2)
        ]
        self._m_values = bkd.asarray(rng.normal(size=(self._nm, 3)))
        self._noise = DiagonalCovarianceOperator(bkd.full((self._nobs,), 0.05), bkd)
        nvars = self._nm + self._na
        # One fixed set of 200 samples shared by every moment computation.
        self._rule = SampledRule(_StandardNormalSampler(nvars, bkd, 17), 200)

    def _model(self, bkd: Backend[Array], xi: Array) -> Array:
        m, a = xi[: self._nm], xi[self._nm :]
        return (
            bkd.sin(bkd.dot(self._amat, m))
            + bkd.dot(self._bmat, a)
            + 0.3 * m[0:1] * a[1:2]
        )

    def _blocks(
        self, bkd: Backend[Array], surrogate: Array
    ) -> CovarianceBlocksProtocol[Array]:
        """Moments of (m, G(m, a) - surrogate m) from the shared samples."""
        nvars = self._nm + self._na

        def residual(xi: Array) -> Array:
            return self._model(bkd, xi) - bkd.dot(surrogate, xi[: self._nm])

        evaluator = SeparateFunctions(
            FunctionFromCallable(self._nobs, nvars, residual, bkd),
            [InputTarget(nvars, bkd, rows=list(range(self._nm)))],
        )
        return QuadratureMoments(self._rule, evaluator).blocks()

    def _surrogate_likelihood(
        self, bkd: Backend[Array], surrogate: Array
    ) -> Tuple[Array, Array]:
        """Mean at self._m_values and covariance, from moments of (m, eps)."""
        blocks = self._blocks(bkd, surrogate)
        cmm = blocks.target_covariance(0)
        cem = blocks.target_obs_covariance(0).T
        gain = bkd.solve(cmm, cem.T).T
        mean = (
            bkd.dot(surrogate, self._m_values)
            + blocks.obs_mean()
            + bkd.dot(gain, self._m_values - blocks.target_mean(0))
        )
        cov = blocks.obs_covariance() - bkd.dot(gain, cem.T) + self._noise.covariance()
        return mean, cov

    def test_independent_of_surrogate_and_equal_to_blp(
        self, bkd: Backend[Array]
    ) -> None:
        self._setup(bkd)
        zero = bkd.zeros((self._nobs, self._nm))
        blocks = self._blocks(bkd, zero)
        assert isinstance(blocks, DenseBlocks)
        joint = JointGaussian(blocks, self._noise)
        blp_mean = blocks.obs_mean() + bkd.dot(
            joint.blp(0), self._m_values - blocks.target_mean(0)
        )
        blp_cov = joint.observation_covariance_given_target(0)
        for surrogate in self._surrogates:
            mean, cov = self._surrogate_likelihood(bkd, surrogate)
            bkd.assert_allclose(mean, blp_mean, rtol=1e-10, atol=1e-12)
            bkd.assert_allclose(cov, blp_cov, rtol=1e-10, atol=1e-12)
