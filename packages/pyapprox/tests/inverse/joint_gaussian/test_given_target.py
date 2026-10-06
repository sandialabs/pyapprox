"""Tests for quantities given the target: Gamma_yy|t, the BLP and the guard.

The data form of the expected information gain,
(log det A_w - log det A_w|t) / 2, must equal the target form,
(log det Gamma_tt - log det Gamma_t|z) / 2; that equivalence is what lets
an information-gain criterion and a D-optimal one rank designs alike.
"""

import numpy as np
import pytest

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.joint import SeparateFunctions
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import (
    DenseCholeskyCovarianceOperator,
    DiagonalCovarianceOperator,
)
from pyapprox.probability.moments import (
    DenseBlocks,
    EigenClip,
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

    def sample(self, nsamples: int) -> tuple[Array, Array]:
        points = self._rng.standard_normal((self._nvars, nsamples))
        return self._bkd.asarray(points), self._bkd.full((nsamples,), 1.0 / nsamples)

    def reset(self) -> None:
        pass


class TestGivenTarget:
    """Input xi ~ N(0, P) in R^3; targets xi, B xi (2x3) and b xi (1x3)."""

    def _setup(self, bkd: Backend[Array]) -> None:
        rng = np.random.default_rng(14)
        root = rng.normal(size=(3, 3))
        self._prior_cov = root @ root.T + 0.2 * np.eye(3)
        self._amat = rng.normal(size=(4, 3))
        self._bmat = rng.normal(size=(2, 3))
        self._bvec = rng.normal(size=(1, 3))
        noise_root = rng.normal(size=(4, 4))
        self._noise_cov = 0.05 * noise_root @ noise_root.T + 0.1 * np.eye(4)

    def _joint(self, bkd: Backend[Array]) -> JointGaussian[Array]:
        blocks = DenseBlocks.from_linear_model(
            bkd.asarray(self._amat),
            bkd.zeros((3, 1)),
            bkd.asarray(self._prior_cov),
            [bkd.eye(3), bkd.asarray(self._bmat), bkd.asarray(self._bvec)],
            bkd,
        )
        noise = DenseCholeskyCovarianceOperator(bkd.asarray(self._noise_cov), bkd)
        return JointGaussian(blocks, noise)

    def test_data_form_equals_target_form(self, bkd: Backend[Array]) -> None:
        """For a 2D and a 1D target, with weights including 0 and 1."""
        self._setup(bkd)
        joint = self._joint(bkd)
        w = bkd.asarray([[0.7], [0.0], [1.0], [0.3]])
        nu = (1.0 - w) * bkd.reshape(bkd.diag(joint.noise().covariance()), (4, 1))
        for index in (1, 2):
            obs = joint.observe(w, nu, index)
            data_form = 0.5 * (obs.logdet_zz() - obs.logdet_zz_given_t())
            _, logdet_tt = bkd.slogdet(joint.blocks().target_covariance(index))
            _, logdet_post = bkd.slogdet(obs.covariance())
            target_form = 0.5 * (logdet_tt - logdet_post)
            bkd.assert_allclose(data_form[0], target_form, rtol=1e-10)

    def test_blp_and_covariance_given_parameter(self, bkd: Backend[Array]) -> None:
        """With g = A xi exactly, B = A and Gamma_yy|xi is just the noise."""
        self._setup(bkd)
        joint = self._joint(bkd)
        bkd.assert_allclose(joint.blp(0), bkd.asarray(self._amat), rtol=1e-10)
        bkd.assert_allclose(
            joint.observation_covariance_given_target(0),
            bkd.asarray(self._noise_cov),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_covariance_given_prediction(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        a, b, p = self._amat, self._bmat, self._prior_cov
        expected = (
            a @ p @ a.T
            - a @ p @ b.T @ np.linalg.solve(b @ p @ b.T, b @ p @ a.T)
            + self._noise_cov
        )
        bkd.assert_allclose(
            self._joint(bkd).observation_covariance_given_target(1),
            bkd.asarray(expected),
            rtol=1e-10,
        )

    def test_computed_once_and_shared(self, bkd: Backend[Array]) -> None:
        """Gamma_yy|t is cached, and select slices the cache."""
        self._setup(bkd)
        joint = self._joint(bkd)
        first = joint.observation_covariance_given_target(1)
        assert joint.observation_covariance_given_target(1) is first
        rows = [3, 1]
        sliced = joint.select(rows).observation_covariance_given_target(1)
        fresh = self._joint(bkd).select(rows).observation_covariance_given_target(1)
        bkd.assert_allclose(sliced, first[rows][:, rows], rtol=1e-12)
        bkd.assert_allclose(sliced, fresh, rtol=1e-10)

    def _sampled_joint(
        self, bkd: Backend[Array], nsamples: int, known: bool
    ) -> JointGaussian[Array]:
        """Blocks of (xi, A xi) from nsamples Monte Carlo samples."""
        chol = bkd.asarray(np.linalg.cholesky(self._prior_cov))
        amat = bkd.asarray(self._amat)
        evaluator = SeparateFunctions(
            FunctionFromCallable(4, 3, lambda z: bkd.dot(amat, bkd.dot(chol, z)), bkd),
            [FunctionFromCallable(3, 3, lambda z: bkd.dot(chol, z), bkd)],
        )
        rule = SampledRule(_StandardNormalSampler(3, bkd, 15), nsamples)
        blocks = QuadratureMoments(rule, evaluator).blocks()
        assert isinstance(blocks, DenseBlocks)
        noise = DiagonalCovarianceOperator(bkd.full((4,), 0.1), bkd)
        if not known:
            return JointGaussian(blocks, noise)
        # Exact prior moments with sampled cross covariances may be
        # indefinite, so this case asks for repair explicitly.
        exact = blocks.with_known_targets(
            {0: (bkd.zeros((3, 1)), bkd.asarray(self._prior_cov))}
        )
        return JointGaussian(exact, noise, EigenClip())

    def test_guard_refuses_collapsed_estimate(self, bkd: Backend[Array]) -> None:
        """A sampled 3D target with N = n_t + 1 = 4 samples."""
        self._setup(bkd)
        joint = self._sampled_joint(bkd, 4, known=False)
        w, nu = bkd.full((4, 1), 0.5), bkd.full((4, 1), 0.05)
        obs = joint.observe(w, nu, 0)
        # Quantities that do not condition on the target are unaffected.
        obs.covariance()
        obs.logdet_zz()
        for call in (
            lambda: joint.observation_covariance_given_target(0),
            obs.logdet_zz_given_t,
            obs.logdet_zz_given_t_gradient,
        ):
            with pytest.raises(ValueError, match="N <= n_t \\+ 1"):
                call()

    def test_guard_allows_enough_samples(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        joint = self._sampled_joint(bkd, 5, known=False)
        joint.observation_covariance_given_target(0)

    def test_guard_allows_known_target(self, bkd: Backend[Array]) -> None:
        """Exact target moments do not collapse, whatever N is."""
        self._setup(bkd)
        joint = self._sampled_joint(bkd, 4, known=True)
        joint.observation_covariance_given_target(0)
