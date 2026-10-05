"""Moment-Gaussian conditioning and EIG against the conjugate-Gaussian code.

For a linear model ``y = A m + e`` with a Gaussian prior the moment-Gaussian
blocks are exact, so ``JointGaussian`` and ``DesignObjective`` must agree
with pyapprox's independent conjugate implementations:

- the posterior of a linear prediction ``q = B m`` given a subset of the
  data, against ``DenseGaussianPrediction`` on
  ``LinearGaussianPredOEDBenchmark``;
- the expected information gain about ``m``, against
  ``LinearGaussianKLOEDBenchmark.exact_eig`` (``compute_exact_eig``) and
  ``ConjugateGaussianOEDExpectedInformationGain``. Those use precision
  weighting, noise variance ``sigma^2 / w``; with diagonal noise and
  reference variances ``s^2 = sigma^2`` the blended relaxation gives the
  same likelihood for every ``w``, so the values agree at interior weights
  too.
"""

from typing import List

import numpy as np
import pytest
from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDExpectedInformationGain,
)
from pyapprox.expdesign.gaussian import (
    BlendedObservation,
    DesignObjective,
    ExpectedInformationGain,
)
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.inverse.pushforward.prediction import DenseGaussianPrediction
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.expdesign import (
    build_linear_gaussian_kl_benchmark,
    build_linear_gaussian_pred_benchmark,
)

_SUBSETS = [[0], [2, 5], [0, 1, 4], [1, 2, 3, 5], [0, 1, 2, 3, 4, 5]]


def _indicator(subset: List[int], nobs: int, bkd: Backend[Array]) -> Array:
    weights = bkd.zeros((nobs, 1))
    for ii in subset:
        weights[ii, 0] = 1.0
    return weights


class TestGoalPosteriorAgainstPrediction:
    """Six observations of a cubic, three predictions."""

    @pytest.mark.parametrize("subset", _SUBSETS)
    def test_prediction_posterior(self, bkd: Backend[Array], subset: List[int]) -> None:
        bench = build_linear_gaussian_pred_benchmark(
            nobs=6, degree=3, noise_std=0.3, prior_std=1.0, npred=3, bkd=bkd
        )
        obs_mat, qoi_mat = bench.design_matrix(), bench.qoi_matrix()
        nparams = obs_mat.shape[1]
        bkd.assert_allclose(bench.problem().obs_map()(bkd.eye(nparams)), obs_mat)
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bench.prior_var() * bkd.eye(nparams)
        noise_cov = bench.noise_var() * bkd.eye(6)
        noise = DenseCholeskyCovarianceOperator(noise_cov, bkd)
        joint = JointGaussian(
            DenseBlocks.from_linear_model(
                obs_mat, prior_mean, prior_cov, [qoi_mat], bkd
            ),
            noise,
        )
        data = bkd.asarray(np.random.default_rng(33).normal(size=(6, 1)))

        reference = DenseGaussianPrediction(
            obs_mat[subset],
            qoi_mat,
            prior_mean,
            prior_cov,
            noise_cov[np.ix_(subset, subset)],
            bkd,
        )
        reference.compute(data[subset])

        mean, cov = joint.select(subset).condition(data[subset], 0)
        bkd.assert_allclose(cov, reference.covariance(), rtol=1e-10, atol=1e-12)
        bkd.assert_allclose(mean, reference.mean(), rtol=1e-10, atol=1e-12)

        weights = _indicator(subset, 6, bkd)
        variances = BlendedObservation.from_noise(noise).variances(weights)
        observation = joint.observe(weights, variances, 0)
        bkd.assert_allclose(
            observation.covariance(), reference.covariance(), rtol=1e-10, atol=1e-12
        )
        bkd.assert_allclose(
            observation.mean(data), reference.mean(), rtol=1e-10, atol=1e-12
        )


class TestEIGAgainstConjugate:
    """Six observations of a cubic, EIG about its four coefficients."""

    def _conjugate_eig(
        self,
        bkd: Backend[Array],
        obs_mat: Array,
        prior_cov: Array,
        noise_var: float,
        weights: Array,
    ) -> float:
        """Precision weighting over the sensors with positive weight."""
        active = [
            ii for ii in range(weights.shape[0]) if bkd.to_float(weights[ii, 0]) > 0
        ]
        eig = ConjugateGaussianOEDExpectedInformationGain(prior_cov, bkd)
        eig.set_observation_matrix(obs_mat[active])
        eig.set_noise_covariance(
            bkd.diag(
                bkd.asarray([noise_var / bkd.to_float(weights[ii, 0]) for ii in active])
            )
        )
        return eig.value()

    @pytest.mark.parametrize(
        "w",
        [
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.7, 0.2, 0.5, 0.9, 0.05, 0.3],
            [0.6, 0.0, 1.0, 0.25, 0.0, 0.8],
        ],
    )
    def test_eig(self, bkd: Backend[Array], w: List[float]) -> None:
        bench = build_linear_gaussian_kl_benchmark(
            nobs=6, degree=3, noise_std=0.3, prior_std=1.0, bkd=bkd
        )
        obs_mat = bench.design_matrix()
        nparams = obs_mat.shape[1]
        prior_cov = bench.prior_var() * bkd.eye(nparams)
        noise = DenseCholeskyCovarianceOperator(bench.noise_var() * bkd.eye(6), bkd)
        joint = JointGaussian(
            DenseBlocks.from_linear_model(
                obs_mat, bkd.zeros((nparams, 1)), prior_cov, [bkd.eye(nparams)], bkd
            ),
            noise,
        )
        objective = DesignObjective(
            joint, BlendedObservation.from_noise(noise), ExpectedInformationGain(), 0
        )
        weights = bkd.asarray(np.array(w)[:, None])
        eig = -objective(weights)[0, 0]
        expected = [
            bench.exact_eig(weights),
            self._conjugate_eig(bkd, obs_mat, prior_cov, bench.noise_var(), weights),
        ]
        bkd.assert_allclose(bkd.stack([eig, eig]), bkd.asarray(expected), rtol=1e-10)
