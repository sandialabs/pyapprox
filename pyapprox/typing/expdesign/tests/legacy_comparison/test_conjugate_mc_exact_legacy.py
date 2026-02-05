"""
Legacy comparison tests for conjugate Gaussian OED MC vs exact formulas.

Replicates test_conjugate_gaussian_prior_OED_for_prediction_exact_formulas
from pyapprox/expdesign/tests/test_bayesoed.py (lines 651-790).

Uses small sample counts to run fast while debugging any discrepancies.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

# Legacy imports
from pyapprox.expdesign.bayesoed_benchmarks import (
    LinearGaussianBayesianOEDForPredictionBenchmark,
    ConjugateGaussianOEDForNormalExpectedStdDev,
    ConjugateGaussianOEDForNormalExpectedEntropicDev,
    ConjugateGaussianOEDForNormalExpectedAVaRDev,
    ConjugateGaussianOEDForNormalExpectedKLDivergence,
    ConjugateGaussianOEDForLogNormalExpectedStdDev,
    ConjugateGaussianOEDForLogNormalAVaRStdDev,
)
from pyapprox.inference.laplace import (
    DenseMatrixLaplacePosteriorApproximation,
    GaussianPushForward,
)
from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.optimization.risk import (
    LogNormalAnalyticalRiskMeasures,
    GaussianAnalyticalRiskMeasures,
    AverageValueAtRisk,
)

# Typed imports
from pyapprox.typing.expdesign.analytical import (
    ConjugateGaussianOEDExpectedStdDev,
    ConjugateGaussianOEDExpectedEntropicDev,
    ConjugateGaussianOEDExpectedAVaRDev,
    ConjugateGaussianOEDExpectedKLDivergence,
    ConjugateGaussianOEDForLogNormalExpectedStdDev as TypedLogNormalStdDev,
    ConjugateGaussianOEDForLogNormalAVaRStdDev as TypedLogNormalAVaR,
)
from pyapprox.typing.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.typing.inverse.pushforward.gaussian import GaussianPushforward


class TestConjugateMCExactLegacy(Generic[Array], unittest.TestCase):
    """Legacy comparison tests for conjugate MC vs exact."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def get_backend(self):
        """Return legacy backend name."""
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

        # Match legacy setup exactly
        self._nobs = 2
        self._min_degree = 0
        self._degree = 3
        self._noise_std = 0.5
        self._prior_std = 0.5
        self._nsamples = 100  # Small for fast debugging (legacy uses 1e4)
        self._lamda = 2.0
        self._beta = 0.5

    def _setup_legacy_problem(self):
        """Set up problem using legacy classes."""
        bkd = self.get_backend()

        problem = LinearGaussianBayesianOEDForPredictionBenchmark(
            self._nobs,
            self._min_degree,
            self._degree,
            self._noise_std,
            self._prior_std,
            backend=bkd,
            nqoi=1,
        )

        prior = DenseCholeskyMultivariateGaussian(
            problem.get_prior().mean(),
            problem.get_prior().covariance(),
            backend=bkd,
        )

        obs_mat = problem.get_observation_model().matrix()
        noise_cov = bkd.diag(problem.get_noise_covariance_diag())
        qoi_mat = problem.get_qoi_model().matrix()

        return problem, prior, obs_mat, noise_cov, qoi_mat

    def _setup_typed_problem(self, prior, obs_mat, noise_cov, qoi_mat):
        """Set up typed problem from legacy arrays."""
        bkd = self._bkd

        # Convert arrays
        prior_mean = bkd.asarray(np.array(prior.mean()))
        prior_cov = bkd.asarray(np.array(prior.covariance()))
        obs_mat_typed = bkd.asarray(np.array(obs_mat))
        noise_cov_typed = bkd.asarray(np.array(noise_cov))
        qoi_mat_typed = bkd.asarray(np.array(qoi_mat))

        return prior_mean, prior_cov, obs_mat_typed, noise_cov_typed, qoi_mat_typed

    def test_expected_stdev_legacy_vs_typed(self) -> None:
        """Compare legacy and typed expected std dev utilities."""
        legacy_bkd = self.get_backend()
        problem, prior, obs_mat, noise_cov, qoi_mat = self._setup_legacy_problem()
        prior_mean, prior_cov, obs_mat_typed, noise_cov_typed, qoi_mat_typed = (
            self._setup_typed_problem(prior, obs_mat, noise_cov, qoi_mat)
        )

        # Legacy utility
        legacy_utility = ConjugateGaussianOEDForNormalExpectedStdDev(prior, qoi_mat)
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_value = legacy_utility.value()

        # Typed utility
        typed_utility = ConjugateGaussianOEDExpectedStdDev(
            prior_mean, prior_cov, qoi_mat_typed, self._bkd
        )
        typed_utility.set_observation_matrix(obs_mat_typed)
        typed_utility.set_noise_covariance(noise_cov_typed)
        typed_value = typed_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray([typed_value]),
            self._bkd.asarray([float(legacy_value)]),
            rtol=1e-10,
        )

    def test_expected_entropic_dev_legacy_vs_typed(self) -> None:
        """Compare legacy and typed expected entropic deviation utilities."""
        legacy_bkd = self.get_backend()
        problem, prior, obs_mat, noise_cov, qoi_mat = self._setup_legacy_problem()
        prior_mean, prior_cov, obs_mat_typed, noise_cov_typed, qoi_mat_typed = (
            self._setup_typed_problem(prior, obs_mat, noise_cov, qoi_mat)
        )

        # Legacy utility
        legacy_utility = ConjugateGaussianOEDForNormalExpectedEntropicDev(
            prior, qoi_mat, self._lamda
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_value = legacy_utility.value()

        # Typed utility
        typed_utility = ConjugateGaussianOEDExpectedEntropicDev(
            prior_mean, prior_cov, qoi_mat_typed, self._lamda, self._bkd
        )
        typed_utility.set_observation_matrix(obs_mat_typed)
        typed_utility.set_noise_covariance(noise_cov_typed)
        typed_value = typed_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray([typed_value]),
            self._bkd.asarray([float(legacy_value)]),
            rtol=1e-10,
        )

    def test_expected_avar_dev_legacy_vs_typed(self) -> None:
        """Compare legacy and typed expected AVaR deviation utilities."""
        legacy_bkd = self.get_backend()
        problem, prior, obs_mat, noise_cov, qoi_mat = self._setup_legacy_problem()
        prior_mean, prior_cov, obs_mat_typed, noise_cov_typed, qoi_mat_typed = (
            self._setup_typed_problem(prior, obs_mat, noise_cov, qoi_mat)
        )

        # Legacy utility
        legacy_utility = ConjugateGaussianOEDForNormalExpectedAVaRDev(
            prior, qoi_mat, self._beta
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_value = legacy_utility.value()

        # Typed utility
        typed_utility = ConjugateGaussianOEDExpectedAVaRDev(
            prior_mean, prior_cov, qoi_mat_typed, self._beta, self._bkd
        )
        typed_utility.set_observation_matrix(obs_mat_typed)
        typed_utility.set_noise_covariance(noise_cov_typed)
        typed_value = typed_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray([typed_value]),
            self._bkd.asarray([float(legacy_value)]),
            rtol=1e-10,
        )

    def test_expected_kl_divergence_legacy_vs_typed(self) -> None:
        """Compare legacy and typed expected KL divergence utilities."""
        legacy_bkd = self.get_backend()
        problem, prior, obs_mat, noise_cov, qoi_mat = self._setup_legacy_problem()
        prior_mean, prior_cov, obs_mat_typed, noise_cov_typed, qoi_mat_typed = (
            self._setup_typed_problem(prior, obs_mat, noise_cov, qoi_mat)
        )

        # Legacy utility
        legacy_utility = ConjugateGaussianOEDForNormalExpectedKLDivergence(
            prior, qoi_mat
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_value = legacy_utility.value()

        # Typed utility
        typed_utility = ConjugateGaussianOEDExpectedKLDivergence(
            prior_mean, prior_cov, qoi_mat_typed, self._bkd
        )
        typed_utility.set_observation_matrix(obs_mat_typed)
        typed_utility.set_noise_covariance(noise_cov_typed)
        typed_value = typed_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray([typed_value]),
            self._bkd.asarray([float(legacy_value)]),
            rtol=1e-10,
        )

    def test_lognormal_expected_stdev_legacy_vs_typed(self) -> None:
        """Compare legacy and typed lognormal expected std dev utilities."""
        legacy_bkd = self.get_backend()
        problem, prior, obs_mat, noise_cov, qoi_mat = self._setup_legacy_problem()
        prior_mean, prior_cov, obs_mat_typed, noise_cov_typed, qoi_mat_typed = (
            self._setup_typed_problem(prior, obs_mat, noise_cov, qoi_mat)
        )

        # Legacy utility
        legacy_utility = ConjugateGaussianOEDForLogNormalExpectedStdDev(
            prior, qoi_mat
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_value = legacy_utility.value()

        # Typed utility
        typed_utility = TypedLogNormalStdDev(
            prior_mean, prior_cov, qoi_mat_typed, self._bkd
        )
        typed_utility.set_observation_matrix(obs_mat_typed)
        typed_utility.set_noise_covariance(noise_cov_typed)
        typed_value = typed_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray([typed_value]),
            self._bkd.asarray([float(legacy_value)]),
            rtol=1e-10,
        )

    def test_lognormal_avar_stdev_legacy_vs_typed(self) -> None:
        """Compare legacy and typed lognormal AVaR std dev utilities."""
        legacy_bkd = self.get_backend()
        problem, prior, obs_mat, noise_cov, qoi_mat = self._setup_legacy_problem()
        prior_mean, prior_cov, obs_mat_typed, noise_cov_typed, qoi_mat_typed = (
            self._setup_typed_problem(prior, obs_mat, noise_cov, qoi_mat)
        )

        # Legacy utility
        legacy_utility = ConjugateGaussianOEDForLogNormalAVaRStdDev(
            prior, qoi_mat
        )
        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_value = legacy_utility.value()

        # Typed utility
        typed_utility = TypedLogNormalAVaR(
            prior_mean, prior_cov, qoi_mat_typed, self._beta, self._bkd
        )
        typed_utility.set_observation_matrix(obs_mat_typed)
        typed_utility.set_noise_covariance(noise_cov_typed)
        typed_value = typed_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray([typed_value]),
            self._bkd.asarray([float(legacy_value)]),
            rtol=1e-10,
        )

    def test_mc_stdev_matches_exact_legacy(self) -> None:
        """Test MC average of std dev matches exact formula (legacy approach)."""
        legacy_bkd = self.get_backend()
        problem, prior, obs_mat, noise_cov, qoi_mat = self._setup_legacy_problem()

        # Generate samples (legacy approach)
        samples = prior.rvs(self._nsamples)
        obs = obs_mat @ samples + legacy_bkd.cholesky(noise_cov) @ legacy_bkd.array(
            np.random.normal(0, 1, (self._nobs, self._nsamples))
        )

        # Compute posterior pushforward for each observation realization
        posterior = DenseMatrixLaplacePosteriorApproximation(
            obs_mat, prior.mean(), prior.covariance(), noise_cov, backend=legacy_bkd
        )
        stdevs = []
        for ii in range(self._nsamples):
            posterior.compute(obs[:, ii : ii + 1])
            post_push = GaussianPushForward(
                qoi_mat,
                posterior.posterior_mean(),
                posterior.posterior_covariance(),
                backend=legacy_bkd,
            )
            stdevs.append(legacy_bkd.sqrt(post_push.covariance()[0, 0]))

        # Compute exact
        std_utility = ConjugateGaussianOEDForNormalExpectedStdDev(prior, qoi_mat)
        std_utility.set_observation_matrix(obs_mat)
        std_utility.set_noise_covariance(noise_cov)

        # Legacy assertion: stdevs are constant (independent of data)
        mc_mean = legacy_bkd.array(stdevs).mean()
        exact = std_utility.value()

        self.assertTrue(
            legacy_bkd.allclose(mc_mean, exact),
            f"MC mean stdev ({float(mc_mean)}) != exact ({float(exact)})",
        )

    def test_mc_avar_dev_matches_exact_legacy(self) -> None:
        """Test MC average of AVaR deviation matches exact formula (legacy approach)."""
        legacy_bkd = self.get_backend()
        problem, prior, obs_mat, noise_cov, qoi_mat = self._setup_legacy_problem()

        # Generate samples
        samples = prior.rvs(self._nsamples)
        obs = obs_mat @ samples + legacy_bkd.cholesky(noise_cov) @ legacy_bkd.array(
            np.random.normal(0, 1, (self._nobs, self._nsamples))
        )

        # Compute posterior pushforward for each observation realization
        posterior = DenseMatrixLaplacePosteriorApproximation(
            obs_mat, prior.mean(), prior.covariance(), noise_cov, backend=legacy_bkd
        )
        avardevs = []
        for ii in range(self._nsamples):
            posterior.compute(obs[:, ii : ii + 1])
            post_push = GaussianPushForward(
                qoi_mat,
                posterior.posterior_mean(),
                posterior.posterior_covariance(),
                backend=legacy_bkd,
            )
            stdev = legacy_bkd.sqrt(post_push.covariance()[0, 0])
            normal_risks = GaussianAnalyticalRiskMeasures(
                legacy_bkd.to_numpy(post_push.mean()[0, 0]),
                legacy_bkd.to_numpy(stdev),
            )
            # AVaR deviation = AVaR - mean
            avardevs.append(normal_risks.AVaR(self._beta) - post_push.mean()[0, 0])

        # Compute exact
        avardev_utility = ConjugateGaussianOEDForNormalExpectedAVaRDev(
            prior, qoi_mat, self._beta
        )
        avardev_utility.set_observation_matrix(obs_mat)
        avardev_utility.set_noise_covariance(noise_cov)

        mc_mean = legacy_bkd.array(avardevs).mean()
        exact = avardev_utility.value()

        self.assertTrue(
            legacy_bkd.allclose(mc_mean, exact, rtol=1e-2),
            f"MC mean AVaR dev ({float(mc_mean)}) != exact ({float(exact)})",
        )

    def test_mc_kl_divergence_matches_exact_legacy(self) -> None:
        """Test MC average of KL divergence matches exact formula (legacy approach)."""
        legacy_bkd = self.get_backend()
        problem, prior, obs_mat, noise_cov, qoi_mat = self._setup_legacy_problem()

        prior_push = GaussianPushForward(
            qoi_mat,
            prior.mean(),
            prior.covariance(),
            backend=legacy_bkd,
        )
        # Convert to DenseCholeskyMultivariateGaussian for kl_divergence
        prior_push_var = prior_push.pushforward_variable()

        # Generate samples
        samples = prior.rvs(self._nsamples)
        obs = obs_mat @ samples + legacy_bkd.cholesky(noise_cov) @ legacy_bkd.array(
            np.random.normal(0, 1, (self._nobs, self._nsamples))
        )

        # Compute posterior pushforward for each observation realization
        posterior = DenseMatrixLaplacePosteriorApproximation(
            obs_mat, prior.mean(), prior.covariance(), noise_cov, backend=legacy_bkd
        )
        kl_divs = []
        for ii in range(self._nsamples):
            posterior.compute(obs[:, ii : ii + 1])
            post_push = GaussianPushForward(
                qoi_mat,
                posterior.posterior_mean(),
                posterior.posterior_covariance(),
                backend=legacy_bkd,
            )
            # KL divergence between posterior pushforward and prior pushforward
            # Need to use pushforward_variable() on both sides to get
            # DenseCholeskyMultivariateGaussian which has kl_divergence
            kl_divs.append(
                post_push.pushforward_variable().kl_divergence(prior_push_var)
            )

        # Compute exact
        kl_utility = ConjugateGaussianOEDForNormalExpectedKLDivergence(
            prior, qoi_mat
        )
        kl_utility.set_observation_matrix(obs_mat)
        kl_utility.set_noise_covariance(noise_cov)

        mc_mean = legacy_bkd.array(kl_divs).mean()
        exact = kl_utility.value()

        # KL divergence has higher variance, need larger tolerance with 100 samples
        # With 10,000 samples (legacy), rtol=1e-2 works; with 100 samples need ~5%
        self.assertTrue(
            legacy_bkd.allclose(mc_mean, exact, rtol=5e-2),
            f"MC mean KL ({float(mc_mean)}) != exact ({float(exact)})",
        )


class TestConjugateMCExactLegacyNumpy(TestConjugateMCExactLegacy[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

    def get_backend(self):
        from pyapprox.util.backends.numpy import NumpyMixin
        return NumpyMixin


class TestConjugateMCExactLegacyTorch(TestConjugateMCExactLegacy[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

    def get_backend(self):
        from pyapprox.util.backends.torch import TorchMixin
        return TorchMixin


if __name__ == "__main__":
    unittest.main()
