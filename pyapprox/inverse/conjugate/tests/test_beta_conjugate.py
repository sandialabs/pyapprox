"""
Tests for Beta conjugate posterior.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray
from scipy import stats

from pyapprox.inverse.conjugate.beta import BetaConjugatePosterior
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestBetaConjugateBase(Generic[Array], unittest.TestCase):
    """Base test class for BetaConjugatePosterior."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        # Uniform prior Beta(1, 1)
        self.alpha_prior = 1.0
        self.beta_prior = 1.0
        self.solver = BetaConjugatePosterior(
            self.alpha_prior, self.beta_prior, self.bkd()
        )

        # Observations: 3 successes, 2 failures
        self.obs = self.bkd().asarray([[1, 1, 0, 1, 0]])

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        self.assertEqual(self.solver.nvars(), 1)

    def test_posterior_alpha(self) -> None:
        """Test posterior alpha = prior_alpha + successes."""
        self.solver.compute(self.obs)
        # 3 successes
        self.assertAlmostEqual(self.solver.posterior_alpha(), 1.0 + 3.0, places=5)

    def test_posterior_beta(self) -> None:
        """Test posterior beta = prior_beta + failures."""
        self.solver.compute(self.obs)
        # 2 failures
        self.assertAlmostEqual(self.solver.posterior_beta(), 1.0 + 2.0, places=5)

    def test_posterior_mean(self) -> None:
        """Test posterior mean = alpha / (alpha + beta)."""
        self.solver.compute(self.obs)
        expected = 4.0 / (4.0 + 3.0)  # Beta(4, 3)
        self.assertAlmostEqual(self.solver.posterior_mean(), expected, places=5)

    def test_posterior_variance(self) -> None:
        """Test posterior variance formula."""
        self.solver.compute(self.obs)
        a, b = 4.0, 3.0  # Posterior Beta(4, 3)
        expected = (a * b) / ((a + b) ** 2 * (a + b + 1))
        self.assertAlmostEqual(self.solver.posterior_variance(), expected, places=5)

    def test_evidence_positive(self) -> None:
        """Test evidence is positive."""
        self.solver.compute(self.obs)
        self.assertGreater(self.solver.evidence(), 0)

    def test_posterior_variable(self) -> None:
        """Test posterior_variable returns scipy Beta distribution."""
        self.solver.compute(self.obs)
        post = self.solver.posterior_variable()
        # Check it's callable (has pdf method)
        self.assertTrue(callable(post.pdf))

    def test_compute_not_called_raises(self) -> None:
        """Test accessing results before compute raises error."""
        with self.assertRaises(RuntimeError):
            self.solver.posterior_alpha()


class TestBetaConjugateVsScipy(Generic[Array], unittest.TestCase):
    """Test against scipy.stats.beta."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_posterior_matches_scipy(self) -> None:
        """Test posterior matches scipy Beta."""
        alpha, beta = 2.0, 3.0
        solver = BetaConjugatePosterior(alpha, beta, self.bkd())

        obs = self.bkd().asarray([[1, 0, 1, 1, 0, 0]])  # 3 successes, 3 failures
        solver.compute(obs)

        # Expected: Beta(2+3, 3+3) = Beta(5, 6)
        expected_dist = stats.beta(5, 6)

        self.assertAlmostEqual(solver.posterior_mean(), expected_dist.mean(), places=5)
        self.assertAlmostEqual(
            solver.posterior_variance(), expected_dist.var(), places=5
        )


# NumPy backend tests
class TestBetaConjugateNumpy(TestBetaConjugateBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestBetaConjugateVsScipyNumpy(TestBetaConjugateVsScipy[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestBetaConjugateTorch(TestBetaConjugateBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestBetaConjugateVsScipyTorch(TestBetaConjugateVsScipy[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
