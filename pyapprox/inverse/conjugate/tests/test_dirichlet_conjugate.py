"""
Tests for Dirichlet conjugate posterior.
"""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch
from scipy import stats

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.inverse.conjugate.dirichlet import DirichletConjugatePosterior


class TestDirichletConjugateBase(Generic[Array], unittest.TestCase):
    """Base test class for DirichletConjugatePosterior."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        # Uniform prior over 3 categories
        self.alphas_prior = self.bkd().asarray([1.0, 1.0, 1.0])
        self.solver = DirichletConjugatePosterior(self.alphas_prior, self.bkd())

        # Observations: category indices
        # 3 in cat 0, 1 in cat 1, 1 in cat 2
        self.obs = self.bkd().asarray([[0, 0, 1, 2, 0]])

    def test_nvars(self) -> None:
        """Test nvars returns number of categories."""
        self.assertEqual(self.solver.nvars(), 3)

    def test_posterior_alphas(self) -> None:
        """Test posterior alphas = prior_alphas + counts."""
        self.solver.compute(self.obs)
        post_alphas = self.bkd().to_numpy(self.solver.posterior_alphas())
        # Expected: [1+3, 1+1, 1+1] = [4, 2, 2]
        np.testing.assert_array_almost_equal(post_alphas, [4.0, 2.0, 2.0])

    def test_posterior_mean(self) -> None:
        """Test posterior mean = alpha_k / sum(alpha)."""
        self.solver.compute(self.obs)
        post_mean = self.bkd().to_numpy(self.solver.posterior_mean())
        # Expected: [4/8, 2/8, 2/8] = [0.5, 0.25, 0.25]
        np.testing.assert_array_almost_equal(post_mean, [0.5, 0.25, 0.25])

    def test_posterior_mean_sums_to_one(self) -> None:
        """Test posterior mean is a proper probability distribution."""
        self.solver.compute(self.obs)
        post_mean = self.bkd().to_numpy(self.solver.posterior_mean())
        self.assertAlmostEqual(sum(post_mean), 1.0, places=5)

    def test_evidence_positive(self) -> None:
        """Test evidence is positive."""
        self.solver.compute(self.obs)
        self.assertGreater(self.solver.evidence(), 0)

    def test_posterior_variable(self) -> None:
        """Test posterior_variable returns scipy Dirichlet distribution."""
        self.solver.compute(self.obs)
        post = self.solver.posterior_variable()
        self.assertTrue(callable(post.pdf))

    def test_compute_not_called_raises(self) -> None:
        """Test accessing results before compute raises error."""
        with self.assertRaises(RuntimeError):
            self.solver.posterior_alphas()

    def test_invalid_category_raises(self) -> None:
        """Test observation with invalid category raises error."""
        bad_obs = self.bkd().asarray([[0, 1, 5]])  # 5 is out of range
        with self.assertRaises(ValueError):
            self.solver.compute(bad_obs)


class TestDirichletConjugateVsScipy(Generic[Array], unittest.TestCase):
    """Test against scipy.stats.dirichlet."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_posterior_matches_scipy(self) -> None:
        """Test posterior matches scipy Dirichlet."""
        alphas = self.bkd().asarray([2.0, 3.0, 1.0])
        solver = DirichletConjugatePosterior(alphas, self.bkd())

        # 2 in cat 0, 1 in cat 1, 3 in cat 2
        obs = self.bkd().asarray([[0, 0, 1, 2, 2, 2]])
        solver.compute(obs)

        # Expected: Dirichlet(2+2, 3+1, 1+3) = Dirichlet(4, 4, 4)
        expected_dist = stats.dirichlet([4, 4, 4])

        post_mean = self.bkd().to_numpy(solver.posterior_mean())
        np.testing.assert_array_almost_equal(post_mean, expected_dist.mean(), decimal=5)


# NumPy backend tests
class TestDirichletConjugateNumpy(TestDirichletConjugateBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestDirichletConjugateVsScipyNumpy(TestDirichletConjugateVsScipy[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestDirichletConjugateTorch(TestDirichletConjugateBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestDirichletConjugateVsScipyTorch(TestDirichletConjugateVsScipy[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
