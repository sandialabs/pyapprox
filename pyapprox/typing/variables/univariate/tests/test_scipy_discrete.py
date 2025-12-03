import unittest
from typing import Generic

import numpy as np
import torch

from scipy.stats import binom, poisson
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.variables.univariate.scipy_discrete import (
    DiscreteScipyRandomVariable1D,
)
from pyapprox.typing.util.abstracttestcase import AbstractTestCase


class TestDiscreteScipyRandomVariable1D(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_pdf(self):
        """
        Test the PMF computation for bounded and unbounded variables.
        """
        # Bounded variable: Binomial distribution
        binom_rv = binom(n=10, p=0.5)
        binom_marginal = DiscreteScipyRandomVariable1D(
            binom_rv, bkd=self.bkd()
        )

        samples = self.bkd().arange(0, 11)
        pmf_vals = binom_marginal(samples)
        expected_pmf_vals = self.bkd().asarray(binom_rv.pmf(samples))
        self.bkd().assert_allclose(pmf_vals, expected_pmf_vals)

        # Unbounded variable: Poisson distribution
        poisson_rv = poisson(mu=3.0)
        poisson_marginal = DiscreteScipyRandomVariable1D(
            poisson_rv, bkd=self.bkd()
        )

        samples = self.bkd().arange(0, 20)
        pmf_vals = poisson_marginal(samples)
        expected_pmf_vals = self.bkd().asarray(poisson_rv.pmf(samples))
        self.bkd().assert_allclose(pmf_vals, expected_pmf_vals)

    def test_is_bounded(self):
        """
        Test boundedness of the variables.
        """
        # Bounded variable: Binomial distribution
        binom_rv = binom(n=10, p=0.5)
        binom_marginal = DiscreteScipyRandomVariable1D(
            binom_rv, bkd=self.bkd()
        )
        self.assertTrue(binom_marginal.is_bounded())

        # Unbounded variable: Poisson distribution
        poisson_rv = poisson(mu=3.0)
        poisson_marginal = DiscreteScipyRandomVariable1D(
            poisson_rv, bkd=self.bkd()
        )
        self.assertFalse(poisson_marginal.is_bounded())


# Derived test class for NumPy backend
class TestDiscreteScipyRandomVariable1DNumpy(
    TestDiscreteScipyRandomVariable1D[Array], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestDiscreteScipyRandomVariable1DTorch(
    TestDiscreteScipyRandomVariable1D[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
