import unittest
from typing import Generic, Any

import numpy as np
from scipy.stats import norm, binom, rv_continuous, rv_discrete
import torch

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.torch import TorchBkd
from pyapprox.typing.variables.univariate.scipy import (
    ScipyRandomVariable1D,
    ContinuousScipyRandomVariable1D,
    DiscreteScipyRandomVariable1D,
)
from pyapprox.typing.util.abstracttestcase import AbstractTestCase


class TestScipyRandomVariable1D(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_continuous_scipy_marginal(self):
        """
        Test the functionality of ContinuousScipyRandomVariable1D.
        """
        scipy_rv = norm(loc=0, scale=1)
        marginal = ContinuousScipyRandomVariable1D(scipy_rv, bkd=self.bkd())

        # Test PDF
        samples = self.bkd().linspace(-3, 3, 100)
        pdf_vals = marginal.pdf(samples)
        expected_pdf_vals = self.bkd().asarray(scipy_rv.pdf(samples))
        self.bkd().assert_allclose(pdf_vals, expected_pdf_vals)

        # Test log PDF
        logpdf_vals = marginal.logpdf(samples)
        expected_logpdf_vals = self.bkd().asarray(scipy_rv.logpdf(samples))
        self.bkd().assert_allclose(logpdf_vals, expected_logpdf_vals)

        # Test CDF
        cdf_vals = marginal.cdf(samples)
        expected_cdf_vals = self.bkd().asarray(scipy_rv.cdf(samples))
        self.bkd().assert_allclose(cdf_vals, expected_cdf_vals)

        # Test interval
        interval = marginal.interval(0.95)
        expected_interval = self.bkd().asarray(scipy_rv.interval(0.95))
        self.bkd().assert_allclose(interval, expected_interval)

        # Test sampling
        nsamples = 10
        samples = marginal.rvs(nsamples)
        self.assertEqual(samples.shape, (nsamples,))

        # Test boundedness
        self.assertTrue(marginal.is_bounded())

    def test_discrete_scipy_marginal(self):
        """
        Test the functionality of DiscreteScipyRandomVariable1D.
        """
        scipy_rv = binom(n=10, p=0.5)
        marginal = DiscreteScipyRandomVariable1D(scipy_rv, bkd=self.bkd())

        # Test PMF (PDF for discrete variables)
        samples = self.bkd().arange(0, 11)
        pmf_vals = marginal.pdf(samples)
        expected_pmf_vals = scipy_rv.pmf(samples)
        self.bkd().assert_allclose(pmf_vals, expected_pmf_vals)

        # Test interval
        interval = marginal.interval(0.95)
        expected_interval = self.bkd().asarray(scipy_rv.interval(0.95))
        self.bkd().assert_allclose(interval, expected_interval)

        # Test sampling
        nsamples = 10
        samples = marginal.rvs(nsamples)
        self.assertEqual(samples.shape, (nsamples,))

        # Test boundedness
        self.assertTrue(marginal.is_bounded())

        # Test probability masses
        xk, pk = marginal.probability_masses()
        expected_xk = self.bkd().arange(
            0.0, 11.0, dtype=self.bkd().double_dtype()
        )
        expected_pk = self.bkd().asarray(scipy_rv.pmf(expected_xk))
        self.bkd().assert_allclose(xk, expected_xk)
        self.bkd().assert_allclose(pk, expected_pk)

    def test_equality(self):
        """
        Test equality of SciPy random variables.
        """
        scipy_rv1 = norm(loc=0, scale=1)
        scipy_rv2 = norm(loc=0, scale=1)
        scipy_rv3 = norm(loc=1, scale=1)

        marginal1 = ContinuousScipyRandomVariable1D(scipy_rv1, bkd=self.bkd())
        marginal2 = ContinuousScipyRandomVariable1D(scipy_rv2, bkd=self.bkd())
        marginal3 = ContinuousScipyRandomVariable1D(scipy_rv3, bkd=self.bkd())

        self.assertTrue(marginal1 == marginal2)
        self.assertFalse(marginal1 == marginal3)

    def test_repr(self):
        """
        Test string representation of SciPy random variables.
        """
        scipy_rv = norm(loc=0, scale=1)
        marginal = ContinuousScipyRandomVariable1D(scipy_rv, bkd=self.bkd())
        self.assertEqual(
            repr(marginal), "ContinuousScipyRandomVariable1D(name=norm)"
        )


# Derived test class for NumPy backend
class TestScipyRandomVariable1DNumpy(
    TestScipyRandomVariable1D[Array], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestScipyRandomVariable1DTorch(
    TestScipyRandomVariable1D[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
