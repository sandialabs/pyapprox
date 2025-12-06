import unittest
from typing import Generic

import numpy as np
import torch
from scipy.stats import norm, beta

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.variables.univariate.scipy_continuous import (
    ContinuousScipyRandomVariable1D,
)


class TestContinuousScipyRandomVariable1D(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_pdf(self):
        """
        Test the PDF computation for bounded and unbounded variables.
        """
        # Unbounded variable: Normal distribution
        norm_rv = norm(loc=0, scale=1)
        norm_marginal = ContinuousScipyRandomVariable1D(
            norm_rv, bkd=self.bkd()
        )

        samples = self.bkd().linspace(-3, 3, 100)
        pdf_vals = norm_marginal.pdf(samples)
        expected_pdf_vals = self.bkd().asarray(norm_rv.pdf(samples))
        self.bkd().assert_allclose(pdf_vals, expected_pdf_vals)

        # Bounded variable: Beta distribution
        beta_rv = beta(a=2.0, b=5.0, loc=0.0, scale=1.0)
        beta_marginal = ContinuousScipyRandomVariable1D(
            beta_rv, bkd=self.bkd()
        )

        samples = self.bkd().linspace(0, 1, 100)
        pdf_vals = beta_marginal.pdf(samples)
        expected_pdf_vals = self.bkd().asarray(beta_rv.pdf(samples))
        self.bkd().assert_allclose(pdf_vals, expected_pdf_vals)

    def test_logpdf(self):
        """
        Test the log PDF computation for bounded and unbounded variables.
        """
        # Unbounded variable: Normal distribution
        norm_rv = norm(loc=0, scale=1)
        norm_marginal = ContinuousScipyRandomVariable1D(
            norm_rv, bkd=self.bkd()
        )

        samples = self.bkd().linspace(-3, 3, 100)
        logpdf_vals = norm_marginal.logpdf(samples)
        expected_logpdf_vals = self.bkd().asarray(norm_rv.logpdf(samples))
        self.bkd().assert_allclose(logpdf_vals, expected_logpdf_vals)

        # Bounded variable: Beta distribution
        beta_rv = beta(a=2.0, b=5.0, loc=0.0, scale=1.0)
        beta_marginal = ContinuousScipyRandomVariable1D(
            beta_rv, bkd=self.bkd()
        )

        samples = self.bkd().linspace(0, 1, 100)
        logpdf_vals = beta_marginal.logpdf(samples)
        expected_logpdf_vals = self.bkd().asarray(beta_rv.logpdf(samples))
        self.bkd().assert_allclose(logpdf_vals, expected_logpdf_vals)

    def test_cdf(self):
        """
        Test the CDF computation for bounded and unbounded variables.
        """
        # Unbounded variable: Normal distribution
        norm_rv = norm(loc=0, scale=1)
        norm_marginal = ContinuousScipyRandomVariable1D(
            norm_rv, bkd=self.bkd()
        )

        samples = self.bkd().linspace(-3, 3, 100)
        cdf_vals = norm_marginal.cdf(samples)
        expected_cdf_vals = self.bkd().asarray(norm_rv.cdf(samples))
        self.bkd().assert_allclose(cdf_vals, expected_cdf_vals)

        # Bounded variable: Beta distribution
        beta_rv = beta(a=2.0, b=5.0, loc=0.0, scale=1.0)
        beta_marginal = ContinuousScipyRandomVariable1D(
            beta_rv, bkd=self.bkd()
        )

        samples = self.bkd().linspace(0, 1, 100)
        cdf_vals = beta_marginal.cdf(samples)
        expected_cdf_vals = self.bkd().asarray(beta_rv.cdf(samples))
        self.bkd().assert_allclose(cdf_vals, expected_cdf_vals)

    def test_ppf(self):
        """
        Test the PPF computation for bounded and unbounded variables.
        """
        # Unbounded variable: Normal distribution
        norm_rv = norm(loc=0, scale=1)
        norm_marginal = ContinuousScipyRandomVariable1D(
            norm_rv, bkd=self.bkd()
        )

        usamples = self.bkd().linspace(0, 1, 100)
        ppf_vals = norm_marginal.ppf(usamples)
        expected_ppf_vals = self.bkd().asarray(norm_rv.ppf(usamples))
        self.bkd().assert_allclose(ppf_vals, expected_ppf_vals)

        # Bounded variable: Beta distribution
        beta_rv = beta(a=2.0, b=5.0, loc=0.0, scale=1.0)
        beta_marginal = ContinuousScipyRandomVariable1D(
            beta_rv, bkd=self.bkd()
        )

        usamples = self.bkd().linspace(0, 1, 100)
        ppf_vals = beta_marginal.ppf(usamples)
        expected_ppf_vals = self.bkd().asarray(beta_rv.ppf(usamples))
        self.bkd().assert_allclose(ppf_vals, expected_ppf_vals)

    def test_rvs(self):
        """
        Test random sampling for bounded and unbounded variables.
        """
        # Unbounded variable: Normal distribution
        norm_rv = norm(loc=0, scale=1)
        norm_marginal = ContinuousScipyRandomVariable1D(
            norm_rv, bkd=self.bkd()
        )

        nsamples = 10
        samples = norm_marginal.rvs(nsamples)
        self.assertEqual(samples.shape, (nsamples,))

        # Bounded variable: Beta distribution
        beta_rv = beta(a=2.0, b=5.0, loc=0.0, scale=1.0)
        beta_marginal = ContinuousScipyRandomVariable1D(
            beta_rv, bkd=self.bkd()
        )

        samples = beta_marginal.rvs(nsamples)
        self.assertEqual(samples.shape, (nsamples,))

    def test_mean(self):
        """
        Test the mean computation for bounded and unbounded variables.
        """
        # Unbounded variable: Normal distribution
        norm_rv = norm(loc=0, scale=1)
        norm_marginal = ContinuousScipyRandomVariable1D(
            norm_rv, bkd=self.bkd()
        )

        mean = norm_marginal.mean()
        expected_mean = self.bkd().asarray([norm_rv.mean()])
        self.bkd().assert_allclose(mean, expected_mean)

        # Bounded variable: Beta distribution
        beta_rv = beta(a=2.0, b=5.0, loc=0.0, scale=1.0)
        beta_marginal = ContinuousScipyRandomVariable1D(
            beta_rv, bkd=self.bkd()
        )

        mean = beta_marginal.mean()
        expected_mean = self.bkd().asarray([beta_rv.mean()])
        self.bkd().assert_allclose(mean, expected_mean)

    def test_is_bounded(self):
        """
        Test boundedness of the variables.
        """
        # Unbounded variable: Normal distribution
        norm_rv = norm(loc=0, scale=1)
        norm_marginal = ContinuousScipyRandomVariable1D(
            norm_rv, bkd=self.bkd()
        )
        self.assertFalse(norm_marginal.is_bounded())

        # Bounded variable: Beta distribution
        beta_rv = beta(a=2.0, b=5.0, loc=0.0, scale=1.0)
        beta_marginal = ContinuousScipyRandomVariable1D(
            beta_rv, bkd=self.bkd()
        )
        self.assertTrue(beta_marginal.is_bounded())


# Derived test class for NumPy backend
class TestContinuousScipyRandomVariable1DNumpy(
    TestContinuousScipyRandomVariable1D[Array]
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestContinuousScipyRandomVariable1DTorch(
    TestContinuousScipyRandomVariable1D[torch.Tensor]
):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
