import unittest
from typing import Generic, Any

import torch
from scipy.stats import norm
from numpy.typing import NDArray

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from pyapprox.typing.variables.univariate.gaussian import GaussianMarginal
from pyapprox.typing.variables.univariate.scipy_continuous import (
    ContinuousScipyRandomVariable1D,
)


class TestGaussianMarginal(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        self.mean = 0.0
        self.stdev = 1.0
        self.marginal = GaussianMarginal(self.mean, self.stdev, self.bkd())
        self.scipy_marginal = ContinuousScipyRandomVariable1D(
            norm(loc=self.mean, scale=self.stdev), self.bkd()
        )

    def test_pdf(self) -> None:
        """
        Test the PDF computation by comparing to SciPy marginal.
        """
        samples = self.bkd().linspace(-3, 3, 100)
        pdf_vals = self.marginal.pdf(samples)
        expected_pdf_vals = self.bkd().asarray(
            self.scipy_marginal.pdf(samples)
        )
        self.bkd().assert_allclose(pdf_vals, expected_pdf_vals)

    def test_logpdf(self) -> None:
        """
        Test the log PDF computation by comparing to SciPy marginal.
        """
        samples = self.bkd().linspace(-3, 3, 100)
        logpdf_vals = self.marginal.logpdf(samples)
        expected_logpdf_vals = self.bkd().asarray(
            self.scipy_marginal.logpdf(samples)
        )
        self.bkd().assert_allclose(logpdf_vals, expected_logpdf_vals)

    def test_cdf(self) -> None:
        """
        Test the CDF computation by comparing to SciPy marginal.
        """
        samples = self.bkd().linspace(-3, 3, 100)
        cdf_vals = self.marginal.cdf(samples)
        expected_cdf_vals = self.bkd().asarray(
            self.scipy_marginal.cdf(samples)
        )
        self.bkd().assert_allclose(cdf_vals, expected_cdf_vals)

    def test_ppf(self) -> None:
        """
        Test the PPF computation by comparing to SciPy marginal.
        """
        usamples = self.bkd().linspace(0, 1, 100)
        ppf_vals = self.marginal.ppf(usamples)
        expected_ppf_vals = self.bkd().asarray(
            self.scipy_marginal.ppf(usamples)
        )
        self.bkd().assert_allclose(ppf_vals, expected_ppf_vals)

    def test_rvs(self) -> None:
        """
        Test random sampling by comparing to SciPy marginal.
        """
        nsamples = 10
        samples = self.marginal.rvs(nsamples)
        scipy_samples = self.scipy_marginal.rvs(nsamples)
        self.assertEqual(samples.shape, (nsamples,))
        self.assertEqual(len(scipy_samples), nsamples)

    def test_mean(self) -> None:
        """
        Test the mean computation by comparing to SciPy marginal.
        """
        mean = self.marginal.mean()
        expected_mean = self.scipy_marginal.mean()
        self.assertEqual(mean, expected_mean)

    def test_median(self) -> None:
        """
        Test the median computation by comparing to SciPy marginal.
        """
        median = self.marginal.median()
        expected_median = self.scipy_marginal.median()
        self.assertEqual(median, expected_median)

    def test_var(self) -> None:
        """
        Test the variance computation by comparing to SciPy marginal.
        """
        var = self.marginal.var()
        expected_var = self.scipy_marginal.var()
        self.assertEqual(var, expected_var)

    def test_std(self) -> None:
        """
        Test the standard deviation computation by comparing to SciPy marginal.
        """
        std = self.marginal.std()
        expected_std = self.scipy_marginal.std()
        self.assertEqual(std, expected_std)

    def test_logpdf_jacobian(self) -> None:
        """
        Test the Jacobian of the log PDF by comparing to SciPy marginal.
        """
        samples = self.bkd().linspace(-3, 3, 100)
        logpdf_jacobian_vals = self.marginal.logpdf_jacobian(samples)
        expected_jacobian_vals = self.bkd().reshape(
            (-(samples - self.mean) / self.stdev**2), (1, -1)
        )
        self.bkd().assert_allclose(
            logpdf_jacobian_vals, expected_jacobian_vals
        )

    def test_equality(self) -> None:
        """
        Test equality of Gaussian distributions.
        """
        marginal1 = GaussianMarginal(self.mean, self.stdev, self.bkd())
        marginal2 = GaussianMarginal(self.mean, self.stdev, self.bkd())
        marginal3 = GaussianMarginal(self.mean + 1.0, self.stdev, self.bkd())

        self.assertTrue(marginal1 == marginal2)
        self.assertFalse(marginal1 == marginal3)

    def test_repr(self) -> None:
        """
        Test string representation of Gaussian distributions.
        """
        self.assertEqual(
            repr(self.marginal),
            f"GaussianMarginal(mean={self.mean}, stdev={self.stdev})",
        )


# Derived test class for NumPy backend
class TestGaussianMarginalNumpy(
    TestGaussianMarginal[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestGaussianMarginalTorch(
    TestGaussianMarginal[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
