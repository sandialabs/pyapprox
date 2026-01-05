"""
Tests for PyTorch autograd compatibility with probability distributions.

These tests verify that logpdf and other methods work correctly with
PyTorch's automatic differentiation.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.probability.univariate import GaussianMarginal
from pyapprox.typing.probability.gaussian import (
    DenseCholeskyMultivariateGaussian,
    DiagonalMultivariateGaussian,
)


class TestGaussianMarginalAutograd(unittest.TestCase):
    """Test autograd compatibility for GaussianMarginal."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        self.mean = 2.0
        self.stdev = 0.5
        self.dist = GaussianMarginal(self.mean, self.stdev, self._bkd)

    def test_logpdf_gradient_wrt_samples(self) -> None:
        """Test autograd gradient of logpdf w.r.t. sample values."""
        # Shape: (1, nsamples) for univariate distribution
        samples = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        logpdf = self.dist.logpdf(samples)  # Returns (1, nsamples)

        # Sum to get scalar for backward
        loss = logpdf.sum()
        loss.backward()

        # Analytical gradient: d/dx log(pdf) = -(x - mu) / sigma^2
        expected_grad = -(samples.detach() - self.mean) / (self.stdev**2)

        self.assertTrue(
            torch.allclose(samples.grad, expected_grad, rtol=1e-6)
        )

    def test_pdf_gradient_wrt_samples(self) -> None:
        """Test autograd gradient of pdf w.r.t. sample values."""
        # Shape: (1, nsamples) for univariate distribution
        samples = torch.tensor([[1.5, 2.0, 2.5]], requires_grad=True)
        pdf = self.dist(samples)  # Returns (1, nsamples)

        loss = pdf.sum()
        loss.backward()

        # Analytical gradient: d/dx pdf = pdf * (-(x - mu) / sigma^2)
        pdf_vals = self.dist(samples.detach())
        expected_grad = pdf_vals * (-(samples.detach() - self.mean) / (self.stdev**2))

        self.assertTrue(
            torch.allclose(samples.grad, expected_grad, rtol=1e-6)
        )

    def test_cdf_gradient_wrt_samples(self) -> None:
        """Test autograd gradient of cdf w.r.t. sample values."""
        # Shape: (1, nsamples) for univariate distribution
        samples = torch.tensor([[1.5, 2.0, 2.5]], requires_grad=True)
        cdf = self.dist.cdf(samples)  # Returns (1, nsamples)

        loss = cdf.sum()
        loss.backward()

        # Analytical gradient: d/dx CDF = pdf
        expected_grad = self.dist(samples.detach())

        self.assertTrue(
            torch.allclose(samples.grad, expected_grad, rtol=1e-6)
        )

    def test_invcdf_gradient_wrt_probs(self) -> None:
        """Test autograd gradient of invcdf w.r.t. probabilities."""
        # Shape: (1, nsamples) for univariate distribution
        probs = torch.tensor([[0.25, 0.5, 0.75]], requires_grad=True)
        quantiles = self.dist.invcdf(probs)  # Returns (1, nsamples)

        loss = quantiles.sum()
        loss.backward()

        # Analytical gradient: d/dp invcdf = 1 / pdf(invcdf(p))
        quantile_vals = self.dist.invcdf(probs.detach())
        expected_grad = 1.0 / self.dist(quantile_vals)

        self.assertTrue(
            torch.allclose(probs.grad, expected_grad, rtol=1e-5)
        )

    def test_logpdf_second_derivative(self) -> None:
        """Test second derivative of logpdf is computable."""
        # Shape: (1, nsamples) for univariate distribution
        samples = torch.tensor([[1.5, 2.0, 2.5]], requires_grad=True)

        # First derivative
        logpdf = self.dist.logpdf(samples)  # Returns (1, nsamples)
        grad1 = torch.autograd.grad(
            logpdf.sum(), samples, create_graph=True
        )[0]

        # Second derivative
        grad2 = torch.autograd.grad(grad1.sum(), samples)[0]

        # Analytical: d^2/dx^2 log(pdf) = -1/sigma^2
        expected = torch.full_like(samples, -1.0 / (self.stdev**2))

        self.assertTrue(
            torch.allclose(grad2, expected, rtol=1e-6)
        )


class TestGaussianMarginalAutogradFiniteDiff(unittest.TestCase):
    """Test autograd gradients match finite differences."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        self.mean = 1.0
        self.stdev = 2.0
        self.dist = GaussianMarginal(self.mean, self.stdev, self._bkd)

    def test_logpdf_gradient_finite_diff(self) -> None:
        """Compare autograd gradient to finite difference."""
        # Shape: (1, nsamples) for univariate distribution
        samples = torch.tensor([[0.5, 1.0, 1.5]], requires_grad=True)
        logpdf = self.dist.logpdf(samples)  # Returns (1, nsamples)
        loss = logpdf.sum()
        loss.backward()
        autograd_grad = samples.grad.clone()

        # Finite difference
        eps = 1e-6
        fd_grad = torch.zeros_like(samples)
        for i in range(samples.shape[1]):
            samples_plus = samples.detach().clone()
            samples_minus = samples.detach().clone()
            samples_plus[0, i] += eps
            samples_minus[0, i] -= eps
            fd_grad[0, i] = (
                self.dist.logpdf(samples_plus).sum()
                - self.dist.logpdf(samples_minus).sum()
            ) / (2 * eps)

        self.assertTrue(
            torch.allclose(autograd_grad, fd_grad, rtol=1e-5)
        )


class TestDenseCholeskyMultivariateGaussianAutograd(unittest.TestCase):
    """Test autograd compatibility for DenseCholeskyMultivariateGaussian."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        self.nvars = 3
        self.mean = torch.tensor([[1.0], [2.0], [3.0]])
        self.cov = torch.tensor([
            [2.0, 0.5, 0.1],
            [0.5, 1.5, 0.3],
            [0.1, 0.3, 1.0],
        ])
        self.dist = DenseCholeskyMultivariateGaussian(
            self.mean, self.cov, self._bkd
        )

    def test_logpdf_gradient_wrt_samples(self) -> None:
        """Test autograd gradient of logpdf w.r.t. sample values."""
        samples = torch.tensor(
            [[0.5, 1.5], [1.0, 2.5], [2.0, 3.5]],
            requires_grad=True,
        )
        logpdf = self.dist.logpdf(samples)

        loss = logpdf.sum()
        loss.backward()

        # Compare to analytical gradient
        expected_grad = self.dist.logpdf_gradient(samples.detach())

        self.assertTrue(
            torch.allclose(samples.grad, expected_grad, rtol=1e-6)
        )

    def test_logpdf_gradient_finite_diff(self) -> None:
        """Compare autograd gradient to finite difference."""
        samples = torch.tensor(
            [[0.5], [1.0], [2.0]],
            requires_grad=True,
        )
        logpdf = self.dist.logpdf(samples)
        loss = logpdf.sum()
        loss.backward()
        autograd_grad = samples.grad.clone()

        # Finite difference
        eps = 1e-6
        fd_grad = torch.zeros_like(samples)
        for i in range(self.nvars):
            samples_plus = samples.detach().clone()
            samples_minus = samples.detach().clone()
            samples_plus[i, 0] += eps
            samples_minus[i, 0] -= eps
            fd_grad[i, 0] = (
                self.dist.logpdf(samples_plus).sum()
                - self.dist.logpdf(samples_minus).sum()
            ) / (2 * eps)

        self.assertTrue(
            torch.allclose(autograd_grad, fd_grad, rtol=1e-5)
        )

    def test_pdf_gradient_wrt_samples(self) -> None:
        """Test autograd gradient of pdf w.r.t. sample values."""
        samples = torch.tensor(
            [[1.0], [2.0], [3.0]],
            requires_grad=True,
        )
        pdf = self.dist.pdf(samples)

        loss = pdf.sum()
        loss.backward()

        # Analytical gradient: d/dx pdf = pdf * grad_log_pdf
        pdf_val = self.dist.pdf(samples.detach())
        logpdf_grad = self.dist.logpdf_gradient(samples.detach())
        expected_grad = pdf_val * logpdf_grad

        self.assertTrue(
            torch.allclose(samples.grad, expected_grad, rtol=1e-6)
        )


class TestDiagonalMultivariateGaussianAutograd(unittest.TestCase):
    """Test autograd compatibility for DiagonalMultivariateGaussian."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        self.nvars = 3
        self.mean = torch.tensor([[0.0], [1.0], [2.0]])
        self.variance = torch.tensor([1.0, 2.0, 0.5])
        self.dist = DiagonalMultivariateGaussian(
            self.mean, self.variance, self._bkd
        )

    def test_logpdf_gradient_wrt_samples(self) -> None:
        """Test autograd gradient of logpdf w.r.t. sample values."""
        samples = torch.tensor(
            [[0.5, -0.5], [1.5, 0.5], [2.5, 1.5]],
            requires_grad=True,
        )
        logpdf = self.dist.logpdf(samples)

        loss = logpdf.sum()
        loss.backward()

        # Compare to analytical gradient
        expected_grad = self.dist.logpdf_gradient(samples.detach())

        self.assertTrue(
            torch.allclose(samples.grad, expected_grad, rtol=1e-6)
        )

    def test_logpdf_gradient_finite_diff(self) -> None:
        """Compare autograd gradient to finite difference."""
        samples = torch.tensor(
            [[0.5], [1.0], [2.0]],
            requires_grad=True,
        )
        logpdf = self.dist.logpdf(samples)
        loss = logpdf.sum()
        loss.backward()
        autograd_grad = samples.grad.clone()

        # Finite difference
        eps = 1e-6
        fd_grad = torch.zeros_like(samples)
        for i in range(self.nvars):
            samples_plus = samples.detach().clone()
            samples_minus = samples.detach().clone()
            samples_plus[i, 0] += eps
            samples_minus[i, 0] -= eps
            fd_grad[i, 0] = (
                self.dist.logpdf(samples_plus).sum()
                - self.dist.logpdf(samples_minus).sum()
            ) / (2 * eps)

        self.assertTrue(
            torch.allclose(autograd_grad, fd_grad, rtol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
