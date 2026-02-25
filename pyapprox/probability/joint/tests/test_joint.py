"""
Tests for joint probability distributions.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.probability.joint import IndependentJoint
from pyapprox.probability.univariate import (
    BetaMarginal,
    GammaMarginal,
    GaussianMarginal,
    ScipyContinuousMarginal,
    UniformMarginal,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd

# Marginal combination test cases: (name, marginal_types)
# Each marginal_type is a tuple of (class, *args) for construction
MARGINAL_COMBOS = [
    ("1_gaussian", [("gaussian", 0.0, 1.0)]),
    ("1_beta", [("beta", 2.0, 5.0)]),
    ("1_gamma", [("gamma", 2.0, 1.0)]),
    ("1_uniform", [("uniform", 0.0, 1.0)]),
    ("2_gaussian_beta", [("gaussian", 0.0, 1.0), ("beta", 2.0, 5.0)]),
    ("2_gaussian_gamma", [("gaussian", 0.0, 1.0), ("gamma", 2.0, 1.0)]),
    ("2_beta_gamma", [("beta", 2.0, 5.0), ("gamma", 2.0, 1.0)]),
    ("2_uniform_gaussian", [("uniform", 0.0, 1.0), ("gaussian", 1.0, 0.5)]),
    (
        "3_gaussian_beta_gamma",
        [("gaussian", 0.0, 1.0), ("beta", 2.0, 5.0), ("gamma", 2.0, 1.0)],
    ),
    (
        "3_uniform_beta_gaussian",
        [("uniform", 0.0, 1.0), ("beta", 3.0, 2.0), ("gaussian", 0.5, 1.5)],
    ),
    (
        "4_all_types",
        [
            ("gaussian", 0.0, 1.0),
            ("beta", 2.0, 5.0),
            ("gamma", 2.0, 1.0),
            ("uniform", 0.0, 1.0),
        ],
    ),
]


class TestIndependentJoint(Generic[Array], unittest.TestCase):
    """Tests for IndependentJoint."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        # Create marginals: standard normal, beta, uniform
        self.marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), self._bkd),
            ScipyContinuousMarginal(stats.beta(2, 5), self._bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd),
        ]
        self.joint = IndependentJoint(self.marginals, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.joint.nvars(), 3)

    def test_marginals(self) -> None:
        """Test marginals returns list of marginals."""
        marginals = self.joint.marginals()
        self.assertEqual(len(marginals), 3)

    def test_marginal(self) -> None:
        """Test accessing individual marginal."""
        marginal = self.joint.marginal(1)
        self.assertEqual(marginal.name, "beta")

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.joint.rvs(100)
        self.assertEqual(samples.shape, (3, 100))

    def test_logpdf_sum_of_marginals(self) -> None:
        """Test logpdf equals sum of marginal logpdfs."""
        samples = self._bkd.asarray(
            [[0.0, 0.5, -1.0], [0.3, 0.5, 0.2], [0.5, 0.2, 0.8]]
        )

        logpdf_joint = self.joint.logpdf(samples)

        # Compute manually as sum - marginals expect 2D input (1, nsamples)
        logpdf_expected = self._bkd.zeros((1, 3))
        for i, marginal in enumerate(self.marginals):
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            logpdf_expected = logpdf_expected + marginal.logpdf(row_2d)
        logpdf_expected = self._bkd.flatten(logpdf_expected)

        self.assertTrue(self._bkd.allclose(logpdf_joint, logpdf_expected, rtol=1e-6))

    def test_pdf_exp_logpdf(self) -> None:
        """Test pdf = exp(logpdf)."""
        samples = self._bkd.asarray([[0.0, 0.5], [0.3, 0.5], [0.5, 0.2]])
        pdf_vals = self.joint.pdf(samples)
        logpdf_vals = self.joint.logpdf(samples)
        self.assertTrue(
            self._bkd.allclose(pdf_vals, self._bkd.exp(logpdf_vals), rtol=1e-6)
        )

    def test_cdf_product_of_marginals(self) -> None:
        """Test cdf equals product of marginal cdfs."""
        samples = self._bkd.asarray([[0.0, 0.5], [0.3, 0.5], [0.5, 0.2]])

        cdf_joint = self.joint.cdf(samples)

        # Compute manually as product - marginals expect 2D input (1, nsamples)
        cdf_expected = self._bkd.ones((1, 2))
        for i, marginal in enumerate(self.marginals):
            row_2d = self._bkd.reshape(samples[i], (1, -1))
            cdf_expected = cdf_expected * marginal.cdf(row_2d)
        cdf_expected = self._bkd.flatten(cdf_expected)

        self.assertTrue(self._bkd.allclose(cdf_joint, cdf_expected, rtol=1e-6))

    def test_invcdf_component_wise(self) -> None:
        """Test invcdf applies to each component."""
        probs = self._bkd.asarray([[0.5, 0.25], [0.5, 0.75], [0.5, 0.1]])

        quantiles = self.joint.invcdf(probs)

        # Verify each component - marginals expect 2D input (1, nsamples)
        for i, marginal in enumerate(self.marginals):
            row_2d = self._bkd.reshape(probs[i], (1, -1))
            expected = self._bkd.flatten(marginal.invcdf(row_2d))
            self.assertTrue(self._bkd.allclose(quantiles[i], expected, rtol=1e-6))

    def test_correlation_matrix_identity(self) -> None:
        """Test correlation matrix is identity for independent marginals."""
        corr = self.joint.correlation_matrix()
        expected = self._bkd.eye(3)
        self.assertTrue(self._bkd.allclose(corr, expected, rtol=1e-6))

    def test_covariance_diagonal(self) -> None:
        """Test covariance is diagonal for independent marginals."""
        cov = self.joint.covariance()
        # Check diagonal
        diag = self._bkd.get_diagonal(cov)
        self.assertTrue(self._bkd.all_bool(diag > 0))
        # Check off-diagonal is zero
        off_diag = cov - self._bkd.diag(diag)
        expected = self._bkd.zeros((3, 3))
        self.assertTrue(self._bkd.allclose(off_diag, expected, atol=1e-10))

    def test_mean(self) -> None:
        """Test mean computation."""
        mean = self.joint.mean()
        self.assertEqual(mean.shape, (3,))
        # Check first marginal (standard normal)
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([mean[0]]),
                self._bkd.asarray([0.0]),
                atol=0.1,
            )
        )

    def test_variance(self) -> None:
        """Test variance computation."""
        var = self.joint.variance()
        self.assertEqual(var.shape, (3,))
        # Check first marginal (standard normal, var=1)
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([var[0]]),
                self._bkd.asarray([1.0]),
                atol=0.1,
            )
        )

    def test_bounds(self) -> None:
        """Test bounds computation."""
        bounds = self.joint.bounds()
        self.assertEqual(bounds.shape, (2, 3))
        # Uniform has bounds [0, 1]
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([bounds[0, 2]]),
                self._bkd.asarray([0.0]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([bounds[1, 2]]),
                self._bkd.asarray([1.0]),
                atol=1e-10,
            )
        )

    def test_empty_marginals_raises(self) -> None:
        """Test empty marginals raises error."""
        with self.assertRaises(ValueError):
            IndependentJoint([], self._bkd)


class TestIndependentJointNumpy(TestIndependentJoint[NDArray[Any]]):
    """NumPy backend tests for IndependentJoint."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointTorch(TestIndependentJoint[torch.Tensor]):
    """PyTorch backend tests for IndependentJoint."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointGaussian(Generic[Array], unittest.TestCase):
    """Tests for IndependentJoint with Gaussian marginals."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.means = [0.0, 1.0, 2.0]
        self.stds = [1.0, 2.0, 0.5]
        self.marginals = [
            GaussianMarginal(m, s, self._bkd) for m, s in zip(self.means, self.stds)
        ]
        self.joint = IndependentJoint(self.marginals, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.joint.nvars(), 3)

    def test_logpdf_vs_multivariate_normal(self) -> None:
        """Test logpdf matches multivariate normal with diagonal covariance."""
        from scipy.stats import multivariate_normal

        cov = np.diag([s**2 for s in self.stds])
        scipy_dist = multivariate_normal(self.means, cov)

        samples = self._bkd.asarray(
            [[0.0, 0.5, -1.0], [1.0, 0.5, 2.0], [2.0, 2.5, 1.5]]
        )

        logpdf_ours = self.joint.logpdf(samples)
        samples_np = self._bkd.to_numpy(samples)
        logpdf_scipy = self._bkd.asarray(scipy_dist.logpdf(samples_np.T))

        self.assertTrue(self._bkd.allclose(logpdf_ours, logpdf_scipy, rtol=1e-6))

    def test_mean_matches_marginal_means(self) -> None:
        """Test mean matches marginal means."""
        mean = self.joint.mean()
        expected = self._bkd.asarray(self.means)
        self.assertTrue(self._bkd.allclose(mean, expected, rtol=1e-6))

    def test_variance_matches_marginal_variances(self) -> None:
        """Test variance matches marginal variances."""
        var = self.joint.variance()
        expected = self._bkd.asarray([s**2 for s in self.stds])
        self.assertTrue(self._bkd.allclose(var, expected, rtol=1e-6))


class TestIndependentJointGaussianNumpy(TestIndependentJointGaussian[NDArray[Any]]):
    """NumPy backend tests for IndependentJoint with Gaussian marginals."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointGaussianTorch(TestIndependentJointGaussian[torch.Tensor]):
    """PyTorch backend tests for IndependentJoint with Gaussian marginals."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointSingleVariable(Generic[Array], unittest.TestCase):
    """Tests for IndependentJoint with single variable."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.marginal = ScipyContinuousMarginal(stats.norm(0, 1), self._bkd)
        self.joint = IndependentJoint([self.marginal], self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        self.assertEqual(self.joint.nvars(), 1)

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self.joint.rvs(50)
        self.assertEqual(samples.shape, (1, 50))

    def test_logpdf_matches_marginal(self) -> None:
        """Test logpdf matches marginal logpdf."""
        samples = self._bkd.asarray([[0.0, 1.0, -1.0]])

        logpdf_joint = self.joint.logpdf(samples)
        # Marginal expects 2D input (1, nsamples), joint samples is already (1,
        # nsamples)
        logpdf_marginal = self._bkd.flatten(self.marginal.logpdf(samples))

        self.assertTrue(self._bkd.allclose(logpdf_joint, logpdf_marginal, rtol=1e-6))


class TestIndependentJointSingleVariableNumpy(
    TestIndependentJointSingleVariable[NDArray[Any]]
):
    """NumPy backend tests for IndependentJoint with single variable."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointSingleVariableTorch(
    TestIndependentJointSingleVariable[torch.Tensor]
):
    """PyTorch backend tests for IndependentJoint with single variable."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointProtocol(Generic[Array], unittest.TestCase):
    """Tests for JointDistributionProtocol compliance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), self._bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd),
        ]
        self.joint = IndependentJoint(self.marginals, self._bkd)

    def test_has_bkd(self) -> None:
        """Test has bkd method."""
        self.assertIsNotNone(self.joint.bkd())

    def test_has_nvars(self) -> None:
        """Test has nvars method."""
        self.assertEqual(self.joint.nvars(), 2)

    def test_has_rvs(self) -> None:
        """Test has rvs method."""
        samples = self.joint.rvs(10)
        self.assertEqual(samples.shape, (2, 10))

    def test_has_logpdf(self) -> None:
        """Test has logpdf method."""
        samples = self._bkd.asarray([[0.0, 0.5], [0.5, 0.8]])
        logpdf = self.joint.logpdf(samples)
        # Joint logpdf returns (1, nsamples) = (1, 2)
        self.assertEqual(logpdf.shape, (1, 2))

    def test_has_marginals(self) -> None:
        """Test has marginals method."""
        marginals = self.joint.marginals()
        self.assertEqual(len(marginals), 2)

    def test_has_correlation_matrix(self) -> None:
        """Test has correlation_matrix method."""
        corr = self.joint.correlation_matrix()
        self.assertEqual(corr.shape, (2, 2))


class TestIndependentJointProtocolNumpy(TestIndependentJointProtocol[NDArray[Any]]):
    """NumPy backend tests for JointDistributionProtocol compliance."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointProtocolTorch(TestIndependentJointProtocol[torch.Tensor]):
    """PyTorch backend tests for JointDistributionProtocol compliance."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointFunctionProtocol(Generic[Array], unittest.TestCase):
    """Tests for FunctionProtocol methods on IndependentJoint."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        # Bounded marginals for domain tests
        self.bounded_marginals = [
            ScipyContinuousMarginal(stats.beta(2, 5), self._bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd),
        ]
        self.bounded_joint = IndependentJoint(self.bounded_marginals, self._bkd)
        # Unbounded marginals
        self.unbounded_marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), self._bkd),
            ScipyContinuousMarginal(stats.norm(1, 2), self._bkd),
        ]
        self.unbounded_joint = IndependentJoint(self.unbounded_marginals, self._bkd)

    def test_nqoi_returns_one(self) -> None:
        """Test nqoi returns 1 for joint PDF."""
        self.assertEqual(self.bounded_joint.nqoi(), 1)
        self.assertEqual(self.unbounded_joint.nqoi(), 1)

    def test_call_same_as_pdf(self) -> None:
        """Test __call__ returns same as pdf."""
        samples = self._bkd.asarray([[0.3, 0.5], [0.5, 0.2]])
        pdf_vals = self.bounded_joint.pdf(samples)
        call_vals = self.bounded_joint(samples)
        self._bkd.assert_allclose(pdf_vals, call_vals)

    def test_domain_shape_bounded(self) -> None:
        """Test domain shape is (nvars, 2) for bounded distributions."""
        domain = self.bounded_joint.domain()
        self.assertEqual(domain.shape, (2, 2))

    def test_domain_shape_unbounded(self) -> None:
        """Test domain shape is (nvars, 2) for unbounded distributions."""
        domain = self.unbounded_joint.domain()
        self.assertEqual(domain.shape, (2, 2))

    def test_domain_values_bounded(self) -> None:
        """Test domain returns correct bounds for bounded marginals."""
        domain = self.bounded_joint.domain()
        # Beta has support [0, 1], uniform(0,1) has support [0, 1]
        self._bkd.assert_allclose(
            domain[0, :], self._bkd.asarray([0.0, 1.0]), atol=1e-10
        )
        self._bkd.assert_allclose(
            domain[1, :], self._bkd.asarray([0.0, 1.0]), atol=1e-10
        )

    def test_domain_values_unbounded(self) -> None:
        """Test domain returns [-inf, inf] for unbounded marginals."""
        domain = self.unbounded_joint.domain()
        domain_np = self._bkd.to_numpy(domain)
        # Both should be unbounded
        self.assertEqual(domain_np[0, 0], -np.inf)
        self.assertEqual(domain_np[0, 1], np.inf)
        self.assertEqual(domain_np[1, 0], -np.inf)
        self.assertEqual(domain_np[1, 1], np.inf)


class TestIndependentJointFunctionProtocolNumpy(
    TestIndependentJointFunctionProtocol[NDArray[Any]]
):
    """NumPy backend tests for FunctionProtocol methods."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointFunctionProtocolTorch(
    TestIndependentJointFunctionProtocol[torch.Tensor]
):
    """PyTorch backend tests for FunctionProtocol methods."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointPlotter(Generic[Array], unittest.TestCase):
    """Tests for plotter() method on IndependentJoint."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_plotter_1d_bounded(self) -> None:
        """Test plotter returns Plotter1D for 1D bounded distribution."""
        from pyapprox.interface.functions.plot.plot1d import Plotter1D

        marginals = [ScipyContinuousMarginal(stats.beta(2, 5), self._bkd)]
        joint = IndependentJoint(marginals, self._bkd)
        plotter = joint.plotter()
        self.assertIsInstance(plotter, Plotter1D)

    def test_plotter_2d_bounded(self) -> None:
        """Test plotter returns Plotter2DRectangularDomain for 2D bounded."""
        from pyapprox.interface.functions.plot.plot2d_rectangular import (
            Plotter2DRectangularDomain,
        )

        marginals = [
            ScipyContinuousMarginal(stats.beta(2, 5), self._bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd),
        ]
        joint = IndependentJoint(marginals, self._bkd)
        plotter = joint.plotter()
        self.assertIsInstance(plotter, Plotter2DRectangularDomain)

    def test_plotter_3d_unbounded_raises(self) -> None:
        """Test plotter raises ValueError for unbounded >2D without limits."""
        marginals = [
            ScipyContinuousMarginal(stats.beta(2, 5), self._bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd),
            ScipyContinuousMarginal(stats.norm(0, 1), self._bkd),
        ]
        joint = IndependentJoint(marginals, self._bkd)
        with self.assertRaises(ValueError):
            joint.plotter()

    def test_plotter_unbounded_requires_limits(self) -> None:
        """Test plotter raises ValueError for unbounded without limits."""
        marginals = [ScipyContinuousMarginal(stats.norm(0, 1), self._bkd)]
        joint = IndependentJoint(marginals, self._bkd)
        with self.assertRaises(ValueError):
            joint.plotter()

    def test_plotter_unbounded_with_limits(self) -> None:
        """Test plotter works for unbounded with plot_limits."""
        from pyapprox.interface.functions.plot.plot1d import Plotter1D

        marginals = [ScipyContinuousMarginal(stats.norm(0, 1), self._bkd)]
        joint = IndependentJoint(marginals, self._bkd)
        plot_limits = self._bkd.asarray([-3.0, 3.0])
        plotter = joint.plotter(plot_limits)
        self.assertIsInstance(plotter, Plotter1D)


class TestIndependentJointPlotterNumpy(TestIndependentJointPlotter[NDArray[Any]]):
    """NumPy backend tests for plotter method."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointPlotterTorch(TestIndependentJointPlotter[torch.Tensor]):
    """PyTorch backend tests for plotter method."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointDynamicBinding(Generic[Array], unittest.TestCase):
    """Tests for dynamic Jacobian method binding on IndependentJoint."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        # GaussianMarginal has logpdf_jacobian and pdf_jacobian
        self.gaussian_marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd),
            GaussianMarginal(1.0, 2.0, self._bkd),
        ]
        self.gaussian_joint = IndependentJoint(self.gaussian_marginals, self._bkd)
        # ScipyContinuousMarginal does NOT have jacobian methods
        self.scipy_marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), self._bkd),
            ScipyContinuousMarginal(stats.beta(2, 5), self._bkd),
        ]
        self.scipy_joint = IndependentJoint(self.scipy_marginals, self._bkd)

    def test_gaussian_has_logpdf_jacobian(self) -> None:
        """Test GaussianMarginal joint has logpdf_jacobian."""
        self.assertTrue(hasattr(self.gaussian_joint, "logpdf_jacobian"))

    def test_gaussian_has_logpdf_jacobian_batch(self) -> None:
        """Test GaussianMarginal joint has logpdf_jacobian_batch."""
        self.assertTrue(hasattr(self.gaussian_joint, "logpdf_jacobian_batch"))

    def test_gaussian_has_jacobian(self) -> None:
        """Test GaussianMarginal joint has jacobian."""
        self.assertTrue(hasattr(self.gaussian_joint, "jacobian"))

    def test_gaussian_has_jacobian_batch(self) -> None:
        """Test GaussianMarginal joint has jacobian_batch."""
        self.assertTrue(hasattr(self.gaussian_joint, "jacobian_batch"))

    def test_scipy_no_logpdf_jacobian(self) -> None:
        """Test ScipyContinuousMarginal joint does NOT have logpdf_jacobian."""
        self.assertFalse(hasattr(self.scipy_joint, "logpdf_jacobian"))

    def test_scipy_no_jacobian(self) -> None:
        """Test ScipyContinuousMarginal joint does NOT have jacobian."""
        self.assertFalse(hasattr(self.scipy_joint, "jacobian"))


class TestIndependentJointDynamicBindingNumpy(
    TestIndependentJointDynamicBinding[NDArray[Any]]
):
    """NumPy backend tests for dynamic binding."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointDynamicBindingTorch(
    TestIndependentJointDynamicBinding[torch.Tensor]
):
    """PyTorch backend tests for dynamic binding."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointLogpdfJacobian(Generic[Array], unittest.TestCase):
    """Tests for logpdf Jacobian methods with DerivativeChecker validation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd),
            GaussianMarginal(1.0, 2.0, self._bkd),
        ]
        self.joint = IndependentJoint(self.marginals, self._bkd)

    def test_logpdf_jacobian_shape(self) -> None:
        """Test logpdf_jacobian returns shape (1, nvars)."""
        sample = self._bkd.asarray([[0.5], [1.5]])  # (nvars=2, 1)
        jac = self.joint.logpdf_jacobian(sample)
        self.assertEqual(jac.shape, (1, 2))

    def test_logpdf_jacobian_batch_shape(self) -> None:
        """Test logpdf_jacobian_batch returns shape (nsamples, 1, nvars)."""
        samples = self._bkd.asarray([[0.0, 0.5, 1.0], [1.0, 1.5, 2.0]])
        jac = self.joint.logpdf_jacobian_batch(samples)
        self.assertEqual(jac.shape, (3, 1, 2))

    def test_logpdf_jacobian_vs_numerical(self) -> None:
        """Test logpdf_jacobian against numerical derivatives using
        DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        # Create a wrapper that has jacobian method for logpdf
        class LogpdfWrapper:
            def __init__(self, joint: IndependentJoint[Array], bkd: Backend[Array]):
                self._joint = joint
                self._bkd = bkd

            def bkd(self) -> Backend[Array]:
                return self._bkd

            def nvars(self) -> int:
                return self._joint.nvars()

            def nqoi(self) -> int:
                return 1

            def __call__(self, samples: Array) -> Array:
                return self._joint.logpdf(samples)

            def jacobian(self, sample: Array) -> Array:
                return self._joint.logpdf_jacobian(sample)

        wrapper = LogpdfWrapper(self.joint, self._bkd)
        checker = DerivativeChecker(wrapper)  # type: ignore[arg-type]
        sample = self._bkd.asarray([[0.3], [1.2]])
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = float(self._bkd.to_numpy(checker.error_ratio(errors[0])))
        # For correct jacobian, ratio should be <= 2e-6
        self.assertLessEqual(ratio, 2e-6)

    def test_logpdf_jacobian_batch_consistency(self) -> None:
        """Test logpdf_jacobian_batch is consistent with single sample version."""
        samples = self._bkd.asarray([[0.0, 0.5], [1.0, 1.5]])
        batch_jac = self.joint.logpdf_jacobian_batch(samples)

        for ii in range(2):
            single = samples[:, ii : ii + 1]
            single_jac = self.joint.logpdf_jacobian(single)
            self._bkd.assert_allclose(batch_jac[ii, 0, :], single_jac[0, :])


class TestIndependentJointLogpdfJacobianNumpy(
    TestIndependentJointLogpdfJacobian[NDArray[Any]]
):
    """NumPy backend tests for logpdf Jacobian."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointLogpdfJacobianTorch(
    TestIndependentJointLogpdfJacobian[torch.Tensor]
):
    """PyTorch backend tests for logpdf Jacobian."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointPdfJacobian(Generic[Array], unittest.TestCase):
    """Tests for PDF Jacobian methods with DerivativeChecker validation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd),
            GaussianMarginal(1.0, 2.0, self._bkd),
        ]
        self.joint = IndependentJoint(self.marginals, self._bkd)

    def test_jacobian_shape(self) -> None:
        """Test jacobian returns shape (1, nvars)."""
        sample = self._bkd.asarray([[0.5], [1.5]])  # (nvars=2, 1)
        jac = self.joint.jacobian(sample)
        self.assertEqual(jac.shape, (1, 2))

    def test_jacobian_batch_shape(self) -> None:
        """Test jacobian_batch returns shape (nsamples, 1, nvars)."""
        samples = self._bkd.asarray([[0.0, 0.5, 1.0], [1.0, 1.5, 2.0]])
        jac = self.joint.jacobian_batch(samples)
        self.assertEqual(jac.shape, (3, 1, 2))

    def test_jacobian_vs_numerical(self) -> None:
        """Test jacobian against numerical derivatives using DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        # The joint already has __call__ (pdf) and jacobian methods
        # Check that it satisfies the protocol
        checker = DerivativeChecker(self.joint)  # type: ignore[arg-type]
        sample = self._bkd.asarray([[0.3], [1.2]])
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = float(self._bkd.to_numpy(checker.error_ratio(errors[0])))
        # For correct jacobian, ratio should be <= 2e-6
        self.assertLessEqual(ratio, 2e-6)

    def test_jacobian_batch_consistency(self) -> None:
        """Test jacobian_batch is consistent with single sample version."""
        samples = self._bkd.asarray([[0.0, 0.5], [1.0, 1.5]])
        batch_jac = self.joint.jacobian_batch(samples)

        for ii in range(2):
            single = samples[:, ii : ii + 1]
            single_jac = self.joint.jacobian(single)
            self._bkd.assert_allclose(batch_jac[ii, 0, :], single_jac[0, :])

    def test_jacobian_product_rule(self) -> None:
        """Test jacobian follows product rule: d/dx_i[prod_j p_j] = p'_i * prod_{j!=i}
        p_j."""
        sample = self._bkd.asarray([[0.3], [1.2]])
        jac = self.joint.jacobian(sample)

        # Compute expected using product rule
        pdf_vals = []
        pdf_jacs = []
        for i, marginal in enumerate(self.marginals):
            row_2d = self._bkd.reshape(sample[i], (1, -1))
            pdf_vals.append(marginal(row_2d)[0, 0])
            pdf_jacs.append(marginal.pdf_jacobian(row_2d)[0, 0])

        # d/dx_0[p_0 * p_1] = p'_0 * p_1
        expected_0 = pdf_jacs[0] * pdf_vals[1]
        # d/dx_1[p_0 * p_1] = p_0 * p'_1
        expected_1 = pdf_vals[0] * pdf_jacs[1]

        self._bkd.assert_allclose(
            jac[0, :], self._bkd.asarray([expected_0, expected_1])
        )


class TestIndependentJointPdfJacobianNumpy(
    TestIndependentJointPdfJacobian[NDArray[Any]]
):
    """NumPy backend tests for PDF Jacobian."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointPdfJacobianTorch(
    TestIndependentJointPdfJacobian[torch.Tensor]
):
    """PyTorch backend tests for PDF Jacobian."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentJointJacobianCombinations(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized tests for Jacobians with different marginal combinations."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_marginal(self, spec: tuple) -> Any:
        """Create a marginal from a specification tuple."""
        mtype = spec[0]
        if mtype == "gaussian":
            return GaussianMarginal(spec[1], spec[2], self._bkd)
        elif mtype == "beta":
            return BetaMarginal(spec[1], spec[2], self._bkd)
        elif mtype == "gamma":
            return GammaMarginal(spec[1], spec[2], self._bkd)
        elif mtype == "uniform":
            return UniformMarginal(spec[1], spec[2], self._bkd)
        else:
            raise ValueError(f"Unknown marginal type: {mtype}")

    def _create_joint(self, marginal_specs: list) -> IndependentJoint[Array]:
        """Create a joint distribution from marginal specifications."""
        marginals = [self._create_marginal(spec) for spec in marginal_specs]
        return IndependentJoint(marginals, self._bkd)

    def _create_sample(self, joint: IndependentJoint[Array]) -> Array:
        """Create a sample in the interior of the support using rvs."""
        np.random.seed(42)
        samples = joint.rvs(1)  # Shape: (nvars, 1)
        return samples

    def _create_samples(self, joint: IndependentJoint[Array], nsamples: int) -> Array:
        """Create samples in the interior of the support using rvs."""
        np.random.seed(42)
        return joint.rvs(nsamples)  # Shape: (nvars, nsamples)

    @parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_jacobian_shape(self, name: str, marginal_specs: list) -> None:
        """Test jacobian shape for different marginal combinations."""
        joint = self._create_joint(marginal_specs)
        sample = self._create_sample(joint)
        jac = joint.jacobian(sample)
        self.assertEqual(jac.shape, (1, joint.nvars()))

    @parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_jacobian_batch_shape(self, name: str, marginal_specs: list) -> None:
        """Test jacobian_batch shape for different marginal combinations."""
        nsamples = 5
        joint = self._create_joint(marginal_specs)
        samples = self._create_samples(joint, nsamples)
        jac = joint.jacobian_batch(samples)
        self.assertEqual(jac.shape, (nsamples, 1, joint.nvars()))

    @parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_logpdf_jacobian_shape(self, name: str, marginal_specs: list) -> None:
        """Test logpdf_jacobian shape for different marginal combinations."""
        joint = self._create_joint(marginal_specs)
        sample = self._create_sample(joint)
        jac = joint.logpdf_jacobian(sample)
        self.assertEqual(jac.shape, (1, joint.nvars()))

    @parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_logpdf_jacobian_batch_shape(self, name: str, marginal_specs: list) -> None:
        """Test logpdf_jacobian_batch shape for different marginal combinations."""
        nsamples = 5
        joint = self._create_joint(marginal_specs)
        samples = self._create_samples(joint, nsamples)
        jac = joint.logpdf_jacobian_batch(samples)
        self.assertEqual(jac.shape, (nsamples, 1, joint.nvars()))

    @parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_jacobian_numerical(self, name: str, marginal_specs: list) -> None:
        """Test jacobian against numerical derivatives."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        # Skip single uniform - has zero Jacobian everywhere, numerical check
        # returns NaN
        if name == "1_uniform":
            self.skipTest("Uniform has zero Jacobian, numerical check not applicable")

        joint = self._create_joint(marginal_specs)
        checker = DerivativeChecker(joint)  # type: ignore[arg-type]
        sample = self._create_sample(joint)
        # Use smaller fd_eps to avoid stepping outside valid domain
        fd_eps = self._bkd.asarray([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
        errors = checker.check_derivatives(sample, fd_eps=fd_eps, verbosity=0)
        ratio = float(self._bkd.to_numpy(checker.error_ratio(errors[0])))
        # Use 1e-5 tolerance for numerical derivatives (some loss expected)
        self.assertLessEqual(ratio, 1e-5)

    @parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_jacobian_batch_consistency(self, name: str, marginal_specs: list) -> None:
        """Test jacobian_batch matches single jacobian."""
        joint = self._create_joint(marginal_specs)
        samples = self._create_samples(joint, 3)
        batch_jac = joint.jacobian_batch(samples)
        for ii in range(3):
            single = samples[:, ii : ii + 1]
            single_jac = joint.jacobian(single)
            self._bkd.assert_allclose(batch_jac[ii, 0, :], single_jac[0, :])


class TestIndependentJointJacobianCombinationsNumpy(
    TestIndependentJointJacobianCombinations[NDArray[Any]]
):
    """NumPy backend tests for Jacobian combinations."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentJointJacobianCombinationsTorch(
    TestIndependentJointJacobianCombinations[torch.Tensor]
):
    """PyTorch backend tests for Jacobian combinations."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
