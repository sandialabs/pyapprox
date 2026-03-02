"""
Tests for joint probability distributions.
"""

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability.joint import IndependentJoint
from pyapprox.probability.univariate import (
    BetaMarginal,
    GammaMarginal,
    GaussianMarginal,
    ScipyContinuousMarginal,
    UniformMarginal,
)

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


class TestIndependentJoint:
    """Tests for IndependentJoint."""

    def _setup(self, bkd):
        # Create marginals: standard normal, beta, uniform
        marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), bkd),
            ScipyContinuousMarginal(stats.beta(2, 5), bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), bkd),
        ]
        joint = IndependentJoint(marginals, bkd)
        return marginals, joint

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct dimension."""
        _, joint = self._setup(bkd)
        assert joint.nvars() == 3

    def test_marginals(self, bkd) -> None:
        """Test marginals returns list of marginals."""
        _, joint = self._setup(bkd)
        marginals = joint.marginals()
        assert len(marginals) == 3

    def test_marginal(self, bkd) -> None:
        """Test accessing individual marginal."""
        _, joint = self._setup(bkd)
        marginal = joint.marginal(1)
        assert marginal.name == "beta"

    def test_rvs_shape(self, bkd) -> None:
        """Test rvs returns correct shape."""
        _, joint = self._setup(bkd)
        samples = joint.rvs(100)
        assert samples.shape == (3, 100)

    def test_logpdf_sum_of_marginals(self, bkd) -> None:
        """Test logpdf equals sum of marginal logpdfs."""
        marginals, joint = self._setup(bkd)
        samples = bkd.asarray([[0.0, 0.5, -1.0], [0.3, 0.5, 0.2], [0.5, 0.2, 0.8]])

        logpdf_joint = joint.logpdf(samples)

        # Compute manually as sum - marginals expect 2D input (1, nsamples)
        logpdf_expected = bkd.zeros((1, 3))
        for i, marginal in enumerate(marginals):
            row_2d = bkd.reshape(samples[i], (1, -1))
            logpdf_expected = logpdf_expected + marginal.logpdf(row_2d)
        logpdf_expected = bkd.flatten(logpdf_expected)

        assert bkd.allclose(logpdf_joint, logpdf_expected, rtol=1e-6)

    def test_pdf_exp_logpdf(self, bkd) -> None:
        """Test pdf = exp(logpdf)."""
        _, joint = self._setup(bkd)
        samples = bkd.asarray([[0.0, 0.5], [0.3, 0.5], [0.5, 0.2]])
        pdf_vals = joint.pdf(samples)
        logpdf_vals = joint.logpdf(samples)
        assert bkd.allclose(pdf_vals, bkd.exp(logpdf_vals), rtol=1e-6)

    def test_cdf_product_of_marginals(self, bkd) -> None:
        """Test cdf equals product of marginal cdfs."""
        marginals, joint = self._setup(bkd)
        samples = bkd.asarray([[0.0, 0.5], [0.3, 0.5], [0.5, 0.2]])

        cdf_joint = joint.cdf(samples)

        # Compute manually as product - marginals expect 2D input (1, nsamples)
        cdf_expected = bkd.ones((1, 2))
        for i, marginal in enumerate(marginals):
            row_2d = bkd.reshape(samples[i], (1, -1))
            cdf_expected = cdf_expected * marginal.cdf(row_2d)
        cdf_expected = bkd.flatten(cdf_expected)

        assert bkd.allclose(cdf_joint, cdf_expected, rtol=1e-6)

    def test_invcdf_component_wise(self, bkd) -> None:
        """Test invcdf applies to each component."""
        marginals, joint = self._setup(bkd)
        probs = bkd.asarray([[0.5, 0.25], [0.5, 0.75], [0.5, 0.1]])

        quantiles = joint.invcdf(probs)

        # Verify each component - marginals expect 2D input (1, nsamples)
        for i, marginal in enumerate(marginals):
            row_2d = bkd.reshape(probs[i], (1, -1))
            expected = bkd.flatten(marginal.invcdf(row_2d))
            assert bkd.allclose(quantiles[i], expected, rtol=1e-6)

    def test_correlation_matrix_identity(self, bkd) -> None:
        """Test correlation matrix is identity for independent marginals."""
        _, joint = self._setup(bkd)
        corr = joint.correlation_matrix()
        expected = bkd.eye(3)
        assert bkd.allclose(corr, expected, rtol=1e-6)

    def test_covariance_diagonal(self, bkd) -> None:
        """Test covariance is diagonal for independent marginals."""
        _, joint = self._setup(bkd)
        cov = joint.covariance()
        # Check diagonal
        diag = bkd.get_diagonal(cov)
        assert bkd.all_bool(diag > 0)
        # Check off-diagonal is zero
        off_diag = cov - bkd.diag(diag)
        expected = bkd.zeros((3, 3))
        assert bkd.allclose(off_diag, expected, atol=1e-10)

    def test_mean(self, bkd) -> None:
        """Test mean computation."""
        _, joint = self._setup(bkd)
        mean = joint.mean()
        assert mean.shape == (3,)
        # Check first marginal (standard normal)
        assert bkd.allclose(
            bkd.asarray([mean[0]]),
            bkd.asarray([0.0]),
            atol=0.1,
        )

    def test_variance(self, bkd) -> None:
        """Test variance computation."""
        _, joint = self._setup(bkd)
        var = joint.variance()
        assert var.shape == (3,)
        # Check first marginal (standard normal, var=1)
        assert bkd.allclose(
            bkd.asarray([var[0]]),
            bkd.asarray([1.0]),
            atol=0.1,
        )

    def test_bounds(self, bkd) -> None:
        """Test bounds computation."""
        _, joint = self._setup(bkd)
        bounds = joint.bounds()
        assert bounds.shape == (2, 3)
        # Uniform has bounds [0, 1]
        assert bkd.allclose(
            bkd.asarray([bounds[0, 2]]),
            bkd.asarray([0.0]),
            atol=1e-10,
        )
        assert bkd.allclose(
            bkd.asarray([bounds[1, 2]]),
            bkd.asarray([1.0]),
            atol=1e-10,
        )

    def test_empty_marginals_raises(self, bkd) -> None:
        """Test empty marginals raises error."""
        with pytest.raises(ValueError):
            IndependentJoint([], bkd)


class TestIndependentJointGaussian:
    """Tests for IndependentJoint with Gaussian marginals."""

    def _setup(self, bkd):
        means = [0.0, 1.0, 2.0]
        stds = [1.0, 2.0, 0.5]
        marginals = [GaussianMarginal(m, s, bkd) for m, s in zip(means, stds)]
        joint = IndependentJoint(marginals, bkd)
        return means, stds, marginals, joint

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct dimension."""
        _, _, _, joint = self._setup(bkd)
        assert joint.nvars() == 3

    def test_logpdf_vs_multivariate_normal(self, bkd) -> None:
        """Test logpdf matches multivariate normal with diagonal covariance."""
        from scipy.stats import multivariate_normal

        means, stds, _, joint = self._setup(bkd)
        cov = np.diag([s**2 for s in stds])
        scipy_dist = multivariate_normal(means, cov)

        samples = bkd.asarray([[0.0, 0.5, -1.0], [1.0, 0.5, 2.0], [2.0, 2.5, 1.5]])

        logpdf_ours = joint.logpdf(samples)
        samples_np = bkd.to_numpy(samples)
        logpdf_scipy = bkd.asarray(scipy_dist.logpdf(samples_np.T))

        assert bkd.allclose(logpdf_ours, logpdf_scipy, rtol=1e-6)

    def test_mean_matches_marginal_means(self, bkd) -> None:
        """Test mean matches marginal means."""
        means, _, _, joint = self._setup(bkd)
        mean = joint.mean()
        expected = bkd.asarray(means)
        assert bkd.allclose(mean, expected, rtol=1e-6)

    def test_variance_matches_marginal_variances(self, bkd) -> None:
        """Test variance matches marginal variances."""
        _, stds, _, joint = self._setup(bkd)
        var = joint.variance()
        expected = bkd.asarray([s**2 for s in stds])
        assert bkd.allclose(var, expected, rtol=1e-6)


class TestIndependentJointSingleVariable:
    """Tests for IndependentJoint with single variable."""

    def _setup(self, bkd):
        marginal = ScipyContinuousMarginal(stats.norm(0, 1), bkd)
        joint = IndependentJoint([marginal], bkd)
        return marginal, joint

    def test_nvars(self, bkd) -> None:
        """Test nvars returns 1."""
        _, joint = self._setup(bkd)
        assert joint.nvars() == 1

    def test_rvs_shape(self, bkd) -> None:
        """Test rvs returns correct shape."""
        _, joint = self._setup(bkd)
        samples = joint.rvs(50)
        assert samples.shape == (1, 50)

    def test_logpdf_matches_marginal(self, bkd) -> None:
        """Test logpdf matches marginal logpdf."""
        marginal, joint = self._setup(bkd)
        samples = bkd.asarray([[0.0, 1.0, -1.0]])

        logpdf_joint = joint.logpdf(samples)
        # Marginal expects 2D input (1, nsamples), joint samples is already (1,
        # nsamples)
        logpdf_marginal = bkd.flatten(marginal.logpdf(samples))

        assert bkd.allclose(logpdf_joint, logpdf_marginal, rtol=1e-6)


class TestIndependentJointProtocol:
    """Tests for JointDistributionProtocol compliance."""

    def _setup(self, bkd):
        marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), bkd),
        ]
        joint = IndependentJoint(marginals, bkd)
        return marginals, joint

    def test_has_bkd(self, bkd) -> None:
        """Test has bkd method."""
        _, joint = self._setup(bkd)
        assert joint.bkd() is not None

    def test_has_nvars(self, bkd) -> None:
        """Test has nvars method."""
        _, joint = self._setup(bkd)
        assert joint.nvars() == 2

    def test_has_rvs(self, bkd) -> None:
        """Test has rvs method."""
        _, joint = self._setup(bkd)
        samples = joint.rvs(10)
        assert samples.shape == (2, 10)

    def test_has_logpdf(self, bkd) -> None:
        """Test has logpdf method."""
        _, joint = self._setup(bkd)
        samples = bkd.asarray([[0.0, 0.5], [0.5, 0.8]])
        logpdf = joint.logpdf(samples)
        # Joint logpdf returns (1, nsamples) = (1, 2)
        assert logpdf.shape == (1, 2)

    def test_has_marginals(self, bkd) -> None:
        """Test has marginals method."""
        _, joint = self._setup(bkd)
        marginals = joint.marginals()
        assert len(marginals) == 2

    def test_has_correlation_matrix(self, bkd) -> None:
        """Test has correlation_matrix method."""
        _, joint = self._setup(bkd)
        corr = joint.correlation_matrix()
        assert corr.shape == (2, 2)


class TestIndependentJointFunctionProtocol:
    """Tests for FunctionProtocol methods on IndependentJoint."""

    def _setup(self, bkd):
        # Bounded marginals for domain tests
        bounded_marginals = [
            ScipyContinuousMarginal(stats.beta(2, 5), bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), bkd),
        ]
        bounded_joint = IndependentJoint(bounded_marginals, bkd)
        # Unbounded marginals
        unbounded_marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), bkd),
            ScipyContinuousMarginal(stats.norm(1, 2), bkd),
        ]
        unbounded_joint = IndependentJoint(unbounded_marginals, bkd)
        return bounded_joint, unbounded_joint

    def test_nqoi_returns_one(self, bkd) -> None:
        """Test nqoi returns 1 for joint PDF."""
        bounded_joint, unbounded_joint = self._setup(bkd)
        assert bounded_joint.nqoi() == 1
        assert unbounded_joint.nqoi() == 1

    def test_call_same_as_pdf(self, bkd) -> None:
        """Test __call__ returns same as pdf."""
        bounded_joint, _ = self._setup(bkd)
        samples = bkd.asarray([[0.3, 0.5], [0.5, 0.2]])
        pdf_vals = bounded_joint.pdf(samples)
        call_vals = bounded_joint(samples)
        bkd.assert_allclose(pdf_vals, call_vals)

    def test_domain_shape_bounded(self, bkd) -> None:
        """Test domain shape is (nvars, 2) for bounded distributions."""
        bounded_joint, _ = self._setup(bkd)
        domain = bounded_joint.domain()
        assert domain.shape == (2, 2)

    def test_domain_shape_unbounded(self, bkd) -> None:
        """Test domain shape is (nvars, 2) for unbounded distributions."""
        _, unbounded_joint = self._setup(bkd)
        domain = unbounded_joint.domain()
        assert domain.shape == (2, 2)

    def test_domain_values_bounded(self, bkd) -> None:
        """Test domain returns correct bounds for bounded marginals."""
        bounded_joint, _ = self._setup(bkd)
        domain = bounded_joint.domain()
        # Beta has support [0, 1], uniform(0,1) has support [0, 1]
        bkd.assert_allclose(domain[0, :], bkd.asarray([0.0, 1.0]), atol=1e-10)
        bkd.assert_allclose(domain[1, :], bkd.asarray([0.0, 1.0]), atol=1e-10)

    def test_domain_values_unbounded(self, bkd) -> None:
        """Test domain returns [-inf, inf] for unbounded marginals."""
        _, unbounded_joint = self._setup(bkd)
        domain = unbounded_joint.domain()
        domain_np = bkd.to_numpy(domain)
        # Both should be unbounded
        assert domain_np[0, 0] == -np.inf
        assert domain_np[0, 1] == np.inf
        assert domain_np[1, 0] == -np.inf
        assert domain_np[1, 1] == np.inf


class TestIndependentJointPlotter:
    """Tests for plotter() method on IndependentJoint."""

    def test_plotter_1d_bounded(self, bkd) -> None:
        """Test plotter returns Plotter1D for 1D bounded distribution."""
        from pyapprox.interface.functions.plot.plot1d import Plotter1D

        marginals = [ScipyContinuousMarginal(stats.beta(2, 5), bkd)]
        joint = IndependentJoint(marginals, bkd)
        plotter = joint.plotter()
        assert isinstance(plotter, Plotter1D)

    def test_plotter_2d_bounded(self, bkd) -> None:
        """Test plotter returns Plotter2DRectangularDomain for 2D bounded."""
        from pyapprox.interface.functions.plot.plot2d_rectangular import (
            Plotter2DRectangularDomain,
        )

        marginals = [
            ScipyContinuousMarginal(stats.beta(2, 5), bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), bkd),
        ]
        joint = IndependentJoint(marginals, bkd)
        plotter = joint.plotter()
        assert isinstance(plotter, Plotter2DRectangularDomain)

    def test_plotter_3d_unbounded_raises(self, bkd) -> None:
        """Test plotter raises ValueError for unbounded >2D without limits."""
        marginals = [
            ScipyContinuousMarginal(stats.beta(2, 5), bkd),
            ScipyContinuousMarginal(stats.uniform(0, 1), bkd),
            ScipyContinuousMarginal(stats.norm(0, 1), bkd),
        ]
        joint = IndependentJoint(marginals, bkd)
        with pytest.raises(ValueError):
            joint.plotter()

    def test_plotter_unbounded_requires_limits(self, bkd) -> None:
        """Test plotter raises ValueError for unbounded without limits."""
        marginals = [ScipyContinuousMarginal(stats.norm(0, 1), bkd)]
        joint = IndependentJoint(marginals, bkd)
        with pytest.raises(ValueError):
            joint.plotter()

    def test_plotter_unbounded_with_limits(self, bkd) -> None:
        """Test plotter works for unbounded with plot_limits."""
        from pyapprox.interface.functions.plot.plot1d import Plotter1D

        marginals = [ScipyContinuousMarginal(stats.norm(0, 1), bkd)]
        joint = IndependentJoint(marginals, bkd)
        plot_limits = bkd.asarray([-3.0, 3.0])
        plotter = joint.plotter(plot_limits)
        assert isinstance(plotter, Plotter1D)


class TestIndependentJointDynamicBinding:
    """Tests for dynamic Jacobian method binding on IndependentJoint."""

    def _setup(self, bkd):
        # GaussianMarginal has logpdf_jacobian and pdf_jacobian
        gaussian_marginals = [
            GaussianMarginal(0.0, 1.0, bkd),
            GaussianMarginal(1.0, 2.0, bkd),
        ]
        gaussian_joint = IndependentJoint(gaussian_marginals, bkd)
        # ScipyContinuousMarginal does NOT have jacobian methods
        scipy_marginals = [
            ScipyContinuousMarginal(stats.norm(0, 1), bkd),
            ScipyContinuousMarginal(stats.beta(2, 5), bkd),
        ]
        scipy_joint = IndependentJoint(scipy_marginals, bkd)
        return gaussian_joint, scipy_joint

    def test_gaussian_has_logpdf_jacobian(self, bkd) -> None:
        """Test GaussianMarginal joint has logpdf_jacobian."""
        gaussian_joint, _ = self._setup(bkd)
        assert hasattr(gaussian_joint, "logpdf_jacobian")

    def test_gaussian_has_logpdf_jacobian_batch(self, bkd) -> None:
        """Test GaussianMarginal joint has logpdf_jacobian_batch."""
        gaussian_joint, _ = self._setup(bkd)
        assert hasattr(gaussian_joint, "logpdf_jacobian_batch")

    def test_gaussian_has_jacobian(self, bkd) -> None:
        """Test GaussianMarginal joint has jacobian."""
        gaussian_joint, _ = self._setup(bkd)
        assert hasattr(gaussian_joint, "jacobian")

    def test_gaussian_has_jacobian_batch(self, bkd) -> None:
        """Test GaussianMarginal joint has jacobian_batch."""
        gaussian_joint, _ = self._setup(bkd)
        assert hasattr(gaussian_joint, "jacobian_batch")

    def test_scipy_no_logpdf_jacobian(self, bkd) -> None:
        """Test ScipyContinuousMarginal joint does NOT have logpdf_jacobian."""
        _, scipy_joint = self._setup(bkd)
        assert not hasattr(scipy_joint, "logpdf_jacobian")

    def test_scipy_no_jacobian(self, bkd) -> None:
        """Test ScipyContinuousMarginal joint does NOT have jacobian."""
        _, scipy_joint = self._setup(bkd)
        assert not hasattr(scipy_joint, "jacobian")


class TestIndependentJointLogpdfJacobian:
    """Tests for logpdf Jacobian methods with DerivativeChecker validation."""

    def _setup(self, bkd):
        marginals = [
            GaussianMarginal(0.0, 1.0, bkd),
            GaussianMarginal(1.0, 2.0, bkd),
        ]
        joint = IndependentJoint(marginals, bkd)
        return marginals, joint

    def test_logpdf_jacobian_shape(self, bkd) -> None:
        """Test logpdf_jacobian returns shape (1, nvars)."""
        _, joint = self._setup(bkd)
        sample = bkd.asarray([[0.5], [1.5]])  # (nvars=2, 1)
        jac = joint.logpdf_jacobian(sample)
        assert jac.shape == (1, 2)

    def test_logpdf_jacobian_batch_shape(self, bkd) -> None:
        """Test logpdf_jacobian_batch returns shape (nsamples, 1, nvars)."""
        _, joint = self._setup(bkd)
        samples = bkd.asarray([[0.0, 0.5, 1.0], [1.0, 1.5, 2.0]])
        jac = joint.logpdf_jacobian_batch(samples)
        assert jac.shape == (3, 1, 2)

    def test_logpdf_jacobian_vs_numerical(self, bkd) -> None:
        """Test logpdf_jacobian against numerical derivatives using
        DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        _, joint = self._setup(bkd)

        # Create a wrapper that has jacobian method for logpdf
        class LogpdfWrapper:
            def __init__(self, joint, bkd):
                self._joint = joint
                self._bkd = bkd

            def bkd(self):
                return self._bkd

            def nvars(self) -> int:
                return self._joint.nvars()

            def nqoi(self) -> int:
                return 1

            def __call__(self, samples):
                return self._joint.logpdf(samples)

            def jacobian(self, sample):
                return self._joint.logpdf_jacobian(sample)

        wrapper = LogpdfWrapper(joint, bkd)
        checker = DerivativeChecker(wrapper)  # type: ignore[arg-type]
        sample = bkd.asarray([[0.3], [1.2]])
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        # For correct jacobian, ratio should be <= 2e-6
        assert ratio <= 2e-6

    def test_logpdf_jacobian_batch_consistency(self, bkd) -> None:
        """Test logpdf_jacobian_batch is consistent with single sample version."""
        _, joint = self._setup(bkd)
        samples = bkd.asarray([[0.0, 0.5], [1.0, 1.5]])
        batch_jac = joint.logpdf_jacobian_batch(samples)

        for ii in range(2):
            single = samples[:, ii : ii + 1]
            single_jac = joint.logpdf_jacobian(single)
            bkd.assert_allclose(batch_jac[ii, 0, :], single_jac[0, :])


class TestIndependentJointPdfJacobian:
    """Tests for PDF Jacobian methods with DerivativeChecker validation."""

    def _setup(self, bkd):
        marginals = [
            GaussianMarginal(0.0, 1.0, bkd),
            GaussianMarginal(1.0, 2.0, bkd),
        ]
        joint = IndependentJoint(marginals, bkd)
        return marginals, joint

    def test_jacobian_shape(self, bkd) -> None:
        """Test jacobian returns shape (1, nvars)."""
        _, joint = self._setup(bkd)
        sample = bkd.asarray([[0.5], [1.5]])  # (nvars=2, 1)
        jac = joint.jacobian(sample)
        assert jac.shape == (1, 2)

    def test_jacobian_batch_shape(self, bkd) -> None:
        """Test jacobian_batch returns shape (nsamples, 1, nvars)."""
        _, joint = self._setup(bkd)
        samples = bkd.asarray([[0.0, 0.5, 1.0], [1.0, 1.5, 2.0]])
        jac = joint.jacobian_batch(samples)
        assert jac.shape == (3, 1, 2)

    def test_jacobian_vs_numerical(self, bkd) -> None:
        """Test jacobian against numerical derivatives using DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        _, joint = self._setup(bkd)

        # The joint already has __call__ (pdf) and jacobian methods
        # Check that it satisfies the protocol
        checker = DerivativeChecker(joint)  # type: ignore[arg-type]
        sample = bkd.asarray([[0.3], [1.2]])
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        # For correct jacobian, ratio should be <= 2e-6
        assert ratio <= 2e-6

    def test_jacobian_batch_consistency(self, bkd) -> None:
        """Test jacobian_batch is consistent with single sample version."""
        _, joint = self._setup(bkd)
        samples = bkd.asarray([[0.0, 0.5], [1.0, 1.5]])
        batch_jac = joint.jacobian_batch(samples)

        for ii in range(2):
            single = samples[:, ii : ii + 1]
            single_jac = joint.jacobian(single)
            bkd.assert_allclose(batch_jac[ii, 0, :], single_jac[0, :])

    def test_jacobian_product_rule(self, bkd) -> None:
        """Test jacobian follows product rule: d/dx_i[prod_j p_j] = p'_i * prod_{j!=i}
        p_j."""
        marginals, joint = self._setup(bkd)
        sample = bkd.asarray([[0.3], [1.2]])
        jac = joint.jacobian(sample)

        # Compute expected using product rule
        pdf_vals = []
        pdf_jacs = []
        for i, marginal in enumerate(marginals):
            row_2d = bkd.reshape(sample[i], (1, -1))
            pdf_vals.append(marginal(row_2d)[0, 0])
            pdf_jacs.append(marginal.pdf_jacobian(row_2d)[0, 0])

        # d/dx_0[p_0 * p_1] = p'_0 * p_1
        expected_0 = pdf_jacs[0] * pdf_vals[1]
        # d/dx_1[p_0 * p_1] = p_0 * p'_1
        expected_1 = pdf_vals[0] * pdf_jacs[1]

        bkd.assert_allclose(jac[0, :], bkd.asarray([expected_0, expected_1]))


class TestIndependentJointJacobianCombinations:
    """Parametrized tests for Jacobians with different marginal combinations."""

    def _create_marginal(self, spec, bkd):
        """Create a marginal from a specification tuple."""
        mtype = spec[0]
        if mtype == "gaussian":
            return GaussianMarginal(spec[1], spec[2], bkd)
        elif mtype == "beta":
            return BetaMarginal(spec[1], spec[2], bkd)
        elif mtype == "gamma":
            return GammaMarginal(spec[1], spec[2], bkd)
        elif mtype == "uniform":
            return UniformMarginal(spec[1], spec[2], bkd)
        else:
            raise ValueError(f"Unknown marginal type: {mtype}")

    def _create_joint(self, marginal_specs, bkd):
        """Create a joint distribution from marginal specifications."""
        marginals = [self._create_marginal(spec, bkd) for spec in marginal_specs]
        return IndependentJoint(marginals, bkd)

    def _create_sample(self, joint, bkd):
        """Create a sample in the interior of the support using rvs."""
        np.random.seed(42)
        samples = joint.rvs(1)  # Shape: (nvars, 1)
        return samples

    def _create_samples(self, joint, bkd, nsamples: int):
        """Create samples in the interior of the support using rvs."""
        np.random.seed(42)
        return joint.rvs(nsamples)  # Shape: (nvars, nsamples)

    @pytest.mark.parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_jacobian_shape(self, bkd, name: str, marginal_specs: list) -> None:
        """Test jacobian shape for different marginal combinations."""
        joint = self._create_joint(marginal_specs, bkd)
        sample = self._create_sample(joint, bkd)
        jac = joint.jacobian(sample)
        assert jac.shape == (1, joint.nvars())

    @pytest.mark.parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_jacobian_batch_shape(self, bkd, name: str, marginal_specs: list) -> None:
        """Test jacobian_batch shape for different marginal combinations."""
        nsamples = 5
        joint = self._create_joint(marginal_specs, bkd)
        samples = self._create_samples(joint, bkd, nsamples)
        jac = joint.jacobian_batch(samples)
        assert jac.shape == (nsamples, 1, joint.nvars())

    @pytest.mark.parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_logpdf_jacobian_shape(self, bkd, name: str, marginal_specs: list) -> None:
        """Test logpdf_jacobian shape for different marginal combinations."""
        joint = self._create_joint(marginal_specs, bkd)
        sample = self._create_sample(joint, bkd)
        jac = joint.logpdf_jacobian(sample)
        assert jac.shape == (1, joint.nvars())

    @pytest.mark.parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_logpdf_jacobian_batch_shape(
        self, bkd, name: str, marginal_specs: list
    ) -> None:
        """Test logpdf_jacobian_batch shape for different marginal combinations."""
        nsamples = 5
        joint = self._create_joint(marginal_specs, bkd)
        samples = self._create_samples(joint, bkd, nsamples)
        jac = joint.logpdf_jacobian_batch(samples)
        assert jac.shape == (nsamples, 1, joint.nvars())

    @pytest.mark.parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_jacobian_numerical(self, bkd, name: str, marginal_specs: list) -> None:
        """Test jacobian against numerical derivatives."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        # Skip single uniform - has zero Jacobian everywhere, numerical check
        # returns NaN
        if name == "1_uniform":
            pytest.skip("Uniform has zero Jacobian, numerical check not applicable")

        joint = self._create_joint(marginal_specs, bkd)
        checker = DerivativeChecker(joint)  # type: ignore[arg-type]
        sample = self._create_sample(joint, bkd)
        # Use smaller fd_eps to avoid stepping outside valid domain
        fd_eps = bkd.asarray([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
        errors = checker.check_derivatives(sample, fd_eps=fd_eps, verbosity=0)
        ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        # Use 1e-5 tolerance for numerical derivatives (some loss expected)
        assert ratio <= 1e-5

    @pytest.mark.parametrize(
        "name,marginal_specs",
        MARGINAL_COMBOS,
    )
    def test_jacobian_batch_consistency(
        self, bkd, name: str, marginal_specs: list
    ) -> None:
        """Test jacobian_batch matches single jacobian."""
        joint = self._create_joint(marginal_specs, bkd)
        samples = self._create_samples(joint, bkd, 3)
        batch_jac = joint.jacobian_batch(samples)
        for ii in range(3):
            single = samples[:, ii : ii + 1]
            single_jac = joint.jacobian(single)
            bkd.assert_allclose(batch_jac[ii, 0, :], single_jac[0, :])
