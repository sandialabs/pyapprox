"""
Tests for SASNormalMarginal distribution.
"""

import numpy as np
import pytest

from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.probability.univariate.sas_normal import SASNormalMarginal


class TestSASNormalMarginal:
    """Tests for SASNormalMarginal."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd) -> None:
        self._bkd = bkd
        # General SAS parameters
        self._xi = 1.0
        self._eta = 2.0
        self._epsilon = 0.5
        self._delta = 1.5
        self._dist = SASNormalMarginal(
            self._xi, self._eta, self._epsilon, self._delta, bkd
        )

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        assert self._dist.nvars() == 1

    def test_nparams(self) -> None:
        """Test nparams returns 4."""
        assert self._dist.nparams() == 4

    def test_is_bounded(self) -> None:
        """Test SAS is unbounded."""
        assert not self._dist.is_bounded()

    def test_gaussian_reduction_logpdf(self) -> None:
        """At epsilon=0, delta=1 SAS reduces to N(xi, eta^2)."""
        bkd = self._bkd
        xi, eta = 2.0, 1.5
        sas = SASNormalMarginal(xi, eta, 0.0, 1.0, bkd)
        gauss = GaussianMarginal(xi, eta, bkd)

        samples = bkd.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]])
        bkd.assert_allclose(sas.logpdf(samples), gauss.logpdf(samples),
                            rtol=1e-12)

    def test_gaussian_reduction_cdf(self) -> None:
        """At epsilon=0, delta=1 SAS CDF matches Gaussian CDF."""
        bkd = self._bkd
        xi, eta = 2.0, 1.5
        sas = SASNormalMarginal(xi, eta, 0.0, 1.0, bkd)
        gauss = GaussianMarginal(xi, eta, bkd)

        samples = bkd.asarray([[-1.0, 0.0, 1.0, 2.0, 3.0, 5.0]])
        bkd.assert_allclose(sas.cdf(samples), gauss.cdf(samples), rtol=1e-12)

    def test_gaussian_reduction_invcdf(self) -> None:
        """At epsilon=0, delta=1 SAS invcdf matches Gaussian invcdf."""
        bkd = self._bkd
        xi, eta = 2.0, 1.5
        sas = SASNormalMarginal(xi, eta, 0.0, 1.0, bkd)
        gauss = GaussianMarginal(xi, eta, bkd)

        probs = bkd.asarray([[0.1, 0.25, 0.5, 0.75, 0.9]])
        bkd.assert_allclose(sas.invcdf(probs), gauss.invcdf(probs),
                            rtol=1e-12)

    def test_gaussian_reduction_reparameterize(self) -> None:
        """At epsilon=0, delta=1 reparameterize matches Gaussian."""
        bkd = self._bkd
        xi, eta = 2.0, 1.5
        sas = SASNormalMarginal(xi, eta, 0.0, 1.0, bkd)
        gauss = GaussianMarginal(xi, eta, bkd)

        base = bkd.asarray([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        bkd.assert_allclose(sas.reparameterize(base),
                            gauss.reparameterize(base), rtol=1e-12)

    def test_cdf_invcdf_roundtrip(self) -> None:
        """Test cdf(invcdf(p)) = p."""
        bkd = self._bkd
        probs = bkd.asarray([[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]])
        recovered = self._dist.cdf(self._dist.invcdf(probs))
        bkd.assert_allclose(recovered, probs, rtol=1e-10)

    def test_invcdf_cdf_roundtrip(self) -> None:
        """Test invcdf(cdf(x)) = x for moderate values."""
        bkd = self._bkd
        # Avoid extreme values where CDF is near 0/1 (erfinv precision)
        samples = bkd.asarray([[0.0, 1.0, 2.0, 3.0]])
        recovered = self._dist.invcdf(self._dist.cdf(samples))
        bkd.assert_allclose(recovered, samples, atol=1e-10)

    def test_reparameterize_matches_invcdf(self) -> None:
        """reparameterize(n) = invcdf(Phi(n))."""
        bkd = self._bkd
        base = bkd.asarray([[-2.0, -1.0, 0.0, 0.5, 1.5]])
        reparam = self._dist.reparameterize(base)
        # Phi(n) = 0.5 * (1 + erf(n / sqrt(2)))
        phi = 0.5 * (1.0 + bkd.erf(base / np.sqrt(2.0)))
        via_invcdf = self._dist.invcdf(phi)
        bkd.assert_allclose(reparam, via_invcdf, rtol=1e-10)

    def test_logpdf_exp_equals_pdf(self) -> None:
        """Test exp(logpdf) = pdf."""
        bkd = self._bkd
        samples = bkd.asarray([[-1.0, 0.0, 1.0, 2.0, 3.0]])
        bkd.assert_allclose(
            bkd.exp(self._dist.logpdf(samples)),
            self._dist.pdf(samples),
            rtol=1e-12,
        )

    def test_pdf_integrates_to_one(self) -> None:
        """Test that PDF integrates to approximately 1 via quadrature."""
        bkd = self._bkd
        # Use invcdf to map uniform grid to sample space
        u = np.linspace(1e-5, 1 - 1e-5, 5000)
        u_bkd = bkd.asarray(u.reshape(1, -1))
        samples = self._dist.invcdf(u_bkd)
        pdf_vals = self._dist.pdf(samples)

        # Integrate: int f(x) dx = int f(F^{-1}(u)) * dF^{-1}/du du
        # = int 1/f(F^{-1}(u)) * f(F^{-1}(u)) du (by invcdf derivative)
        # = int 1 du = 1
        # But directly: int f(x)dx via trapezoidal on x grid
        x_np = bkd.to_numpy(samples[0])
        f_np = bkd.to_numpy(pdf_vals[0])
        integral = float(np.trapezoid(f_np, x_np))
        bkd.assert_allclose(
            bkd.asarray([integral]), bkd.asarray([1.0]), atol=5e-4
        )

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self._dist.rvs(100)
        assert samples.shape == (1, 100)

    def test_input_validation_1d(self) -> None:
        """Test that 1D input raises ValueError."""
        bkd = self._bkd
        with pytest.raises(ValueError, match="2D"):
            self._dist.logpdf(bkd.asarray([1.0, 2.0]))

    def test_input_validation_wrong_first_dim(self) -> None:
        """Test that wrong first dimension raises ValueError."""
        bkd = self._bkd
        with pytest.raises(ValueError, match="shape"):
            self._dist.logpdf(bkd.asarray([[1.0], [2.0]]))

    def test_base_distribution(self) -> None:
        """Test base_distribution returns standard normal."""
        base = self._dist.base_distribution()
        assert isinstance(base, GaussianMarginal)

    def test_repr(self) -> None:
        """Test string representation."""
        repr_str = repr(self._dist)
        assert "SASNormalMarginal" in repr_str

    def test_equality(self) -> None:
        """Test equality comparison."""
        bkd = self._bkd
        dist2 = SASNormalMarginal(
            self._xi, self._eta, self._epsilon, self._delta, bkd
        )
        dist3 = SASNormalMarginal(0.0, 1.0, 0.0, 1.0, bkd)
        assert self._dist == dist2
        assert self._dist != dist3

    def test_ppf_alias(self) -> None:
        """Test ppf is alias for invcdf."""
        bkd = self._bkd
        probs = bkd.asarray([[0.25, 0.5, 0.75]])
        bkd.assert_allclose(
            self._dist.ppf(probs), self._dist.invcdf(probs), rtol=1e-12
        )

    def test_skewness_shifts_median(self) -> None:
        """Positive epsilon shifts the median above the location."""
        bkd = self._bkd
        sas_pos = SASNormalMarginal(0.0, 1.0, 1.0, 1.0, bkd)
        sas_neg = SASNormalMarginal(0.0, 1.0, -1.0, 1.0, bkd)
        median_pos = sas_pos.invcdf(bkd.asarray([[0.5]]))
        median_neg = sas_neg.invcdf(bkd.asarray([[0.5]]))
        # Positive skew -> median > 0, negative skew -> median < 0
        assert float(bkd.to_numpy(median_pos[0, 0])) > 0.0
        assert float(bkd.to_numpy(median_neg[0, 0])) < 0.0

    def test_tail_weight_controls_kurtosis(self) -> None:
        """Small delta produces heavier tails (wider spread)."""
        bkd = self._bkd
        sas_light = SASNormalMarginal(0.0, 1.0, 0.0, 2.0, bkd)
        sas_heavy = SASNormalMarginal(0.0, 1.0, 0.0, 0.5, bkd)

        # Compare 99th percentile: heavy tails should be wider
        p99 = bkd.asarray([[0.99]])
        q_light = float(bkd.to_numpy(sas_light.invcdf(p99)[0, 0]))
        q_heavy = float(bkd.to_numpy(sas_heavy.invcdf(p99)[0, 0]))
        assert q_heavy > q_light
