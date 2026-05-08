"""
Tests for DVineCopula.
"""

import math
from typing import Dict, List

import numpy as np
import pytest

from pyapprox.probability.copula.bivariate.gaussian import (
    BivariateGaussianCopula,
)
from pyapprox.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.probability.copula.gaussian import GaussianCopula
from pyapprox.probability.copula.vine.dvine import DVineCopula
from pyapprox.probability.copula.vine.helpers import (
    correlation_from_partial_correlations,
)
from pyapprox.probability.gaussian.dense import (
    DenseCholeskyMultivariateGaussian,
)

_SQRT2 = math.sqrt(2.0)

# TODO: Fix typing issues


def _make_gaussian_dvine(partial_corrs, nvars, bkd):
    """Build a DVineCopula with BivariateGaussianCopula pair copulas."""
    truncation_level = max(partial_corrs.keys()) if partial_corrs else 0
    pair_copulas: Dict[int, list] = {}
    for t in range(1, truncation_level + 1):
        pair_copulas[t] = [
            BivariateGaussianCopula(rho, bkd) for rho in partial_corrs[t]
        ]
    return DVineCopula(pair_copulas, nvars, truncation_level, bkd)


def _make_gaussian_copula(partial_corrs, nvars, bkd):
    """Build a GaussianCopula with the correlation matrix implied by
    the given partial correlations."""
    R = correlation_from_partial_correlations(partial_corrs, nvars, bkd)
    L = bkd.cholesky(R)
    # Extract strict lower-triangular elements row-by-row
    chol_lower_vals = []
    for i in range(nvars):
        for j in range(i):
            chol_lower_vals.append(L[i, j])
    chol_lower = bkd.hstack([bkd.reshape(v, (1,)) for v in chol_lower_vals])
    corr_param = CholeskyCorrelationParameterization(chol_lower, nvars, bkd)
    return GaussianCopula(corr_param, bkd)



# TODO: Do we really need to redefine these
# Can we import from probability.univariate.gaussian
def _standard_normal_invcdf(u, bkd):
    """Phi^{-1}(u) using erfinv (autograd-safe, no scipy)."""
    return _SQRT2 * bkd.erfinv(2.0 * u - 1.0)


# TODO: Do we really need to redefine these
# Can we import from probability.univariate.gaussian
def _standard_normal_cdf(z, bkd):
    """Phi(z) using erf (autograd-safe, no scipy)."""
    return 0.5 * (1.0 + bkd.erf(z / _SQRT2))

# TODO: Cant we just use bkd.cov if it does not exist in backend
# add it to backend protocol
def _empirical_correlation(z, bkd):
    """Compute empirical correlation matrix from (nvars, nsamples) array."""
    nvars = z.shape[0]
    nsamples = z.shape[1]
    mu = bkd.mean(z, axis=1)  # (nvars,)
    z_centered = z - bkd.reshape(mu, (nvars, 1))
    cov = (z_centered @ z_centered.T) / (nsamples - 1)
    d = bkd.sqrt(bkd.get_diagonal(cov))
    d_inv = 1.0 / d
    return cov * bkd.outer(d_inv, d_inv)


class TestDVineCopula:
    """Base tests for DVineCopula."""

    # -- Constructor validation --

    def test_constructor_validation_missing_tree(self, bkd) -> None:
        """Missing tree level raises ValueError."""
        pc = {
            1: [BivariateGaussianCopula(0.5, bkd) for _ in range(2)],
        }
        with pytest.raises(ValueError):
            DVineCopula(pc, 3, 2, bkd)

    def test_constructor_validation_wrong_count(self, bkd) -> None:
        """Wrong number of copulas at a tree level raises ValueError."""
        pc = {
            1: [BivariateGaussianCopula(0.5, bkd)],  # need 2
        }
        with pytest.raises(ValueError):
            DVineCopula(pc, 3, 1, bkd)

    def test_constructor_validation_invalid_truncation(self, bkd) -> None:
        """Truncation level > nvars - 1 raises ValueError."""
        pc = {
            1: [BivariateGaussianCopula(0.5, bkd)],
        }
        with pytest.raises(ValueError):
            DVineCopula(pc, 2, 5, bkd)

    def test_constructor_validation_negative_truncation(self, bkd) -> None:
        """Negative truncation level raises ValueError."""
        with pytest.raises(ValueError):
            DVineCopula({}, 3, -1, bkd)

    # -- Property methods --

    def test_nvars_nparams(self, bkd) -> None:
        """Correct nvars and nparams for a 3-var vine."""
        partial_corrs = {1: [0.6, 0.5], 2: [0.3]}
        dvine = _make_gaussian_dvine(partial_corrs, 3, bkd)
        assert dvine.nvars() == 3
        assert dvine.nparams() == 3
        assert dvine.npair_copulas() == 3
        assert dvine.truncation_level() == 2

    def test_pair_copula_access(self, bkd) -> None:
        """Retrieve specific pair copula by tree and edge."""
        partial_corrs = {1: [0.6, 0.5], 2: [0.3]}
        dvine = _make_gaussian_dvine(partial_corrs, 3, bkd)
        pc = dvine.pair_copula(1, 0)
        assert isinstance(pc, BivariateGaussianCopula)

    # -- Logpdf tests --

    def test_logpdf_shape(self, bkd) -> None:
        """Logpdf output has shape (1, nsamples)."""
        np.random.seed(42)
        partial_corrs = {1: [0.6, 0.5], 2: [0.3]}
        dvine = _make_gaussian_dvine(partial_corrs, 3, bkd)
        u = bkd.asarray(np.random.uniform(0.01, 0.99, (3, 20)).astype(np.float64))
        result = dvine.logpdf(u)
        assert result.shape == (1, 20)

    def test_logpdf_independence_is_zero(self, bkd) -> None:
        """All rho=0 produces logpdf approximately zero."""
        np.random.seed(42)
        partial_corrs = {1: [0.0, 0.0, 0.0]}
        dvine = _make_gaussian_dvine(partial_corrs, 4, bkd)
        u = bkd.asarray(np.random.uniform(0.01, 0.99, (4, 50)).astype(np.float64))
        result = dvine.logpdf(u)
        expected = bkd.zeros((1, 50))
        bkd.assert_allclose(result, expected, atol=1e-10)

    def test_logpdf_bivariate_matches(self, bkd) -> None:
        """2-var D-vine matches BivariateGaussianCopula."""
        np.random.seed(42)
        rho = 0.7
        dvine = _make_gaussian_dvine({1: [rho]}, 2, bkd)
        biv = BivariateGaussianCopula(rho, bkd)

        u = bkd.asarray(np.random.uniform(0.01, 0.99, (2, 30)).astype(np.float64))
        dvine_logpdf = dvine.logpdf(u)
        biv_logpdf = biv.logpdf(u)
        bkd.assert_allclose(dvine_logpdf, biv_logpdf, rtol=1e-12)

    def test_logpdf_3var_gaussian_matches(self, bkd) -> None:
        """3-var all-Gaussian D-vine matches GaussianCopula(Sigma)."""
        np.random.seed(42)
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5],
            2: [0.3],
        }
        dvine = _make_gaussian_dvine(partial_corrs, 3, bkd)
        gc = _make_gaussian_copula(partial_corrs, 3, bkd)

        u = bkd.asarray(np.random.uniform(0.01, 0.99, (3, 50)).astype(np.float64))
        dvine_logpdf = dvine.logpdf(u)
        gc_logpdf = gc.logpdf(u)
        bkd.assert_allclose(dvine_logpdf, gc_logpdf, rtol=1e-8)

    def test_logpdf_4var_gaussian_matches(self, bkd) -> None:
        """4-var all-Gaussian D-vine matches GaussianCopula(Sigma)."""
        np.random.seed(42)
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
            2: [0.3, -0.2],
            3: [0.1],
        }
        dvine = _make_gaussian_dvine(partial_corrs, 4, bkd)
        gc = _make_gaussian_copula(partial_corrs, 4, bkd)

        u = bkd.asarray(np.random.uniform(0.01, 0.99, (4, 50)).astype(np.float64))
        dvine_logpdf = dvine.logpdf(u)
        gc_logpdf = gc.logpdf(u)
        bkd.assert_allclose(dvine_logpdf, gc_logpdf, rtol=1e-8)

    def test_logpdf_4var_matches_mvn_density(self, bkd) -> None:
        """DVine copula logpdf matches the exact MVN density decomposition.

        For a Gaussian copula: log c(u) = logpdf_MVN(z; 0, R) - sum logpdf_N(z_i)
        where z_i = Phi^{-1}(u_i) and R is the correlation matrix.
        """
        np.random.seed(42)
        nvars = 4
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
            2: [0.3, -0.2],
            3: [0.1],
        }
        dvine = _make_gaussian_dvine(partial_corrs, nvars, bkd)

        R = correlation_from_partial_correlations(partial_corrs, nvars, bkd)

        mean_zero = bkd.zeros((nvars, 1))
        mvn = DenseCholeskyMultivariateGaussian(mean_zero, R, bkd)

        u = bkd.asarray(
            np.random.uniform(0.01, 0.99, (nvars, 50)).astype(np.float64)
        )
        u_clipped = bkd.clip(u, 1e-10, 1.0 - 1e-10)
        z = _standard_normal_invcdf(u_clipped, bkd)

        mvn_logpdf = mvn.logpdf(z)  # (1, nsamples)

        log_2pi = math.log(2.0 * math.pi)
        marginal_logpdf = bkd.sum(-0.5 * log_2pi - 0.5 * z * z, axis=0)
        marginal_logpdf = bkd.reshape(marginal_logpdf, (1, -1))

        expected_copula_logpdf = mvn_logpdf - marginal_logpdf

        dvine_logpdf = dvine.logpdf(u)
        bkd.assert_allclose(dvine_logpdf, expected_copula_logpdf, rtol=1e-8)

    def test_logpdf_truncated_vs_full(self, bkd) -> None:
        """Truncated vine ignores higher tree contributions."""
        # TODO: Weak test. Check this is already likely covered better
        # by gaussian newtwork comparison that checks covariance from
        # truncated dvine matches that of Gaussian network with the same graph
        np.random.seed(42)
        partial_corrs_full: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
            2: [0.3, -0.2],
            3: [0.1],
        }
        dvine_full = _make_gaussian_dvine(partial_corrs_full, 4, bkd)

        partial_corrs_trunc: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
        }
        dvine_trunc = _make_gaussian_dvine(partial_corrs_trunc, 4, bkd)

        u = bkd.asarray(np.random.uniform(0.01, 0.99, (4, 30)).astype(np.float64))
        logpdf_full = dvine_full.logpdf(u)
        logpdf_trunc = dvine_trunc.logpdf(u)

        # They should differ (higher trees contribute non-trivially)
        diff = bkd.abs(logpdf_full - logpdf_trunc)
        max_diff = bkd.max(diff)
        assert float(bkd.to_numpy(max_diff)) > 0.01

        # Truncated should match a vine with rho=0 in higher trees
        partial_corrs_zero_higher: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
            2: [0.0, 0.0],
            3: [0.0],
        }
        dvine_zero = _make_gaussian_dvine(partial_corrs_zero_higher, 4, bkd)
        logpdf_zero = dvine_zero.logpdf(u)
        bkd.assert_allclose(logpdf_trunc, logpdf_zero, rtol=1e-12)

    # -- Sampling tests --

    def test_sample_shape_and_range(self, bkd) -> None:
        """Samples have correct shape and values in (0, 1)."""
        np.random.seed(42)
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5],
            2: [0.3],
        }
        dvine = _make_gaussian_dvine(partial_corrs, 3, bkd)
        samples = dvine.sample(100)
        assert samples.shape == (3, 100)
        # All values in (0, 1)
        min_val = bkd.min(samples)
        max_val = bkd.max(samples)
        assert float(bkd.to_numpy(min_val)) > 0.0
        assert float(bkd.to_numpy(max_val)) < 1.0

    def test_sample_correlation_matches(self, bkd) -> None:
        """Empirical correlation from large sample approximates the
        correlation matrix implied by the D-vine partial correlations."""
        np.random.seed(42)
        nvars = 4
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
            2: [0.3, -0.2],
            3: [0.1],
        }
        dvine = _make_gaussian_dvine(partial_corrs, nvars, bkd)

        R = correlation_from_partial_correlations(partial_corrs, nvars, bkd)

        nsamples = 50000
        samples = dvine.sample(nsamples)

        # Transform to standard normal using backend erfinv
        u_clipped = bkd.clip(samples, 1e-10, 1.0 - 1e-10)
        z = _standard_normal_invcdf(u_clipped, bkd)

        empirical_corr = _empirical_correlation(z, bkd)

        bkd.assert_allclose(empirical_corr, R, atol=0.03)

    def test_sample_logpdf_recovers_mvn_density(self, bkd) -> None:
        """copula_logpdf(u) + sum marginal_logpdf(x_i) = mvn_logpdf(x).

        Generate samples from DenseCholeskyMultivariateGaussian(0, R),
        transform to copula space u_i = Phi(x_i), evaluate D-vine
        copula logpdf, and verify the density decomposition holds exactly.
        """
        np.random.seed(42)
        nvars = 4
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
            2: [0.3, -0.2],
            3: [0.1],
        }
        dvine = _make_gaussian_dvine(partial_corrs, nvars, bkd)

        R = correlation_from_partial_correlations(partial_corrs, nvars, bkd)
        mean_zero = bkd.zeros((nvars, 1))
        mvn = DenseCholeskyMultivariateGaussian(mean_zero, R, bkd)

        nsamples = 100
        x = mvn.rvs(nsamples)  # (nvars, nsamples)

        # Transform to copula space: u_i = Phi(x_i)
        u = _standard_normal_cdf(x, bkd)

        # MVN logpdf at x
        mvn_logpdf = mvn.logpdf(x)  # (1, nsamples)

        # Sum of marginal standard normal logpdf
        log_2pi = math.log(2.0 * math.pi)
        marginal_logpdf_sum = bkd.sum(-0.5 * log_2pi - 0.5 * x * x, axis=0)
        marginal_logpdf_sum = bkd.reshape(marginal_logpdf_sum, (1, -1))

        # Copula logpdf
        copula_logpdf = dvine.logpdf(u)

        # Verify: copula_logpdf + marginal_sum = mvn_logpdf
        bkd.assert_allclose(
            copula_logpdf + marginal_logpdf_sum, mvn_logpdf, rtol=1e-8
        )

    def test_sample_gaussian_statistics(self, bkd) -> None:
        """For all-Gaussian D-vine: sample covariance approximates
        the correlation matrix when transformed to normal space."""
        np.random.seed(42)
        nvars = 4
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
            2: [0.3, -0.2],
            3: [0.1],
        }
        dvine = _make_gaussian_dvine(partial_corrs, nvars, bkd)

        R = correlation_from_partial_correlations(partial_corrs, nvars, bkd)

        nsamples = 50000
        samples = dvine.sample(nsamples)

        u_clipped = bkd.clip(samples, 1e-10, 1.0 - 1e-10)
        z = _standard_normal_invcdf(u_clipped, bkd)

        # Mean should be approximately 0
        z_mean = bkd.mean(z, axis=1)
        bkd.assert_allclose(z_mean, bkd.zeros((nvars,)), atol=0.05)

        # Covariance should approximate correlation matrix
        z_centered = z - bkd.reshape(z_mean, (nvars, 1))
        empirical_cov = (z_centered @ z_centered.T) / (nsamples - 1)
        bkd.assert_allclose(empirical_cov, R, atol=0.03)

    # -- Precision / correlation matrix tests --

    def test_from_precision_matrix_roundtrip_tridiag(self, bkd) -> None:
        """Tridiag Omega -> from_precision_matrix -> to_precision_matrix
        recovers exact values."""
        # Build a known tridiagonal precision matrix (bandwidth 1)
        # Start from partial correlations, build R, invert to get Omega
        nvars = 4
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
        }
        R = correlation_from_partial_correlations(partial_corrs, nvars, bkd)
        Omega = bkd.inv(R)

        dvine = DVineCopula.from_precision_matrix(Omega, bkd)
        assert dvine.truncation_level() == 1
        assert dvine.nvars() == nvars

        Omega_recovered = dvine.to_precision_matrix()
        bkd.assert_allclose(Omega_recovered, Omega, rtol=1e-10, atol=1e-14)

    def test_from_precision_matrix_roundtrip_bandwidth2(self, bkd) -> None:
        """Pentadiag Omega -> from_precision_matrix -> to_precision_matrix
        recovers exact values."""
        nvars = 5
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7, 0.4],
            2: [0.3, -0.2, 0.1],
        }
        R = correlation_from_partial_correlations(partial_corrs, nvars, bkd)
        Omega = bkd.inv(R)

        dvine = DVineCopula.from_precision_matrix(Omega, bkd)
        assert dvine.truncation_level() == 2

        Omega_recovered = dvine.to_precision_matrix()
        bkd.assert_allclose(Omega_recovered, Omega, rtol=1e-10, atol=1e-14)

    def test_from_precision_matrix_diagonal(self, bkd) -> None:
        """Diagonal Omega -> trunc=0, to_precision_matrix recovers it."""
        Omega = bkd.asarray(np.diag([2.0, 3.0, 1.5, 4.0]).astype(np.float64))
        dvine = DVineCopula.from_precision_matrix(Omega, bkd)
        assert dvine.truncation_level() == 0
        assert dvine.npair_copulas() == 0

    def test_from_precision_matrix_correlation_recovery(self, bkd) -> None:
        """to_correlation_matrix matches normalized inv(Omega)."""
        nvars = 4
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
            2: [0.3, -0.2],
            3: [0.1],
        }
        R = correlation_from_partial_correlations(partial_corrs, nvars, bkd)
        Omega = bkd.inv(R)

        dvine = DVineCopula.from_precision_matrix(Omega, bkd)
        R_recovered = dvine.to_correlation_matrix()
        bkd.assert_allclose(R_recovered, R, rtol=1e-10)

    def test_from_precision_matrix_logpdf_matches_mvn(self, bkd) -> None:
        """DVine from precision matrix has logpdf matching MVN density."""
        np.random.seed(42)
        nvars = 4
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5, 0.7],
            2: [0.3, -0.2],
        }
        R = correlation_from_partial_correlations(partial_corrs, nvars, bkd)
        Omega = bkd.inv(R)

        dvine = DVineCopula.from_precision_matrix(Omega, bkd)

        mean_zero = bkd.zeros((nvars, 1))
        mvn = DenseCholeskyMultivariateGaussian(mean_zero, R, bkd)

        # Generate samples from MVN and verify density decomposition
        x = mvn.rvs(50)
        u = _standard_normal_cdf(x, bkd)

        mvn_logpdf = mvn.logpdf(x)
        log_2pi = math.log(2.0 * math.pi)
        marginal_logpdf_sum = bkd.sum(-0.5 * log_2pi - 0.5 * x * x, axis=0)
        marginal_logpdf_sum = bkd.reshape(marginal_logpdf_sum, (1, -1))

        copula_logpdf = dvine.logpdf(u)
        bkd.assert_allclose(
            copula_logpdf + marginal_logpdf_sum, mvn_logpdf, rtol=1e-8
        )

    def test_to_correlation_matrix_gaussian(self, bkd) -> None:
        """to_correlation_matrix recovers correct R from known partials."""
        nvars = 3
        partial_corrs: Dict[int, List[float]] = {
            1: [0.6, 0.5],
            2: [0.3],
        }
        dvine = _make_gaussian_dvine(partial_corrs, nvars, bkd)
        R_expected = correlation_from_partial_correlations(
            partial_corrs, nvars, bkd
        )
        R_actual = dvine.to_correlation_matrix()
        bkd.assert_allclose(R_actual, R_expected, rtol=1e-12)

    def test_to_correlation_matrix_rejects_non_gaussian(self, bkd) -> None:
        """Non-Gaussian pair copula raises TypeError."""
        from pyapprox.probability.copula.bivariate.clayton import (
            ClaytonCopula,
        )

        pc = {
            1: [ClaytonCopula(2.0, bkd)],
        }
        dvine = DVineCopula(pc, 2, 1, bkd)
        with pytest.raises(TypeError):
            dvine.to_correlation_matrix()
