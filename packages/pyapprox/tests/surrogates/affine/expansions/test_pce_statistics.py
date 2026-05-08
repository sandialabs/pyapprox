"""Tests for PCE statistics and Sobol sensitivity indices.

Tests the statistics functions in:
- pyapprox.surrogates.affine.expansions.pce_statistics
- PolynomialChaosExpansion methods: mean, variance, std, covariance
- Sobol indices: total, main effect, and interaction indices
"""

import numpy as np

from pyapprox.probability.univariate import UniformMarginal
from pyapprox.surrogates.affine.expansions import (
    create_pce,
    pce_statistics,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d


class TestPCEMeanVariance:
    """Test PCE mean and variance computation."""

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_mean_constant_function(self, bkd):
        """Test mean for constant function: E[c] = c."""
        pce = self._create_pce(bkd, nvars=2, max_level=2)

        # Set coefficient for constant term only
        coef = bkd.zeros((pce.nterms(), 1))
        const_idx = pce._get_constant_index()
        coef[const_idx, 0] = 5.0
        pce.set_coefficients(coef)

        mean = pce.mean()
        bkd.assert_allclose(mean, bkd.asarray([5.0]))

    def test_variance_constant_function(self, bkd):
        """Test variance for constant function: Var[c] = 0."""
        pce = self._create_pce(bkd, nvars=2, max_level=2)

        coef = bkd.zeros((pce.nterms(), 1))
        const_idx = pce._get_constant_index()
        coef[const_idx, 0] = 5.0
        pce.set_coefficients(coef)

        var = pce.variance()
        bkd.assert_allclose(var, bkd.asarray([0.0]))

    def test_variance_single_nonconst_term(self, bkd):
        """Test variance with one non-constant term: Var[f] = c_i^2."""
        pce = self._create_pce(bkd, nvars=2, max_level=2)

        coef = bkd.zeros((pce.nterms(), 1))
        const_idx = pce._get_constant_index()
        coef[const_idx, 0] = 2.0  # Mean = 2
        other_idx = 1 if const_idx != 1 else 2
        coef[other_idx, 0] = 3.0  # Single non-constant coefficient
        pce.set_coefficients(coef)

        var = pce.variance()
        # Variance = 3^2 = 9
        bkd.assert_allclose(var, bkd.asarray([9.0]))

    def test_std_matches_sqrt_variance(self, bkd):
        """Test that std = sqrt(variance)."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        # Fit to a function
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x * y, (1, -1))
        pce.fit(samples, values)

        std = pce.std()
        var = pce.variance()
        bkd.assert_allclose(std, bkd.sqrt(var))

    def test_analytical_mean_variance(self, bkd):
        """Test mean/variance against analytical values.

        For f(x,y) = 1 + x + y + xy on uniform [-1,1]^2:
        - Mean = E[1] + E[x] + E[y] + E[xy] = 1 + 0 + 0 + 0 = 1
        - Var = Var[x] + Var[y] + Var[xy] = 1/3 + 1/3 + 1/9 = 7/9
        """
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(1.0 + x + y + x * y, (1, -1))
        pce.fit(samples, values)

        mean = pce.mean()
        var = pce.variance()

        bkd.assert_allclose(mean, bkd.asarray([1.0]), atol=1e-10)
        bkd.assert_allclose(var, bkd.asarray([7.0 / 9.0]), atol=1e-10)


class TestPCECovariance:
    """Test PCE covariance computation for multi-QoI."""

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_covariance_diagonal_equals_variance(self, bkd):
        """Test that diagonal of covariance equals variance."""
        pce = self._create_pce(bkd, nvars=2, max_level=3, nqoi=2)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.vstack(
            [
                bkd.reshape(x + y, (1, -1)),
                bkd.reshape(x * y, (1, -1)),
            ]
        )
        pce.fit(samples, values)

        cov = pce.covariance()
        var = pce.variance()

        # Diagonal should equal variance
        bkd.assert_allclose(bkd.diag(cov), var)

    def test_covariance_symmetry(self, bkd):
        """Test covariance matrix is symmetric."""
        pce = self._create_pce(bkd, nvars=2, max_level=3, nqoi=3)

        np.random.seed(42)
        coef = bkd.asarray(np.random.randn(pce.nterms(), 3))
        pce.set_coefficients(coef)

        cov = pce.covariance()
        bkd.assert_allclose(cov, cov.T)

    def test_covariance_positive_semidefinite(self, bkd):
        """Test covariance matrix is positive semidefinite."""
        pce = self._create_pce(bkd, nvars=2, max_level=3, nqoi=3)

        np.random.seed(42)
        coef = bkd.asarray(np.random.randn(pce.nterms(), 3))
        pce.set_coefficients(coef)

        cov = pce.covariance()
        eigenvalues, _ = bkd.eigh(cov)

        # All eigenvalues should be >= 0 (with small tolerance for numerics)
        min_eigenvalue = float(bkd.to_numpy(bkd.min(eigenvalues)))
        assert min_eigenvalue + 1e-10 >= 0.0


class TestSobolIndices:
    """Test PCE Sobol sensitivity indices."""

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_total_sobol_sum_with_interactions(self, bkd):
        """Test total Sobol indices sum >= 1 when interactions present."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        # f(x,y) = x + y + xy has interactions
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x * y, (1, -1))
        pce.fit(samples, values)

        total_indices = pce.total_sobol_indices()

        # Sum >= 1 due to double-counting interactions
        total_sum = float(bkd.to_numpy(bkd.sum(total_indices)))
        assert total_sum >= 1.0 - 1e-10

        # Each index should be non-negative
        assert np.all(bkd.to_numpy(total_indices) >= -1e-10)

    def test_main_effect_sum_bounded(self, bkd):
        """Test main effect Sobol indices sum to <= 1."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x * y, (1, -1))
        pce.fit(samples, values)

        main_indices = pce.main_effect_sobol_indices()

        # Main effects should sum to <= 1
        main_sum = float(bkd.to_numpy(bkd.sum(main_indices)))
        assert main_sum <= 1.0 + 1e-10

    def test_additive_function_main_equals_total(self, bkd):
        """Test Sobol indices for additive function (no interactions).

        For f(x,y) = x + 2y, main effect = total effect.
        """
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + 2 * y, (1, -1))
        pce.fit(samples, values)

        main_indices = pce.main_effect_sobol_indices()
        total_indices = pce.total_sobol_indices()

        # For additive function, main = total
        bkd.assert_allclose(main_indices, total_indices, atol=1e-10)

        # Main effects should sum to 1
        bkd.assert_allclose(
            bkd.reshape(bkd.sum(main_indices), (1,)), bkd.asarray([1.0]), atol=1e-10
        )

    def test_analytical_sobol_indices(self, bkd):
        """Test Sobol indices against analytical values.

        For f(x,y) = x + 2y on uniform [-1,1]^2:
        - Var[x] = 1/3, Var[2y] = 4/3
        - Total variance = 5/3
        - S_x = (1/3) / (5/3) = 1/5 = 0.2
        - S_y = (4/3) / (5/3) = 4/5 = 0.8
        """
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + 2 * y, (1, -1))
        pce.fit(samples, values)

        main_indices = pce.main_effect_sobol_indices()

        # S_x = 0.2, S_y = 0.8
        expected = bkd.asarray([[0.2], [0.8]])
        bkd.assert_allclose(main_indices, expected, atol=1e-10)


class TestInteractionSobolIndices:
    """Test interaction Sobol indices."""

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_interaction_indices_additive_function(self, bkd):
        """Test interaction indices are zero for additive function."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        # Additive function: no interactions
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y, (1, -1))
        pce.fit(samples, values)

        # Interaction index S_{0,1} should be 0
        interaction = pce_statistics.interaction_sobol_indices(pce, [(0, 1)])
        bkd.assert_allclose(interaction, bkd.zeros((1, 1)), atol=1e-10)

    def test_interaction_indices_with_product_term(self, bkd):
        """Test interaction indices are nonzero for product term."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        # f(x,y) = xy has interaction but no main effects
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x * y, (1, -1))
        pce.fit(samples, values)

        # Interaction index S_{0,1} should be ~1 (all variance from interaction)
        interaction = pce_statistics.interaction_sobol_indices(pce, [(0, 1)])
        bkd.assert_allclose(interaction, bkd.asarray([[1.0]]), atol=1e-10)

        # Main effects should be ~0
        main_indices = pce.main_effect_sobol_indices()
        bkd.assert_allclose(main_indices, bkd.zeros((2, 1)), atol=1e-10)

    def test_sobol_decomposition(self, bkd):
        """Test that main + interactions sum to 1.

        For f(x,y) = x + y + xy:
        S_x + S_y + S_{xy} = 1
        """
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x * y, (1, -1))
        pce.fit(samples, values)

        main_indices = pce.main_effect_sobol_indices()
        interaction = pce_statistics.interaction_sobol_indices(pce, [(0, 1)])

        # S_x + S_y + S_{xy} should equal 1
        total = bkd.sum(main_indices) + bkd.sum(interaction)
        bkd.assert_allclose(bkd.reshape(total, (1,)), bkd.asarray([1.0]), atol=1e-10)


class TestPCEStatisticsFunctions:
    """Test standalone PCE statistics functions match methods."""

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_functions_match_methods(self, bkd):
        """Test standalone functions match PCE methods."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x * y, (1, -1))
        pce.fit(samples, values)

        # Compare function outputs to method outputs
        bkd.assert_allclose(pce_statistics.mean(pce), pce.mean())
        bkd.assert_allclose(pce_statistics.variance(pce), pce.variance())
        bkd.assert_allclose(pce_statistics.std(pce), pce.std())
        bkd.assert_allclose(
            pce_statistics.total_sobol_indices(pce), pce.total_sobol_indices()
        )
        bkd.assert_allclose(
            pce_statistics.main_effect_sobol_indices(pce),
            pce.main_effect_sobol_indices(),
        )
