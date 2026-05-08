"""Tests for multifidelity benchmark functions and benchmarks."""

import pytest

from pyapprox_benchmarks.functions.multifidelity.polynomial_ensemble import (
    PolynomialModelFunction,
)
from pyapprox_benchmarks.statest.statistics_mixin import (
    MultifidelityStatisticsMixin,
)
from pyapprox_benchmarks.statest import (
    BraninEnsembleProblem,
    ForresterEnsembleProblem,
    MultiOutputEnsembleBenchmark,
    PolynomialEnsembleBenchmark,
    TunableEnsembleBenchmark,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)


class TestPolynomialModelFunction:
    """Tests for PolynomialModelFunction."""

    def test_protocol_compliance(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=3)
        assert isinstance(func, FunctionProtocol)

    def test_nvars(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=3)
        assert func.nvars() == 1

    def test_nqoi(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=3)
        assert func.nqoi() == 1

    def test_degree(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=4)
        assert func.degree() == 4

    def test_evaluation_single(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5]])
        result = func(sample)
        expected = bkd.array([[0.5**3]])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_evaluation_batch(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=2)
        samples = bkd.array([[0.0, 0.5, 1.0]])
        result = func(samples)
        expected = bkd.array([[0.0, 0.25, 1.0]])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_mean_analytical(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=3)
        bkd.assert_allclose(
            bkd.asarray([func.mean()]),
            bkd.asarray([0.25]),
            rtol=1e-12,
        )

    def test_variance_analytical(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=2)
        expected_var = 1.0 / 5 - 1.0 / 9
        bkd.assert_allclose(
            bkd.asarray([func.variance()]),
            bkd.asarray([expected_var]),
            rtol=1e-12,
        )

    def test_jacobian_shape(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5]])
        jac = func.jacobian(sample)
        assert jac.shape == (1, 1)

    def test_jacobian_value(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5]])
        jac = func.jacobian(sample)
        expected = bkd.array([[0.75]])
        bkd.assert_allclose(jac, expected, rtol=1e-12)

    def test_jacobian_invalid_shape(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5, 0.6]])
        with pytest.raises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5]])
        vec = bkd.array([[1.0]])
        hvp = func.hvp(sample, vec)
        assert hvp.shape == (1, 1)

    def test_hvp_value(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5]])
        vec = bkd.array([[1.0]])
        hvp = func.hvp(sample, vec)
        expected = bkd.array([[3.0]])
        bkd.assert_allclose(hvp, expected, rtol=1e-12)

    def test_hvp_degree_1(self, bkd) -> None:
        func = PolynomialModelFunction(bkd, degree=1)
        sample = bkd.array([[0.5]])
        vec = bkd.array([[1.0]])
        hvp = func.hvp(sample, vec)
        expected = bkd.array([[0.0]])
        bkd.assert_allclose(hvp, expected, atol=1e-12)


class TestPolynomialEnsembleBenchmark:
    """Tests for PolynomialEnsembleBenchmark."""

    def test_nmodels(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=5)
        assert bm.problem().nmodels() == 5

    def test_costs_shape(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=5)
        costs = bm.problem().costs()
        assert costs.shape == (5,)

    def test_costs_decreasing(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=5)
        costs = bm.problem().costs()
        for i in range(4):
            assert costs[i] > costs[i + 1]

    def test_means_shape(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=5)
        means = bm.ensemble_means()
        assert means.shape == (5, 1)

    def test_means_values(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=5)
        means = bm.ensemble_means()
        expected = bkd.array([[1 / 6], [1 / 5], [1 / 4], [1 / 3], [1 / 2]])
        bkd.assert_allclose(means, expected, rtol=1e-12)

    def test_covariance_matrix_shape(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=5)
        cov = bm.ensemble_covariance()
        assert cov.shape == (5, 5)

    def test_covariance_matrix_symmetric(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=5)
        cov = bm.ensemble_covariance()
        bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_covariance_matrix_positive_diagonal(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=5)
        cov = bm.ensemble_covariance()
        for i in range(5):
            assert cov[i, i] > 0

    def test_3model(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=3)
        assert bm.problem().nmodels() == 3
        assert bm.ensemble_means().shape == (3, 1)
        assert bm.ensemble_covariance().shape == (3, 3)

    def test_models_are_functions(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=5)
        models = bm.problem().models()
        assert len(models) == 5
        sample = bkd.array([[0.5]])
        for m in models:
            result = m(sample)
            assert result.shape == (1, 1)

    def test_prior_rvs(self, bkd) -> None:
        bm = PolynomialEnsembleBenchmark(bkd, nmodels=5)
        samples = bm.problem().prior().rvs(10)
        assert samples.shape == (1, 10)


class TestMultiOutputEnsembleBenchmark:
    """Tests for MultiOutputEnsembleBenchmark and statistics mixin."""

    def test_nmodels(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        assert bm.problem().nmodels() == 3

    def test_costs_shape(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        costs = bm.problem().costs()
        assert costs.shape == (3,)

    def test_means_shape(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        means = bm.ensemble_means()
        assert means.shape == (3, 3)

    def test_covariance_matrix_shape(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        cov = bm.ensemble_covariance()
        assert cov.shape == (9, 9)

    def test_covariance_matrix_symmetric(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        cov = bm.ensemble_covariance()
        bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_analytical_vs_numerical_covariance(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        analytical_cov = bm.ensemble_covariance()
        numerical_cov = MultifidelityStatisticsMixin.covariance_matrix(bm)
        bkd.assert_allclose(analytical_cov, numerical_cov, rtol=1e-6)

    def test_analytical_vs_numerical_means(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        analytical_means = bm.ensemble_means()
        numerical_means = MultifidelityStatisticsMixin.means(bm)
        bkd.assert_allclose(
            analytical_means, numerical_means, rtol=1e-6, atol=1e-14
        )

    def test_kronecker_product_covariance_shape(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        W = bm.covariance_of_centered_values_kronecker_product()
        assert W.shape == (27, 27)

    def test_kronecker_product_covariance_symmetric(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        W = bm.covariance_of_centered_values_kronecker_product()
        bkd.assert_allclose(W, W.T, rtol=1e-10)

    def test_mean_variance_covariance_shape(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        B = bm.covariance_of_mean_and_variance_estimators()
        assert B.shape == (9, 27)

    def test_covariance_subproblem_single_model_single_qoi(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        full_cov = bm.ensemble_covariance()
        sub_cov = bm.covariance_subproblem([0], [0])
        assert sub_cov.shape == (1, 1)
        bkd.assert_allclose(sub_cov[0, 0], full_cov[0, 0], rtol=1e-12)

    def test_covariance_subproblem_multiple(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        full_cov = bm.ensemble_covariance()
        model_idx = [0, 1]
        qoi_idx = [0, 2]
        sub_cov = bm.covariance_subproblem(model_idx, qoi_idx)
        assert sub_cov.shape == (4, 4)
        bkd.assert_allclose(sub_cov[0, 0], full_cov[0, 0], rtol=1e-12)
        bkd.assert_allclose(sub_cov[1, 1], full_cov[2, 2], rtol=1e-12)
        bkd.assert_allclose(sub_cov[2, 2], full_cov[3, 3], rtol=1e-12)

    def test_kronecker_subproblem_matches_full(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        model_idx = [0, 1]
        qoi_idx = [0, 2]
        W_sub = bm.covariance_of_centered_values_kronecker_product_subproblem(
            model_idx, qoi_idx
        )
        assert W_sub.shape == (8, 8)
        bkd.assert_allclose(W_sub, W_sub.T, rtol=1e-10)

    def test_mean_variance_subproblem_matches_full(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd)
        model_idx = [0, 1]
        qoi_idx = [0, 2]
        B_sub = bm.covariance_of_mean_and_variance_estimators_subproblem(
            model_idx, qoi_idx
        )
        assert B_sub.shape == (4, 8)


class TestPSDMultiOutputEnsembleBenchmark:
    """Tests for PSD variant of MultiOutputEnsembleBenchmark."""

    def test_covariance_matrix_shape(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd, psd=True)
        cov = bm.ensemble_covariance()
        assert cov.shape == (9, 9)

    def test_covariance_matrix_symmetric(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd, psd=True)
        cov = bm.ensemble_covariance()
        bkd.assert_allclose(cov, cov.T, rtol=1e-10)

    def test_covariance_matrix_positive_diagonal(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd, psd=True)
        cov = bm.ensemble_covariance()
        for i in range(9):
            assert cov[i, i] > 0

    def test_means_shape(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd, psd=True)
        means = bm.ensemble_means()
        assert means.shape == (3, 3)

    def test_kronecker_product_covariance_shape(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd, psd=True)
        W = bm.covariance_of_centered_values_kronecker_product()
        assert W.shape == (27, 27)

    def test_mean_variance_covariance_shape(self, bkd) -> None:
        bm = MultiOutputEnsembleBenchmark(bkd, psd=True)
        B = bm.covariance_of_mean_and_variance_estimators()
        assert B.shape == (9, 27)

    def test_different_from_regular(self, bkd) -> None:
        regular = MultiOutputEnsembleBenchmark(bkd)
        psd = MultiOutputEnsembleBenchmark(bkd, psd=True)
        regular_cov = regular.ensemble_covariance()
        psd_cov = psd.ensemble_covariance()
        diff = bkd.max(bkd.abs(regular_cov - psd_cov))
        assert bkd.to_numpy(diff) > 0.01


class TestTunableEnsembleBenchmark:
    """Tests for TunableEnsembleBenchmark."""

    def test_nmodels(self, bkd) -> None:
        bm = TunableEnsembleBenchmark(bkd)
        assert bm.problem().nmodels() == 3

    def test_costs_shape(self, bkd) -> None:
        bm = TunableEnsembleBenchmark(bkd)
        costs = bm.problem().costs()
        assert costs.shape == (3,)

    def test_means_shape(self, bkd) -> None:
        bm = TunableEnsembleBenchmark(bkd)
        means = bm.ensemble_means()
        assert means.shape == (3, 1)

    def test_means_zero_no_shifts(self, bkd) -> None:
        bm = TunableEnsembleBenchmark(bkd)
        means = bm.ensemble_means()
        bkd.assert_allclose(means, bkd.zeros((3, 1)), atol=1e-14)

    def test_covariance_shape(self, bkd) -> None:
        bm = TunableEnsembleBenchmark(bkd)
        cov = bm.ensemble_covariance()
        assert cov.shape == (3, 3)

    def test_covariance_symmetric(self, bkd) -> None:
        bm = TunableEnsembleBenchmark(bkd)
        cov = bm.ensemble_covariance()
        bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_covariance_unit_diagonal(self, bkd) -> None:
        bm = TunableEnsembleBenchmark(bkd)
        cov = bm.ensemble_covariance()
        for i in range(3):
            bkd.assert_allclose(
                bkd.asarray([cov[i, i]]),
                bkd.asarray([1.0]),
                rtol=1e-12,
            )

    def test_model_evaluation(self, bkd) -> None:
        bm = TunableEnsembleBenchmark(bkd)
        samples = bkd.array([[0.5, -0.3], [0.1, 0.7]])
        for model in bm.problem().models():
            result = model(samples)
            assert result.shape == (1, 2)

    def test_covariance_vs_numerical(self, bkd) -> None:
        import numpy as np

        bm = TunableEnsembleBenchmark(bkd)
        np.random.seed(42)
        nsamples = 100000
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        vals = []
        for model in bm.problem().models():
            vals.append(model(samples)[0, :])
        vals_array = bkd.vstack(vals)
        numerical_cov = bkd.zeros((3, 3))
        means = bkd.mean(vals_array, axis=1)
        for i in range(3):
            for j in range(3):
                numerical_cov[i, j] = bkd.mean(
                    (vals_array[i, :] - means[i]) * (vals_array[j, :] - means[j])
                )
        bkd.assert_allclose(bm.ensemble_covariance(), numerical_cov, rtol=5e-2)

    def test_domain_bounds(self, bkd) -> None:
        bm = TunableEnsembleBenchmark(bkd)
        expected = bkd.array([[-1.0, 1.0], [-1.0, 1.0]])
        bkd.assert_allclose(bm.domain().bounds(), expected, rtol=1e-12)

    def test_prior_rvs(self, bkd) -> None:
        bm = TunableEnsembleBenchmark(bkd)
        samples = bm.problem().prior().rvs(10)
        assert samples.shape == (2, 10)


class TestBraninEnsembleProblem:
    """Tests for BraninEnsembleProblem."""

    def test_nmodels(self, bkd) -> None:
        bm = BraninEnsembleProblem(bkd)
        assert bm.problem().nmodels() == 3

    def test_models_evaluate(self, bkd) -> None:
        bm = BraninEnsembleProblem(bkd)
        samples = bkd.array([[0.0, 1.0], [5.0, 10.0]])
        for m in bm.problem().models():
            result = m(samples)
            assert result.shape == (1, 2)

    def test_prior_rvs(self, bkd) -> None:
        bm = BraninEnsembleProblem(bkd)
        samples = bm.problem().prior().rvs(10)
        assert samples.shape == (2, 10)


class TestForresterEnsembleProblem:
    """Tests for ForresterEnsembleProblem."""

    def test_nmodels(self, bkd) -> None:
        bm = ForresterEnsembleProblem(bkd)
        assert bm.problem().nmodels() == 2

    def test_models_evaluate(self, bkd) -> None:
        bm = ForresterEnsembleProblem(bkd)
        samples = bkd.array([[0.0, 0.5, 1.0]])
        for m in bm.problem().models():
            result = m(samples)
            assert result.shape == (1, 3)

    def test_prior_rvs(self, bkd) -> None:
        bm = ForresterEnsembleProblem(bkd)
        samples = bm.problem().prior().rvs(10)
        assert samples.shape == (1, 10)
