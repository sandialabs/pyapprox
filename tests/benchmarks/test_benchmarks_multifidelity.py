"""Tests for multifidelity benchmark functions and instances."""

# TODO: this test class should be where function is defined
# not at this level which is for integration tests.


import pytest

from pyapprox.benchmarks.functions.multifidelity.multioutput_ensemble import (
    MultiOutputModelEnsemble,
    PSDMultiOutputModelEnsemble,
)
from pyapprox.benchmarks.functions.multifidelity.polynomial_ensemble import (
    PolynomialEnsemble,
    PolynomialModelFunction,
)
from pyapprox.benchmarks.functions.multifidelity.statistics_mixin import (
    MultifidelityStatisticsMixin,
)
from pyapprox.benchmarks.functions.multifidelity.tunable_ensemble import (
    TunableModelEnsemble,
)
from pyapprox.benchmarks.instances.multifidelity import (
    multioutput_ensemble_3x3,
    polynomial_ensemble_3model,
    polynomial_ensemble_5model,
    psd_multioutput_ensemble_3x3,
    tunable_ensemble_3model,
)
from pyapprox.benchmarks.protocols import (
    HasEnsembleCovariance,
    HasEnsembleMeans,
    HasEnsembleModels,
    HasEstimatedEvaluationCost,
    HasModelCosts,
    HasPrior,
    HasSmoothness,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)


class TestPolynomialModelFunction:
    """Tests for PolynomialModelFunction."""

    def test_protocol_compliance(self, bkd) -> None:
        """Test that PolynomialModelFunction satisfies FunctionProtocol."""
        func = PolynomialModelFunction(bkd, degree=3)
        assert isinstance(func, FunctionProtocol)

    def test_nvars(self, bkd) -> None:
        """Test nvars returns 1."""
        func = PolynomialModelFunction(bkd, degree=3)
        assert func.nvars() == 1

    def test_nqoi(self, bkd) -> None:
        """Test nqoi returns 1."""
        func = PolynomialModelFunction(bkd, degree=3)
        assert func.nqoi() == 1

    def test_degree(self, bkd) -> None:
        """Test degree returns correct value."""
        func = PolynomialModelFunction(bkd, degree=4)
        assert func.degree() == 4

    def test_evaluation_single(self, bkd) -> None:
        """Test evaluation at single sample."""
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5]])
        result = func(sample)
        expected = bkd.array([[0.5**3]])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_evaluation_batch(self, bkd) -> None:
        """Test batch evaluation."""
        func = PolynomialModelFunction(bkd, degree=2)
        samples = bkd.array([[0.0, 0.5, 1.0]])
        result = func(samples)
        expected = bkd.array([[0.0, 0.25, 1.0]])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_mean_analytical(self, bkd) -> None:
        """Test analytical mean."""
        func = PolynomialModelFunction(bkd, degree=3)
        # E[x^3] = 1/4 for U[0,1]
        bkd.assert_allclose(
            bkd.asarray([func.mean()]),
            bkd.asarray([0.25]),
            rtol=1e-12,
        )

    def test_variance_analytical(self, bkd) -> None:
        """Test analytical variance."""
        func = PolynomialModelFunction(bkd, degree=2)
        # Var[x^2] = 1/5 - 1/9 = 4/45
        expected_var = 1.0 / 5 - 1.0 / 9
        bkd.assert_allclose(
            bkd.asarray([func.variance()]),
            bkd.asarray([expected_var]),
            rtol=1e-12,
        )

    def test_jacobian_shape(self, bkd) -> None:
        """Test Jacobian has correct shape."""
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5]])
        jac = func.jacobian(sample)
        assert jac.shape == (1, 1)

    def test_jacobian_value(self, bkd) -> None:
        """Test Jacobian has correct value."""
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5]])
        jac = func.jacobian(sample)
        # d/dx(x^3) = 3x^2 = 3*0.25 = 0.75
        expected = bkd.array([[0.75]])
        bkd.assert_allclose(jac, expected, rtol=1e-12)

    def test_jacobian_invalid_shape(self, bkd) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5, 0.6]])
        with pytest.raises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self, bkd) -> None:
        """Test HVP has correct shape."""
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5]])
        vec = bkd.array([[1.0]])
        hvp = func.hvp(sample, vec)
        assert hvp.shape == (1, 1)

    def test_hvp_value(self, bkd) -> None:
        """Test HVP has correct value."""
        func = PolynomialModelFunction(bkd, degree=3)
        sample = bkd.array([[0.5]])
        vec = bkd.array([[1.0]])
        hvp = func.hvp(sample, vec)
        # d^2/dx^2(x^3) = 6x = 3.0
        expected = bkd.array([[3.0]])
        bkd.assert_allclose(hvp, expected, rtol=1e-12)

    def test_hvp_degree_1(self, bkd) -> None:
        """Test HVP is zero for degree 1 polynomial."""
        func = PolynomialModelFunction(bkd, degree=1)
        sample = bkd.array([[0.5]])
        vec = bkd.array([[1.0]])
        hvp = func.hvp(sample, vec)
        expected = bkd.array([[0.0]])
        bkd.assert_allclose(hvp, expected, atol=1e-12)


class TestPolynomialEnsemble:
    """Tests for PolynomialEnsemble."""

    def test_nmodels(self, bkd) -> None:
        """Test nmodels returns correct value."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        assert ensemble.nmodels() == 5

    def test_nvars(self, bkd) -> None:
        """Test nvars returns 1."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        assert ensemble.nvars() == 1

    def test_nqoi(self, bkd) -> None:
        """Test nqoi returns 1."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        assert ensemble.nqoi() == 1

    def test_model_degrees(self, bkd) -> None:
        """Test models have correct degrees."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        models = ensemble.models()
        # Degrees should be 5, 4, 3, 2, 1
        assert models[0].degree() == 5
        assert models[1].degree() == 4
        assert models[2].degree() == 3
        assert models[3].degree() == 2
        assert models[4].degree() == 1

    def test_getitem(self, bkd) -> None:
        """Test __getitem__ access."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        assert ensemble[0].degree() == 5
        assert ensemble[4].degree() == 1

    def test_costs_shape(self, bkd) -> None:
        """Test costs has correct shape."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        costs = ensemble.costs()
        assert costs.shape == (5,)

    def test_costs_decreasing(self, bkd) -> None:
        """Test costs are decreasing."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        costs = ensemble.costs()
        for i in range(4):
            assert costs[i] > costs[i + 1]

    def test_means_shape(self, bkd) -> None:
        """Test means has correct shape."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        means = ensemble.means()
        assert means.shape == (5,)

    def test_means_values(self, bkd) -> None:
        """Test means have correct values."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        means = ensemble.means()
        # E[x^k] = 1/(k+1), degrees are 5, 4, 3, 2, 1
        expected = bkd.array([1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2])
        bkd.assert_allclose(means, expected, rtol=1e-12)

    def test_variances_shape(self, bkd) -> None:
        """Test variances has correct shape."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        variances = ensemble.variances()
        assert variances.shape == (5,)

    def test_covariance_matrix_shape(self, bkd) -> None:
        """Test covariance matrix has correct shape."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        cov = ensemble.covariance_matrix()
        assert cov.shape == (5, 5)

    def test_covariance_matrix_symmetric(self, bkd) -> None:
        """Test covariance matrix is symmetric."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        cov = ensemble.covariance_matrix()
        bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_covariance_matrix_positive_diagonal(self, bkd) -> None:
        """Test covariance matrix has positive diagonal."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        cov = ensemble.covariance_matrix()
        for i in range(5):
            assert cov[i, i] > 0

    def test_correlation_matrix_diagonal_ones(self, bkd) -> None:
        """Test correlation matrix has ones on diagonal."""
        ensemble = PolynomialEnsemble(bkd, nmodels=5)
        corr = ensemble.correlation_matrix()
        for i in range(5):
            bkd.assert_allclose(
                bkd.asarray([corr[i, i]]),
                bkd.asarray([1.0]),
                rtol=1e-12,
            )


class TestMultifidelityBenchmarkInstances:
    """Tests for multifidelity benchmark instances."""

    def test_5model_name(self, bkd) -> None:
        """Test 5-model benchmark name."""
        benchmark = polynomial_ensemble_5model(bkd)
        assert benchmark.name() == "polynomial_ensemble_5model"

    def test_5model_nmodels(self, bkd) -> None:
        """Test 5-model benchmark has 5 models."""
        benchmark = polynomial_ensemble_5model(bkd)
        assert benchmark.nmodels() == 5

    def test_5model_ensemble_means(self, bkd) -> None:
        """Test 5-model ensemble means shape and high-fidelity value."""
        benchmark = polynomial_ensemble_5model(bkd)
        means = benchmark.ensemble_means()  # (nmodels, 1)
        assert means.shape == (5, 1)
        # High fidelity mean is E[x^5] = 1/6
        bkd.assert_allclose(
            means[0:1, 0:1],
            bkd.asarray([[1 / 6]]),
            rtol=1e-12,
        )

    def test_5model_ensemble_covariance(self, bkd) -> None:
        """Test 5-model ensemble covariance shape and symmetry."""
        benchmark = polynomial_ensemble_5model(bkd)
        cov = benchmark.ensemble_covariance()  # (nmodels, nmodels)
        assert cov.shape == (5, 5)
        bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_3model_name(self, bkd) -> None:
        """Test 3-model benchmark name."""
        benchmark = polynomial_ensemble_3model(bkd)
        assert benchmark.name() == "polynomial_ensemble_3model"

    def test_3model_nmodels(self, bkd) -> None:
        """Test 3-model benchmark has 3 models."""
        benchmark = polynomial_ensemble_3model(bkd)
        assert benchmark.nmodels() == 3

    def test_domain_bounds(self, bkd) -> None:
        """Test domain bounds are [0, 1]."""
        benchmark = polynomial_ensemble_5model(bkd)
        domain = benchmark.domain()
        expected = bkd.array([[0.0, 1.0]])
        bkd.assert_allclose(domain.bounds(), expected, rtol=1e-12)


class TestMultiOutputModelEnsemble:
    """Tests for MultiOutputModelEnsemble and statistics mixin."""

    def test_nmodels(self, bkd) -> None:
        """Test nmodels returns correct value."""
        ensemble = MultiOutputModelEnsemble(bkd)
        assert ensemble.nmodels() == 3

    def test_nqoi(self, bkd) -> None:
        """Test nqoi returns correct value."""
        ensemble = MultiOutputModelEnsemble(bkd)
        assert ensemble.nqoi() == 3

    def test_nvars(self, bkd) -> None:
        """Test nvars returns 1."""
        ensemble = MultiOutputModelEnsemble(bkd)
        assert ensemble.nvars() == 1

    def test_costs_shape(self, bkd) -> None:
        """Test costs has correct shape."""
        ensemble = MultiOutputModelEnsemble(bkd)
        costs = ensemble.costs()
        assert costs.shape == (3,)

    def test_means_shape(self, bkd) -> None:
        """Test means has correct shape."""
        ensemble = MultiOutputModelEnsemble(bkd)
        means = ensemble.means()
        assert means.shape == (3, 3)

    def test_covariance_matrix_shape(self, bkd) -> None:
        """Test covariance matrix has correct shape."""
        ensemble = MultiOutputModelEnsemble(bkd)
        cov = ensemble.covariance_matrix()
        assert cov.shape == (9, 9)

    def test_covariance_matrix_symmetric(self, bkd) -> None:
        """Test covariance matrix is symmetric."""
        ensemble = MultiOutputModelEnsemble(bkd)
        cov = ensemble.covariance_matrix()
        bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_analytical_vs_numerical_covariance(self, bkd) -> None:
        """Test analytical covariance matches numerical quadrature."""
        ensemble = MultiOutputModelEnsemble(bkd)
        # Get analytical covariance (from override)
        analytical_cov = ensemble.covariance_matrix()
        # Get numerical covariance by calling mixin method directly
        numerical_cov = MultifidelityStatisticsMixin.covariance_matrix(ensemble)
        # Should match to high precision
        bkd.assert_allclose(analytical_cov, numerical_cov, rtol=1e-6)

    def test_analytical_vs_numerical_means(self, bkd) -> None:
        """Test analytical means match numerical quadrature."""
        ensemble = MultiOutputModelEnsemble(bkd)
        # Get analytical means (from override)
        analytical_means = ensemble.means()
        # Get numerical means by calling mixin method directly
        numerical_means = MultifidelityStatisticsMixin.means(ensemble)
        # Should match to high precision
        # atol needed for values that are analytically 0 but numerically ~1e-16
        bkd.assert_allclose(
            analytical_means, numerical_means, rtol=1e-6, atol=1e-14
        )

    def test_kronecker_product_covariance_shape(self, bkd) -> None:
        """Test W matrix has correct shape."""
        ensemble = MultiOutputModelEnsemble(bkd)
        W = ensemble.covariance_of_centered_values_kronecker_product()
        # Shape is (nmodels * nqoi^2, nmodels * nqoi^2) = (27, 27)
        assert W.shape == (27, 27)

    def test_kronecker_product_covariance_symmetric(self, bkd) -> None:
        """Test W matrix is symmetric."""
        ensemble = MultiOutputModelEnsemble(bkd)
        W = ensemble.covariance_of_centered_values_kronecker_product()
        bkd.assert_allclose(W, W.T, rtol=1e-10)

    def test_mean_variance_covariance_shape(self, bkd) -> None:
        """Test B matrix has correct shape."""
        ensemble = MultiOutputModelEnsemble(bkd)
        B = ensemble.covariance_of_mean_and_variance_estimators()
        # Shape is (nmodels * nqoi, nmodels * nqoi^2) = (9, 27)
        assert B.shape == (9, 27)

    def test_covariance_subproblem_single_model_single_qoi(self, bkd) -> None:
        """Test covariance subproblem for single model and QoI."""
        ensemble = MultiOutputModelEnsemble(bkd)
        full_cov = ensemble.covariance_matrix()
        # Get subproblem for model 0, qoi 0
        sub_cov = ensemble.covariance_subproblem([0], [0])
        assert sub_cov.shape == (1, 1)
        bkd.assert_allclose(sub_cov[0, 0], full_cov[0, 0], rtol=1e-12)

    def test_covariance_subproblem_multiple(self, bkd) -> None:
        """Test covariance subproblem for multiple models and QoI."""
        ensemble = MultiOutputModelEnsemble(bkd)
        full_cov = ensemble.covariance_matrix()
        # Get subproblem for models [0, 1] and qoi [0, 2]
        model_idx = [0, 1]
        qoi_idx = [0, 2]
        sub_cov = ensemble.covariance_subproblem(model_idx, qoi_idx)
        # Shape should be (2*2, 2*2) = (4, 4)
        assert sub_cov.shape == (4, 4)
        # Check a few entries
        # (0, 0) -> model 0 qoi 0 vs model 0 qoi 0 -> full[0, 0]
        bkd.assert_allclose(sub_cov[0, 0], full_cov[0, 0], rtol=1e-12)
        # (1, 1) -> model 0 qoi 2 vs model 0 qoi 2 -> full[2, 2]
        bkd.assert_allclose(sub_cov[1, 1], full_cov[2, 2], rtol=1e-12)
        # (2, 2) -> model 1 qoi 0 vs model 1 qoi 0 -> full[3, 3]
        bkd.assert_allclose(sub_cov[2, 2], full_cov[3, 3], rtol=1e-12)

    def test_kronecker_subproblem_matches_full(self, bkd) -> None:
        """Test W subproblem extraction matches direct computation."""
        ensemble = MultiOutputModelEnsemble(bkd)
        model_idx = [0, 1]
        qoi_idx = [0, 2]
        # Get subproblem from full W matrix
        W_sub = ensemble.covariance_of_centered_values_kronecker_product_subproblem(
            model_idx, qoi_idx
        )
        # Shape: (nsub_models * nsub_qoi^2, nsub_models * nsub_qoi^2) = (2*4, 2*4) = (8,
        # 8)
        assert W_sub.shape == (8, 8)
        # Verify symmetry
        bkd.assert_allclose(W_sub, W_sub.T, rtol=1e-10)

    def test_mean_variance_subproblem_matches_full(self, bkd) -> None:
        """Test B subproblem extraction matches direct computation."""
        ensemble = MultiOutputModelEnsemble(bkd)
        model_idx = [0, 1]
        qoi_idx = [0, 2]
        # Get subproblem from full B matrix
        B_sub = ensemble.covariance_of_mean_and_variance_estimators_subproblem(
            model_idx, qoi_idx
        )
        # Shape: (nsub_models * nsub_qoi, nsub_models * nsub_qoi^2) = (2*2, 2*4) = (4,
        # 8)
        assert B_sub.shape == (4, 8)


class TestPSDMultiOutputModelEnsemble:
    """Tests for PSD variant of MultiOutputModelEnsemble."""

    def test_covariance_matrix_shape(self, bkd) -> None:
        """Test covariance matrix has correct shape."""
        ensemble = PSDMultiOutputModelEnsemble(bkd)
        cov = ensemble.covariance_matrix()
        assert cov.shape == (9, 9)

    def test_covariance_matrix_symmetric(self, bkd) -> None:
        """Test covariance matrix is symmetric."""
        ensemble = PSDMultiOutputModelEnsemble(bkd)
        cov = ensemble.covariance_matrix()
        bkd.assert_allclose(cov, cov.T, rtol=1e-10)

    def test_covariance_matrix_positive_diagonal(self, bkd) -> None:
        """Test covariance matrix has positive diagonal."""
        ensemble = PSDMultiOutputModelEnsemble(bkd)
        cov = ensemble.covariance_matrix()
        for i in range(9):
            assert cov[i, i] > 0

    def test_means_shape(self, bkd) -> None:
        """Test means has correct shape."""
        ensemble = PSDMultiOutputModelEnsemble(bkd)
        means = ensemble.means()
        assert means.shape == (3, 3)

    def test_kronecker_product_covariance_shape(self, bkd) -> None:
        """Test W matrix has correct shape."""
        ensemble = PSDMultiOutputModelEnsemble(bkd)
        W = ensemble.covariance_of_centered_values_kronecker_product()
        assert W.shape == (27, 27)

    def test_mean_variance_covariance_shape(self, bkd) -> None:
        """Test B matrix has correct shape."""
        ensemble = PSDMultiOutputModelEnsemble(bkd)
        B = ensemble.covariance_of_mean_and_variance_estimators()
        assert B.shape == (9, 27)

    def test_different_from_regular_ensemble(self, bkd) -> None:
        """Test PSD ensemble has different covariance from regular."""
        regular = MultiOutputModelEnsemble(bkd)
        psd = PSDMultiOutputModelEnsemble(bkd)

        regular_cov = regular.covariance_matrix()
        psd_cov = psd.covariance_matrix()

        # They should be different (PSD has perturbations)
        diff = bkd.max(bkd.abs(regular_cov - psd_cov))
        assert bkd.to_numpy(diff) > 0.01


class TestTunableModelEnsemble:
    """Tests for TunableModelEnsemble."""

    def test_nmodels(self, bkd) -> None:
        """Test nmodels returns 3."""
        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        assert ensemble.nmodels() == 3

    def test_nvars(self, bkd) -> None:
        """Test nvars returns 2."""
        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        assert ensemble.nvars() == 2

    def test_nqoi(self, bkd) -> None:
        """Test nqoi returns 1."""
        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        assert ensemble.nqoi() == 1

    def test_costs_shape(self, bkd) -> None:
        """Test costs has correct shape."""
        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        costs = ensemble.costs()
        assert costs.shape == (3,)

    def test_means_shape(self, bkd) -> None:
        """Test means has correct shape."""
        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        means = ensemble.means()
        assert means.shape == (3, 1)

    def test_means_zero_no_shifts(self, bkd) -> None:
        """Test means are zero when no shifts applied."""
        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        means = ensemble.means()
        bkd.assert_allclose(means, bkd.zeros((3, 1)), atol=1e-14)

    def test_covariance_shape(self, bkd) -> None:
        """Test covariance has correct shape."""
        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        cov = ensemble.covariance()
        assert cov.shape == (3, 3)

    def test_covariance_symmetric(self, bkd) -> None:
        """Test covariance matrix is symmetric."""
        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        cov = ensemble.covariance()
        bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_covariance_unit_diagonal(self, bkd) -> None:
        """Test covariance has unit diagonal (models have unit variance)."""
        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        cov = ensemble.covariance()
        for i in range(3):
            bkd.assert_allclose(
                bkd.asarray([cov[i, i]]),
                bkd.asarray([1.0]),
                rtol=1e-12,
            )

    def test_model_evaluation(self, bkd) -> None:
        """Test model evaluation shape."""
        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        samples = bkd.array([[0.5, -0.3], [0.1, 0.7]])  # (2, 2)
        for model in ensemble.models():
            result = model(samples)
            assert result.shape == (1, 2)

    def test_covariance_vs_numerical(self, bkd) -> None:
        """Test analytical covariance matches numerical estimate."""
        import numpy as np

        ensemble = TunableModelEnsemble(theta1=1.0, bkd=bkd)
        np.random.seed(42)
        nsamples = 100000
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        vals = []
        for model in ensemble.models():
            vals.append(model(samples)[0, :])  # (nsamples,)
        vals_array = bkd.vstack(vals)  # (3, nsamples)
        numerical_cov = bkd.zeros((3, 3))
        means = bkd.mean(vals_array, axis=1)
        for i in range(3):
            for j in range(3):
                numerical_cov[i, j] = bkd.mean(
                    (vals_array[i, :] - means[i]) * (vals_array[j, :] - means[j])
                )
        bkd.assert_allclose(ensemble.covariance(), numerical_cov, rtol=5e-2)


class TestMultiOutputEnsembleInstance:
    """Tests for multioutput_ensemble_3x3 benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        bm = multioutput_ensemble_3x3(bkd)
        assert bm.name() == "multioutput_ensemble_3x3"

    def test_nmodels(self, bkd) -> None:
        """Test nmodels returns 3."""
        bm = multioutput_ensemble_3x3(bkd)
        assert bm.nmodels() == 3

    def test_models_count(self, bkd) -> None:
        """Test models list has 3 entries."""
        bm = multioutput_ensemble_3x3(bkd)
        assert len(bm.models()) == 3

    def test_ensemble_means_shape(self, bkd) -> None:
        """Test ensemble_means returns (nmodels, nqoi) = (3, 3)."""
        bm = multioutput_ensemble_3x3(bkd)
        means = bm.ensemble_means()
        assert means.shape == (3, 3)

    def test_ensemble_covariance_shape(self, bkd) -> None:
        """Test ensemble_covariance returns (9, 9) block structure."""
        bm = multioutput_ensemble_3x3(bkd)
        cov = bm.ensemble_covariance()
        assert cov.shape == (9, 9)

    def test_ensemble_covariance_symmetric(self, bkd) -> None:
        """Test covariance matrix is symmetric."""
        bm = multioutput_ensemble_3x3(bkd)
        cov = bm.ensemble_covariance()
        bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_costs_shape(self, bkd) -> None:
        """Test costs shape."""
        bm = multioutput_ensemble_3x3(bkd)
        costs = bm.costs()
        assert costs.shape == (3,)

    def test_domain_bounds(self, bkd) -> None:
        """Test domain bounds are [0, 1]."""
        bm = multioutput_ensemble_3x3(bkd)
        expected = bkd.array([[0.0, 1.0]])
        bkd.assert_allclose(bm.domain().bounds(), expected, rtol=1e-12)

    def test_protocol_compliance(self, bkd) -> None:
        """Test benchmark satisfies expected protocols."""
        bm = multioutput_ensemble_3x3(bkd)
        assert isinstance(bm, HasEnsembleModels)
        assert isinstance(bm, HasModelCosts)
        assert isinstance(bm, HasPrior)
        assert isinstance(bm, HasEnsembleMeans)
        assert isinstance(bm, HasEnsembleCovariance)
        assert isinstance(bm, HasSmoothness)
        assert isinstance(bm, HasEstimatedEvaluationCost)


class TestPSDMultiOutputEnsembleInstance:
    """Tests for psd_multioutput_ensemble_3x3 benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        bm = psd_multioutput_ensemble_3x3(bkd)
        assert bm.name() == "psd_multioutput_ensemble_3x3"

    def test_nmodels(self, bkd) -> None:
        """Test nmodels returns 3."""
        bm = psd_multioutput_ensemble_3x3(bkd)
        assert bm.nmodels() == 3

    def test_ensemble_means_shape(self, bkd) -> None:
        """Test ensemble_means returns (3, 3)."""
        bm = psd_multioutput_ensemble_3x3(bkd)
        means = bm.ensemble_means()
        assert means.shape == (3, 3)

    def test_ensemble_covariance_shape(self, bkd) -> None:
        """Test ensemble_covariance returns (9, 9)."""
        bm = psd_multioutput_ensemble_3x3(bkd)
        cov = bm.ensemble_covariance()
        assert cov.shape == (9, 9)

    def test_ensemble_covariance_symmetric(self, bkd) -> None:
        """Test covariance matrix is symmetric."""
        bm = psd_multioutput_ensemble_3x3(bkd)
        cov = bm.ensemble_covariance()
        bkd.assert_allclose(cov, cov.T, rtol=1e-10)

    def test_protocol_compliance(self, bkd) -> None:
        """Test benchmark satisfies expected protocols."""
        bm = psd_multioutput_ensemble_3x3(bkd)
        assert isinstance(bm, HasEnsembleModels)
        assert isinstance(bm, HasEnsembleMeans)
        assert isinstance(bm, HasEnsembleCovariance)


class TestTunableEnsembleInstance:
    """Tests for tunable_ensemble_3model benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        bm = tunable_ensemble_3model(bkd)
        assert bm.name() == "tunable_ensemble_3model"

    def test_nmodels(self, bkd) -> None:
        """Test nmodels returns 3."""
        bm = tunable_ensemble_3model(bkd)
        assert bm.nmodels() == 3

    def test_models_count(self, bkd) -> None:
        """Test models list has 3 entries."""
        bm = tunable_ensemble_3model(bkd)
        assert len(bm.models()) == 3

    def test_ensemble_means_shape(self, bkd) -> None:
        """Test ensemble_means returns (3, 1)."""
        bm = tunable_ensemble_3model(bkd)
        means = bm.ensemble_means()
        assert means.shape == (3, 1)

    def test_ensemble_covariance_shape(self, bkd) -> None:
        """Test ensemble_covariance returns (3, 3)."""
        bm = tunable_ensemble_3model(bkd)
        cov = bm.ensemble_covariance()
        assert cov.shape == (3, 3)

    def test_ensemble_covariance_symmetric(self, bkd) -> None:
        """Test covariance matrix is symmetric."""
        bm = tunable_ensemble_3model(bkd)
        cov = bm.ensemble_covariance()
        bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_domain_bounds(self, bkd) -> None:
        """Test domain bounds are [-1, 1]^2."""
        bm = tunable_ensemble_3model(bkd)
        expected = bkd.array([[-1.0, 1.0], [-1.0, 1.0]])
        bkd.assert_allclose(bm.domain().bounds(), expected, rtol=1e-12)

    def test_costs_shape(self, bkd) -> None:
        """Test costs shape."""
        bm = tunable_ensemble_3model(bkd)
        costs = bm.costs()
        assert costs.shape == (3,)

    def test_protocol_compliance(self, bkd) -> None:
        """Test benchmark satisfies expected protocols."""
        bm = tunable_ensemble_3model(bkd)
        assert isinstance(bm, HasEnsembleModels)
        assert isinstance(bm, HasModelCosts)
        assert isinstance(bm, HasPrior)
        assert isinstance(bm, HasEnsembleMeans)
        assert isinstance(bm, HasEnsembleCovariance)
        assert isinstance(bm, HasSmoothness)
        assert isinstance(bm, HasEstimatedEvaluationCost)


class TestBenchmarkRegistryMultifidelity:
    """Tests for BenchmarkRegistry multifidelity category."""

    def test_5model_registered(self) -> None:
        """Test polynomial_ensemble_5model is registered."""
        assert "polynomial_ensemble_5model" in BenchmarkRegistry.list_all()

    def test_3model_registered(self) -> None:
        """Test polynomial_ensemble_3model is registered."""
        assert "polynomial_ensemble_3model" in BenchmarkRegistry.list_all()

    def test_multioutput_registered(self) -> None:
        """Test multioutput_ensemble_3x3 is registered."""
        assert "multioutput_ensemble_3x3" in BenchmarkRegistry.list_all()

    def test_psd_multioutput_registered(self) -> None:
        """Test psd_multioutput_ensemble_3x3 is registered."""
        assert "psd_multioutput_ensemble_3x3" in BenchmarkRegistry.list_all()

    def test_tunable_registered(self) -> None:
        """Test tunable_ensemble_3model is registered."""
        assert "tunable_ensemble_3model" in BenchmarkRegistry.list_all()

    def test_multifidelity_category(self) -> None:
        """Test benchmarks are in multifidelity category."""
        mf_benchmarks = BenchmarkRegistry.list_category("multifidelity")
        assert "polynomial_ensemble_5model" in mf_benchmarks
        assert "polynomial_ensemble_3model" in mf_benchmarks
        assert "multioutput_ensemble_3x3" in mf_benchmarks
        assert "psd_multioutput_ensemble_3x3" in mf_benchmarks
        assert "tunable_ensemble_3model" in mf_benchmarks
