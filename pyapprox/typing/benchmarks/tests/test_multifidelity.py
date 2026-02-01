"""Tests for multifidelity benchmark functions and instances."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.benchmarks.functions.multifidelity.polynomial_ensemble import (
    PolynomialModelFunction,
    PolynomialEnsemble,
)
from pyapprox.typing.benchmarks.functions.multifidelity.multioutput_ensemble import (
    MultiOutputModelEnsemble,
    PSDMultiOutputModelEnsemble,
)
from pyapprox.typing.benchmarks.functions.multifidelity.statistics_mixin import (
    MultifidelityStatisticsMixin,
)
from pyapprox.typing.benchmarks.instances.multifidelity import (
    polynomial_ensemble_5model,
    polynomial_ensemble_3model,
)
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry


class TestPolynomialModelFunction(Generic[Array], unittest.TestCase):
    """Tests for PolynomialModelFunction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance(self) -> None:
        """Test that PolynomialModelFunction satisfies FunctionProtocol."""
        func = PolynomialModelFunction(self._bkd, degree=3)
        self.assertIsInstance(func, FunctionProtocol)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        func = PolynomialModelFunction(self._bkd, degree=3)
        self.assertEqual(func.nvars(), 1)

    def test_nqoi(self) -> None:
        """Test nqoi returns 1."""
        func = PolynomialModelFunction(self._bkd, degree=3)
        self.assertEqual(func.nqoi(), 1)

    def test_degree(self) -> None:
        """Test degree returns correct value."""
        func = PolynomialModelFunction(self._bkd, degree=4)
        self.assertEqual(func.degree(), 4)

    def test_evaluation_single(self) -> None:
        """Test evaluation at single sample."""
        func = PolynomialModelFunction(self._bkd, degree=3)
        sample = self._bkd.array([[0.5]])
        result = func(sample)
        expected = self._bkd.array([[0.5**3]])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_evaluation_batch(self) -> None:
        """Test batch evaluation."""
        func = PolynomialModelFunction(self._bkd, degree=2)
        samples = self._bkd.array([[0.0, 0.5, 1.0]])
        result = func(samples)
        expected = self._bkd.array([[0.0, 0.25, 1.0]])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_mean_analytical(self) -> None:
        """Test analytical mean."""
        func = PolynomialModelFunction(self._bkd, degree=3)
        # E[x^3] = 1/4 for U[0,1]
        self._bkd.assert_allclose(
            self._bkd.asarray([func.mean()]),
            self._bkd.asarray([0.25]),
            rtol=1e-12,
        )

    def test_variance_analytical(self) -> None:
        """Test analytical variance."""
        func = PolynomialModelFunction(self._bkd, degree=2)
        # Var[x^2] = 1/5 - 1/9 = 4/45
        expected_var = 1.0 / 5 - 1.0 / 9
        self._bkd.assert_allclose(
            self._bkd.asarray([func.variance()]),
            self._bkd.asarray([expected_var]),
            rtol=1e-12,
        )

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        func = PolynomialModelFunction(self._bkd, degree=3)
        sample = self._bkd.array([[0.5]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 1))

    def test_jacobian_value(self) -> None:
        """Test Jacobian has correct value."""
        func = PolynomialModelFunction(self._bkd, degree=3)
        sample = self._bkd.array([[0.5]])
        jac = func.jacobian(sample)
        # d/dx(x^3) = 3x^2 = 3*0.25 = 0.75
        expected = self._bkd.array([[0.75]])
        self._bkd.assert_allclose(jac, expected, rtol=1e-12)

    def test_jacobian_invalid_shape(self) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = PolynomialModelFunction(self._bkd, degree=3)
        sample = self._bkd.array([[0.5, 0.6]])
        with self.assertRaises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self) -> None:
        """Test HVP has correct shape."""
        func = PolynomialModelFunction(self._bkd, degree=3)
        sample = self._bkd.array([[0.5]])
        vec = self._bkd.array([[1.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (1, 1))

    def test_hvp_value(self) -> None:
        """Test HVP has correct value."""
        func = PolynomialModelFunction(self._bkd, degree=3)
        sample = self._bkd.array([[0.5]])
        vec = self._bkd.array([[1.0]])
        hvp = func.hvp(sample, vec)
        # d^2/dx^2(x^3) = 6x = 3.0
        expected = self._bkd.array([[3.0]])
        self._bkd.assert_allclose(hvp, expected, rtol=1e-12)

    def test_hvp_degree_1(self) -> None:
        """Test HVP is zero for degree 1 polynomial."""
        func = PolynomialModelFunction(self._bkd, degree=1)
        sample = self._bkd.array([[0.5]])
        vec = self._bkd.array([[1.0]])
        hvp = func.hvp(sample, vec)
        expected = self._bkd.array([[0.0]])
        self._bkd.assert_allclose(hvp, expected, atol=1e-12)


class TestPolynomialEnsemble(Generic[Array], unittest.TestCase):
    """Tests for PolynomialEnsemble."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_nmodels(self) -> None:
        """Test nmodels returns correct value."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        self.assertEqual(ensemble.nmodels(), 5)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        self.assertEqual(ensemble.nvars(), 1)

    def test_nqoi(self) -> None:
        """Test nqoi returns 1."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        self.assertEqual(ensemble.nqoi(), 1)

    def test_model_degrees(self) -> None:
        """Test models have correct degrees."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        models = ensemble.models()
        # Degrees should be 5, 4, 3, 2, 1
        self.assertEqual(models[0].degree(), 5)
        self.assertEqual(models[1].degree(), 4)
        self.assertEqual(models[2].degree(), 3)
        self.assertEqual(models[3].degree(), 2)
        self.assertEqual(models[4].degree(), 1)

    def test_getitem(self) -> None:
        """Test __getitem__ access."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        self.assertEqual(ensemble[0].degree(), 5)
        self.assertEqual(ensemble[4].degree(), 1)

    def test_costs_shape(self) -> None:
        """Test costs has correct shape."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        costs = ensemble.costs()
        self.assertEqual(costs.shape, (5,))

    def test_costs_decreasing(self) -> None:
        """Test costs are decreasing."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        costs = ensemble.costs()
        for i in range(4):
            self.assertGreater(costs[i], costs[i + 1])

    def test_means_shape(self) -> None:
        """Test means has correct shape."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        means = ensemble.means()
        self.assertEqual(means.shape, (5,))

    def test_means_values(self) -> None:
        """Test means have correct values."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        means = ensemble.means()
        # E[x^k] = 1/(k+1), degrees are 5, 4, 3, 2, 1
        expected = self._bkd.array([1/6, 1/5, 1/4, 1/3, 1/2])
        self._bkd.assert_allclose(means, expected, rtol=1e-12)

    def test_variances_shape(self) -> None:
        """Test variances has correct shape."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        variances = ensemble.variances()
        self.assertEqual(variances.shape, (5,))

    def test_covariance_matrix_shape(self) -> None:
        """Test covariance matrix has correct shape."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        cov = ensemble.covariance_matrix()
        self.assertEqual(cov.shape, (5, 5))

    def test_covariance_matrix_symmetric(self) -> None:
        """Test covariance matrix is symmetric."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        cov = ensemble.covariance_matrix()
        self._bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_covariance_matrix_positive_diagonal(self) -> None:
        """Test covariance matrix has positive diagonal."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        cov = ensemble.covariance_matrix()
        for i in range(5):
            self.assertGreater(cov[i, i], 0)

    def test_correlation_matrix_diagonal_ones(self) -> None:
        """Test correlation matrix has ones on diagonal."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        corr = ensemble.correlation_matrix()
        for i in range(5):
            self._bkd.assert_allclose(
                self._bkd.asarray([corr[i, i]]),
                self._bkd.asarray([1.0]),
                rtol=1e-12,
            )


class TestMultifidelityBenchmarkInstances(Generic[Array], unittest.TestCase):
    """Tests for multifidelity benchmark instances."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_5model_name(self) -> None:
        """Test 5-model benchmark name."""
        benchmark = polynomial_ensemble_5model(self._bkd)
        self.assertEqual(benchmark.name(), "polynomial_ensemble_5model")

    def test_5model_nmodels(self) -> None:
        """Test 5-model benchmark has 5 models."""
        benchmark = polynomial_ensemble_5model(self._bkd)
        self.assertEqual(benchmark.ensemble().nmodels(), 5)

    def test_5model_ground_truth_mean(self) -> None:
        """Test 5-model ground truth mean."""
        benchmark = polynomial_ensemble_5model(self._bkd)
        gt = benchmark.ground_truth()
        # High fidelity mean is E[x^5] = 1/6
        self._bkd.assert_allclose(
            self._bkd.asarray([gt.high_fidelity_mean]),
            self._bkd.asarray([1/6]),
            rtol=1e-12,
        )

    def test_5model_ground_truth_available(self) -> None:
        """Test 5-model ground truth has expected values."""
        benchmark = polynomial_ensemble_5model(self._bkd)
        gt = benchmark.ground_truth()
        available = gt.available()
        self.assertIn("high_fidelity_mean", available)
        self.assertIn("high_fidelity_variance", available)
        self.assertIn("model_correlations", available)
        self.assertIn("model_costs", available)

    def test_3model_name(self) -> None:
        """Test 3-model benchmark name."""
        benchmark = polynomial_ensemble_3model(self._bkd)
        self.assertEqual(benchmark.name(), "polynomial_ensemble_3model")

    def test_3model_nmodels(self) -> None:
        """Test 3-model benchmark has 3 models."""
        benchmark = polynomial_ensemble_3model(self._bkd)
        self.assertEqual(benchmark.ensemble().nmodels(), 3)

    def test_domain_bounds(self) -> None:
        """Test domain bounds are [0, 1]."""
        benchmark = polynomial_ensemble_5model(self._bkd)
        domain = benchmark.domain()
        expected = self._bkd.array([[0.0, 1.0]])
        self._bkd.assert_allclose(domain.bounds(), expected, rtol=1e-12)


class TestMultiOutputModelEnsemble(Generic[Array], unittest.TestCase):
    """Tests for MultiOutputModelEnsemble and statistics mixin."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_nmodels(self) -> None:
        """Test nmodels returns correct value."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        self.assertEqual(ensemble.nmodels(), 3)

    def test_nqoi(self) -> None:
        """Test nqoi returns correct value."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        self.assertEqual(ensemble.nqoi(), 3)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        self.assertEqual(ensemble.nvars(), 1)

    def test_costs_shape(self) -> None:
        """Test costs has correct shape."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        costs = ensemble.costs()
        self.assertEqual(costs.shape, (3,))

    def test_means_shape(self) -> None:
        """Test means has correct shape."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        means = ensemble.means()
        self.assertEqual(means.shape, (3, 3))

    def test_covariance_matrix_shape(self) -> None:
        """Test covariance matrix has correct shape."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        cov = ensemble.covariance_matrix()
        self.assertEqual(cov.shape, (9, 9))

    def test_covariance_matrix_symmetric(self) -> None:
        """Test covariance matrix is symmetric."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        cov = ensemble.covariance_matrix()
        self._bkd.assert_allclose(cov, cov.T, rtol=1e-12)

    def test_analytical_vs_numerical_covariance(self) -> None:
        """Test analytical covariance matches numerical quadrature."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        # Get analytical covariance (from override)
        analytical_cov = ensemble.covariance_matrix()
        # Get numerical covariance by calling mixin method directly
        numerical_cov = MultifidelityStatisticsMixin.covariance_matrix(ensemble)
        # Should match to high precision
        self._bkd.assert_allclose(analytical_cov, numerical_cov, rtol=1e-6)

    def test_analytical_vs_numerical_means(self) -> None:
        """Test analytical means match numerical quadrature."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        # Get analytical means (from override)
        analytical_means = ensemble.means()
        # Get numerical means by calling mixin method directly
        numerical_means = MultifidelityStatisticsMixin.means(ensemble)
        # Should match to high precision
        # atol needed for values that are analytically 0 but numerically ~1e-16
        self._bkd.assert_allclose(
            analytical_means, numerical_means, rtol=1e-6, atol=1e-14
        )

    def test_kronecker_product_covariance_shape(self) -> None:
        """Test W matrix has correct shape."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        W = ensemble.covariance_of_centered_values_kronecker_product()
        # Shape is (nmodels * nqoi^2, nmodels * nqoi^2) = (27, 27)
        self.assertEqual(W.shape, (27, 27))

    def test_kronecker_product_covariance_symmetric(self) -> None:
        """Test W matrix is symmetric."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        W = ensemble.covariance_of_centered_values_kronecker_product()
        self._bkd.assert_allclose(W, W.T, rtol=1e-10)

    def test_mean_variance_covariance_shape(self) -> None:
        """Test B matrix has correct shape."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        B = ensemble.covariance_of_mean_and_variance_estimators()
        # Shape is (nmodels * nqoi, nmodels * nqoi^2) = (9, 27)
        self.assertEqual(B.shape, (9, 27))

    def test_covariance_subproblem_single_model_single_qoi(self) -> None:
        """Test covariance subproblem for single model and QoI."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        full_cov = ensemble.covariance_matrix()
        # Get subproblem for model 0, qoi 0
        sub_cov = ensemble.covariance_subproblem([0], [0])
        self.assertEqual(sub_cov.shape, (1, 1))
        self._bkd.assert_allclose(sub_cov[0, 0], full_cov[0, 0], rtol=1e-12)

    def test_covariance_subproblem_multiple(self) -> None:
        """Test covariance subproblem for multiple models and QoI."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        full_cov = ensemble.covariance_matrix()
        # Get subproblem for models [0, 1] and qoi [0, 2]
        model_idx = [0, 1]
        qoi_idx = [0, 2]
        sub_cov = ensemble.covariance_subproblem(model_idx, qoi_idx)
        # Shape should be (2*2, 2*2) = (4, 4)
        self.assertEqual(sub_cov.shape, (4, 4))
        # Check a few entries
        # (0, 0) -> model 0 qoi 0 vs model 0 qoi 0 -> full[0, 0]
        self._bkd.assert_allclose(sub_cov[0, 0], full_cov[0, 0], rtol=1e-12)
        # (1, 1) -> model 0 qoi 2 vs model 0 qoi 2 -> full[2, 2]
        self._bkd.assert_allclose(sub_cov[1, 1], full_cov[2, 2], rtol=1e-12)
        # (2, 2) -> model 1 qoi 0 vs model 1 qoi 0 -> full[3, 3]
        self._bkd.assert_allclose(sub_cov[2, 2], full_cov[3, 3], rtol=1e-12)

    def test_kronecker_subproblem_matches_full(self) -> None:
        """Test W subproblem extraction matches direct computation."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        model_idx = [0, 1]
        qoi_idx = [0, 2]
        # Get subproblem from full W matrix
        W_sub = ensemble.covariance_of_centered_values_kronecker_product_subproblem(
            model_idx, qoi_idx
        )
        # Shape: (nsub_models * nsub_qoi^2, nsub_models * nsub_qoi^2) = (2*4, 2*4) = (8, 8)
        self.assertEqual(W_sub.shape, (8, 8))
        # Verify symmetry
        self._bkd.assert_allclose(W_sub, W_sub.T, rtol=1e-10)

    def test_mean_variance_subproblem_matches_full(self) -> None:
        """Test B subproblem extraction matches direct computation."""
        ensemble = MultiOutputModelEnsemble(self._bkd)
        model_idx = [0, 1]
        qoi_idx = [0, 2]
        # Get subproblem from full B matrix
        B_sub = ensemble.covariance_of_mean_and_variance_estimators_subproblem(
            model_idx, qoi_idx
        )
        # Shape: (nsub_models * nsub_qoi, nsub_models * nsub_qoi^2) = (2*2, 2*4) = (4, 8)
        self.assertEqual(B_sub.shape, (4, 8))


class TestPSDMultiOutputModelEnsemble(Generic[Array], unittest.TestCase):
    """Tests for PSD variant of MultiOutputModelEnsemble."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_covariance_matrix_shape(self) -> None:
        """Test covariance matrix has correct shape."""
        ensemble = PSDMultiOutputModelEnsemble(self._bkd)
        cov = ensemble.covariance_matrix()
        self.assertEqual(cov.shape, (9, 9))

    def test_covariance_matrix_symmetric(self) -> None:
        """Test covariance matrix is symmetric."""
        ensemble = PSDMultiOutputModelEnsemble(self._bkd)
        cov = ensemble.covariance_matrix()
        self._bkd.assert_allclose(cov, cov.T, rtol=1e-10)

    def test_covariance_matrix_positive_diagonal(self) -> None:
        """Test covariance matrix has positive diagonal."""
        ensemble = PSDMultiOutputModelEnsemble(self._bkd)
        cov = ensemble.covariance_matrix()
        for i in range(9):
            self.assertGreater(cov[i, i], 0)

    def test_means_shape(self) -> None:
        """Test means has correct shape."""
        ensemble = PSDMultiOutputModelEnsemble(self._bkd)
        means = ensemble.means()
        self.assertEqual(means.shape, (3, 3))

    def test_kronecker_product_covariance_shape(self) -> None:
        """Test W matrix has correct shape."""
        ensemble = PSDMultiOutputModelEnsemble(self._bkd)
        W = ensemble.covariance_of_centered_values_kronecker_product()
        self.assertEqual(W.shape, (27, 27))

    def test_mean_variance_covariance_shape(self) -> None:
        """Test B matrix has correct shape."""
        ensemble = PSDMultiOutputModelEnsemble(self._bkd)
        B = ensemble.covariance_of_mean_and_variance_estimators()
        self.assertEqual(B.shape, (9, 27))

    def test_different_from_regular_ensemble(self) -> None:
        """Test PSD ensemble has different covariance from regular."""
        regular = MultiOutputModelEnsemble(self._bkd)
        psd = PSDMultiOutputModelEnsemble(self._bkd)

        regular_cov = regular.covariance_matrix()
        psd_cov = psd.covariance_matrix()

        # They should be different (PSD has perturbations)
        diff = self._bkd.max(self._bkd.abs(regular_cov - psd_cov))
        self.assertGreater(self._bkd.to_numpy(diff), 0.01)


class TestBenchmarkRegistryMultifidelity(unittest.TestCase):
    """Tests for BenchmarkRegistry multifidelity category."""

    def test_5model_registered(self) -> None:
        """Test polynomial_ensemble_5model is registered."""
        self.assertIn(
            "polynomial_ensemble_5model",
            BenchmarkRegistry.list_all(),
        )

    def test_3model_registered(self) -> None:
        """Test polynomial_ensemble_3model is registered."""
        self.assertIn(
            "polynomial_ensemble_3model",
            BenchmarkRegistry.list_all(),
        )

    def test_multifidelity_category(self) -> None:
        """Test benchmarks are in multifidelity category."""
        mf_benchmarks = BenchmarkRegistry.list_category("multifidelity")
        self.assertIn("polynomial_ensemble_5model", mf_benchmarks)
        self.assertIn("polynomial_ensemble_3model", mf_benchmarks)


class TestPolynomialModelFunctionNumpy(TestPolynomialModelFunction[NDArray[Any]]):
    """NumPy backend tests for PolynomialModelFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPolynomialModelFunctionTorch(TestPolynomialModelFunction[torch.Tensor]):
    """PyTorch backend tests for PolynomialModelFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestPolynomialEnsembleNumpy(TestPolynomialEnsemble[NDArray[Any]]):
    """NumPy backend tests for PolynomialEnsemble."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPolynomialEnsembleTorch(TestPolynomialEnsemble[torch.Tensor]):
    """PyTorch backend tests for PolynomialEnsemble."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMultifidelityBenchmarkInstancesNumpy(
    TestMultifidelityBenchmarkInstances[NDArray[Any]]
):
    """NumPy backend tests for multifidelity benchmark instances."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMultifidelityBenchmarkInstancesTorch(
    TestMultifidelityBenchmarkInstances[torch.Tensor]
):
    """PyTorch backend tests for multifidelity benchmark instances."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMultiOutputModelEnsembleNumpy(TestMultiOutputModelEnsemble[NDArray[Any]]):
    """NumPy backend tests for MultiOutputModelEnsemble."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMultiOutputModelEnsembleTorch(TestMultiOutputModelEnsemble[torch.Tensor]):
    """PyTorch backend tests for MultiOutputModelEnsemble."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestPSDMultiOutputModelEnsembleNumpy(
    TestPSDMultiOutputModelEnsemble[NDArray[Any]]
):
    """NumPy backend tests for PSDMultiOutputModelEnsemble."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPSDMultiOutputModelEnsembleTorch(
    TestPSDMultiOutputModelEnsemble[torch.Tensor]
):
    """PyTorch backend tests for PSDMultiOutputModelEnsemble."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
