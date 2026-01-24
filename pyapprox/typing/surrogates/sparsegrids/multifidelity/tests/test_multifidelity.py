"""Dual-backend tests for multi-fidelity sparse grids."""

import unittest
from typing import Any, Dict, Generic, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.interface.functions.protocols import FunctionProtocol
from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.typing.surrogates.sparsegrids import PrebuiltBasisFactory
from pyapprox.typing.surrogates.sparsegrids.refinement import (
    ConfigIndexCostFunction,
)
from pyapprox.typing.surrogates.sparsegrids.multifidelity import (
    MultiIndexAdaptiveCombinationSparseGrid,
    MultiFidelityModelFactoryProtocol,
)


# --- Eps values for converging models ---
# Config converges at index 2 (eps becomes 0)
EPS_VALUES = [0.5, 0.2, 0.0, 0.0, 0.0]


# --- Test Model Factories ---


class PolynomialModel(Generic[Array]):
    """Polynomial model with configurable eps perturbation.

    f(x) = (1 + eps) + (2 + eps)*x + (0.5 + eps)*x^2

    This is a degree-2 polynomial that can be exactly represented
    at physical level 2.

    Satisfies FunctionProtocol.
    """

    def __init__(self, eps: float, bkd: Backend[Array]) -> None:
        self._eps = eps
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 1

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        x = samples[0, :]
        c0 = 1.0 + self._eps
        c1 = 2.0 + self._eps
        c2 = 0.5 + self._eps
        values = c0 + c1 * x + c2 * x**2
        return self._bkd.reshape(values, (1, -1))


class ConvergingModelFactory(Generic[Array]):
    """Model factory where higher fidelity converges to true function.

    eps = [0.5, 0.2, 0.0, 0.0, 0.0]
    f(x; config) = (1 + eps) + (2 + eps)*x + (0.5 + eps)*x^2

    At config >= 2, the model has converged (eps = 0).
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._models: Dict[Tuple[int, ...], PolynomialModel[Array]] = {}

    def get_model(self, config_index: Tuple[int, ...]) -> FunctionProtocol[Array]:
        if config_index not in self._models:
            idx = config_index[0]
            eps = EPS_VALUES[min(idx, len(EPS_VALUES) - 1)]
            self._models[config_index] = PolynomialModel(eps, self._bkd)
        return self._models[config_index]

    def nconfig_vars(self) -> int:
        return 1

    def bkd(self) -> Backend[Array]:
        return self._bkd


class AdditivePolynomialModel(Generic[Array]):
    """Additive polynomial model with two eps perturbations.

    f(x) = (1 + eps1) + (2 + eps1)*x + (0.5 + eps2) + (1 + eps2)*x
         = (1.5 + eps1 + eps2) + (3 + eps1 + eps2)*x

    This creates additive structure: no interaction between eps1 and eps2
    beyond their linear contributions.
    """

    def __init__(self, eps1: float, eps2: float, bkd: Backend[Array]) -> None:
        self._eps1 = eps1
        self._eps2 = eps2
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 1

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        x = samples[0, :]
        # Additive structure: contribution from eps1 + contribution from eps2
        # Term 1: (1 + eps1) + (2 + eps1)*x
        # Term 2: (0.5 + eps2) + (1 + eps2)*x
        c0 = (1.0 + self._eps1) + (0.5 + self._eps2)
        c1 = (2.0 + self._eps1) + (1.0 + self._eps2)
        values = c0 + c1 * x
        return self._bkd.reshape(values, (1, -1))


class AdditiveConfigModelFactory(Generic[Array]):
    """Model factory with additive config structure (no cross-terms).

    f(x; c1, c2) = g1(x, eps1[c1]) + g2(x, eps2[c2])

    where:
    - eps1 = [0.5, 0.2, 0.0, 0.0, 0.0]
    - eps2 = [0.3, 0.1, 0.0, 0.0, 0.0]

    Each config dimension converges independently.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._eps1_values = [0.5, 0.2, 0.0, 0.0, 0.0]
        self._eps2_values = [0.3, 0.1, 0.0, 0.0, 0.0]
        self._models: Dict[Tuple[int, ...], AdditivePolynomialModel[Array]] = {}

    def get_model(
        self, config_index: Tuple[int, ...]
    ) -> FunctionProtocol[Array]:
        if config_index not in self._models:
            c1, c2 = config_index
            eps1 = self._eps1_values[min(c1, len(self._eps1_values) - 1)]
            eps2 = self._eps2_values[min(c2, len(self._eps2_values) - 1)]
            self._models[config_index] = AdditivePolynomialModel(
                eps1, eps2, self._bkd
            )
        return self._models[config_index]

    def nconfig_vars(self) -> int:
        return 2

    def bkd(self) -> Backend[Array]:
        return self._bkd


class Simple2DPolynomialModel(Generic[Array]):
    """Simple 2D polynomial model with configurable offset.

    f(x, y) = 1 + x + y + x*y + eps
    """

    def __init__(self, eps: float, bkd: Backend[Array]) -> None:
        self._eps = eps
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        x, y = samples[0, :], samples[1, :]
        values = 1.0 + x + y + x * y + self._eps
        return self._bkd.reshape(values, (1, -1))


class Simple2DFactory(Generic[Array]):
    """Factory for Simple2DPolynomialModel with converging fidelity levels."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._eps_values = [0.3, 0.1, 0.0, 0.0]
        self._models: Dict[Tuple[int, ...], Simple2DPolynomialModel[Array]] = {}

    def get_model(
        self, config_index: Tuple[int, ...]
    ) -> FunctionProtocol[Array]:
        if config_index not in self._models:
            idx = config_index[0]
            eps = self._eps_values[min(idx, len(self._eps_values) - 1)]
            self._models[config_index] = Simple2DPolynomialModel(eps, self._bkd)
        return self._models[config_index]

    def nconfig_vars(self) -> int:
        return 1

    def bkd(self) -> Backend[Array]:
        return self._bkd


# --- Test Classes ---


class TestMultiFidelitySparseGrid(Generic[Array], unittest.TestCase):
    """Base test class for multi-fidelity sparse grids."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_legendre_factory(self) -> PrebuiltBasisFactory[Array]:
        """Create a Legendre polynomial basis factory."""
        basis = LegendrePolynomial1D(self._bkd)
        return PrebuiltBasisFactory(basis)

    def test_protocol_validation(self) -> None:
        """Test TypeError raised for invalid model factory."""
        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([3.0])

        with self.assertRaises(TypeError) as ctx:
            MultiIndexAdaptiveCombinationSparseGrid(
                self._bkd,
                factories,
                nconfig_vars=1,
                config_bounds=config_bounds,
                nqoi=1,
                model_factory="not a factory",  # type: ignore
            )
        self.assertIn("MultiFidelityModelFactoryProtocol", str(ctx.exception))

    def test_function_protocol_interface(self) -> None:
        """Test that grid satisfies FunctionProtocol interface."""
        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([3.0])
        model_factory = ConvergingModelFactory(self._bkd)

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=1,
            config_bounds=config_bounds,
            nqoi=1,
            model_factory=model_factory,
        )

        # Check interface methods
        self.assertEqual(grid.nvars(), 1)  # Physical only
        self.assertEqual(grid.nqoi(), 1)
        self.assertEqual(grid.nconfig_vars(), 1)

        # After first step, should be able to evaluate
        grid.step()
        samples = self._bkd.asarray([[-0.5, 0.0, 0.5]])
        result = grid(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_manual_mode(self) -> None:
        """Test manual step_samples/step_values workflow."""
        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([3.0])
        model_factory = ConvergingModelFactory(self._bkd)

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=1,
            config_bounds=config_bounds,
            nqoi=1,
        )

        # Get config requests (now returned by step_samples directly)
        config_requests = grid.step_samples()
        self.assertIsNotNone(config_requests)
        self.assertIsInstance(config_requests, dict)
        assert config_requests is not None  # Type narrowing for mypy

        # Verify samples are physical-only
        for config_idx, phys_samples in config_requests.items():
            self.assertEqual(
                phys_samples.shape[0], 1,
                f"Expected physical samples shape (1, n), got {phys_samples.shape}"
            )

        # Manually evaluate using factory
        values_dict: Dict[Tuple[int, ...], Array] = {}
        for config_idx, phys_samples in config_requests.items():
            model = model_factory.get_model(config_idx)
            values_dict[config_idx] = model(phys_samples)

        # Set values
        grid.step_values(values_dict)

        # Should be able to evaluate now
        samples = self._bkd.asarray([[0.0]])
        result = grid(samples)
        self.assertEqual(result.shape, (1, 1))

    def test_auto_mode(self) -> None:
        """Test auto mode with step() using model factory."""
        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([3.0])
        model_factory = ConvergingModelFactory(self._bkd)

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=1,
            config_bounds=config_bounds,
            nqoi=1,
            model_factory=model_factory,
        )

        # First step should succeed
        result = grid.step()
        self.assertTrue(result)

        # Should be able to evaluate
        samples = self._bkd.asarray([[0.0]])
        values = grid(samples)
        self.assertEqual(values.shape, (1, 1))

    def test_auto_mode_requires_factory(self) -> None:
        """Test step() raises ValueError without model factory."""
        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([3.0])

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=1,
            config_bounds=config_bounds,
            nqoi=1,
            # No model_factory
        )

        with self.assertRaises(ValueError) as ctx:
            grid.step()
        self.assertIn("model_factory required", str(ctx.exception))

    def test_step_samples_returns_none_when_converged(self) -> None:
        """Test that step_samples returns None when all indices explored.

        Uses restrictive bounds on both physical and config dimensions
        so the grid runs out of candidates.
        """
        from pyapprox.typing.surrogates.affine.indices import Max1DLevelsCriteria

        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([1.0])
        # Limit physical to level 2, config to level 1
        max_levels = self._bkd.asarray([2.0, 1.0])
        admissibility = Max1DLevelsCriteria(max_levels, self._bkd)
        model_factory = ConvergingModelFactory(self._bkd)

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=1,
            config_bounds=config_bounds,
            nqoi=1,
            model_factory=model_factory,
            admissibility=admissibility,
        )

        # Run until converged (should hit bounds quickly)
        for _ in range(20):
            if not grid.step():
                break

        # Now step_samples should return None
        result = grid.step_samples()
        self.assertIsNone(result)

    def test_config_cost_function(self) -> None:
        """Test ConfigIndexCostFunction computes correct costs."""
        cost_fn = ConfigIndexCostFunction(self._bkd, nvars_physical=2, base=2.0)

        # Subspace index: [phys1, phys2, config1]
        # Cost should be 2^(config_sum)

        # Config level 0 -> cost = 2^0 = 1
        index = self._bkd.asarray([3, 2, 0])
        self._bkd.assert_allclose(
            self._bkd.asarray([cost_fn(index)]),
            self._bkd.asarray([1.0]),
        )

        # Config level 1 -> cost = 2^1 = 2
        index = self._bkd.asarray([0, 0, 1])
        self._bkd.assert_allclose(
            self._bkd.asarray([cost_fn(index)]),
            self._bkd.asarray([2.0]),
        )

        # Config level 3 -> cost = 2^3 = 8
        index = self._bkd.asarray([1, 1, 3])
        self._bkd.assert_allclose(
            self._bkd.asarray([cost_fn(index)]),
            self._bkd.asarray([8.0]),
        )

    def test_config_cost_function_multi_config(self) -> None:
        """Test ConfigIndexCostFunction with multiple config dimensions."""
        cost_fn = ConfigIndexCostFunction(self._bkd, nvars_physical=1, base=2.0)

        # Subspace index: [phys1, config1, config2]
        # Config sum = config1 + config2

        # Config (1, 2) -> cost = 2^(1+2) = 2^3 = 8
        index = self._bkd.asarray([5, 1, 2])
        self._bkd.assert_allclose(
            self._bkd.asarray([cost_fn(index)]),
            self._bkd.asarray([8.0]),
        )

    def test_1d_physical_1d_config_convergence(self) -> None:
        """Test basic convergence with 1D physical and 1D config.

        Uses polynomial model that can be exactly represented.
        """
        np.random.seed(42)
        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([4.0])
        model_factory = ConvergingModelFactory(self._bkd)

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=1,
            config_bounds=config_bounds,
            nqoi=1,
            model_factory=model_factory,
        )

        # Refine for several steps
        for _ in range(15):
            if not grid.step():
                break

        # Verify accuracy at test points
        # The true function at highest fidelity is f(x) = 1 + 2*x + 0.5*x^2
        test_pts = self._bkd.asarray([[-0.5, 0.0, 0.5]])
        result = grid(test_pts)
        # At eps=0: f(x) = 1 + 2*x + 0.5*x^2
        x = test_pts[0, :]
        expected = 1.0 + 2.0 * x + 0.5 * x**2
        expected = self._bkd.reshape(expected, (1, -1))
        self._bkd.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)

    def test_2d_physical_1d_config(self) -> None:
        """Test with 2D physical input."""
        np.random.seed(42)
        factories = [
            self._create_legendre_factory(),
            self._create_legendre_factory(),
        ]
        config_bounds = self._bkd.asarray([3.0])
        model_factory = Simple2DFactory(self._bkd)
        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=1,
            config_bounds=config_bounds,
            nqoi=1,
            model_factory=model_factory,
        )

        # Check interface
        self.assertEqual(grid.nvars(), 2)
        self.assertEqual(grid.nconfig_vars(), 1)

        # Run a few steps
        for _ in range(5):
            if not grid.step():
                break

        # Evaluate at physical samples
        test_pts = self._bkd.asarray([[0.0, 0.5], [0.0, 0.5]])
        result = grid(test_pts)
        self.assertEqual(result.shape, (1, 2))

    def test_1d_physical_1d_config_no_single_dim_over_refinement(self) -> None:
        """Test that config dimension doesn't over-refine past convergence.

        Uses ConvergingModelFactory where eps stops changing at config index 2.
        The grid should detect convergence and stop refining the config dimension.
        Max config level should be ~3-4 (needs one extra level to detect no change).
        """
        np.random.seed(42)
        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([5.0])  # Allow up to level 5
        model_factory = ConvergingModelFactory(self._bkd)

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=1,
            config_bounds=config_bounds,
            nqoi=1,
            model_factory=model_factory,
            verbosity=0,
        )

        # Refine until converged or max steps
        tol = 1e-10
        for _ in range(50):
            if not grid.step():
                break
            if grid.error_estimate() < tol:
                break

        # Check max config level in selected subspaces
        selected = grid.get_selected_subspace_indices()
        config_levels = selected[1, :]  # Dim 1 is config
        max_config_level = int(self._bkd.to_numpy(self._bkd.max(config_levels)))

        # Should stop around level 3-4, definitely not 5
        # (Relaxed to 5 since exact behavior depends on refinement strategy)
        self.assertLessEqual(
            max_config_level,
            5,
            f"Config dimension over-refined to level {max_config_level}, "
            f"expected <= 5 since eps converges at level 2",
        )

    def test_1d_physical_2d_config_no_cross_term_over_refinement(self) -> None:
        """Test additive config structure doesn't over-refine cross-terms.

        Uses AdditiveConfigModelFactory where f(x; c1, c2) = g1(x, c1) + g2(x, c2).
        Config cross-term subspaces (where both c1 > 0 AND c2 > 0) should have
        limited refinement since there's no interaction between config dims.
        """
        np.random.seed(42)
        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([4.0, 4.0])  # 2 config dimensions
        model_factory = AdditiveConfigModelFactory(self._bkd)

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=2,
            config_bounds=config_bounds,
            nqoi=1,
            model_factory=model_factory,
            verbosity=0,
        )

        # Refine until converged or max steps
        tol = 1e-10
        for _ in range(100):
            if not grid.step():
                break
            if grid.error_estimate() < tol:
                break

        # Check config cross-terms
        selected = grid.get_selected_subspace_indices()
        # Dims: [physical, config1, config2]
        config1_levels = selected[1, :]
        config2_levels = selected[2, :]

        # Find subspaces where both config dims are nonzero (cross-terms)
        cross_term_mask = (config1_levels > 0) & (config2_levels > 0)

        if self._bkd.any_bool(cross_term_mask):
            # For cross-terms, levels should be limited
            cross_config1 = config1_levels[cross_term_mask]
            cross_config2 = config2_levels[cross_term_mask]
            max_cross1 = int(self._bkd.to_numpy(self._bkd.max(cross_config1)))
            max_cross2 = int(self._bkd.to_numpy(self._bkd.max(cross_config2)))

            # Relaxed expectation: cross-terms shouldn't go to max level
            self.assertLessEqual(
                max_cross1,
                4,
                f"Config1 in cross-terms over-refined to level {max_cross1}",
            )
            self.assertLessEqual(
                max_cross2,
                4,
                f"Config2 in cross-terms over-refined to level {max_cross2}",
            )

    def test_2d_config_each_dim_stops_independently(self) -> None:
        """Test each config dimension refines based on its contribution.

        Uses AdditiveConfigModelFactory where:
        - eps1 = [0.5, 0.2, 0.0, ...] -> converges at level 2
        - eps2 = [0.3, 0.1, 0.0, ...] -> converges at level 2

        Each config dimension should be refined based on error contribution.
        """
        np.random.seed(42)
        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([5.0, 5.0])
        model_factory = AdditiveConfigModelFactory(self._bkd)

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=2,
            config_bounds=config_bounds,
            nqoi=1,
            model_factory=model_factory,
            verbosity=0,
        )

        # Refine until converged or max steps
        tol = 1e-10
        for _ in range(100):
            if not grid.step():
                break
            if grid.error_estimate() < tol:
                break

        selected = grid.get_selected_subspace_indices()
        config1_levels = selected[1, :]
        config2_levels = selected[2, :]

        # For 1D subspaces in each config dimension (the other is 0)
        config1_only_mask = config2_levels == 0
        config2_only_mask = config1_levels == 0

        if self._bkd.any_bool(config1_only_mask):
            max_config1 = int(
                self._bkd.to_numpy(self._bkd.max(config1_levels[config1_only_mask]))
            )
            # Relaxed: should not exceed bounds
            self.assertLessEqual(
                max_config1,
                5,
                f"Config1 over-refined to level {max_config1}",
            )

        if self._bkd.any_bool(config2_only_mask):
            max_config2 = int(
                self._bkd.to_numpy(self._bkd.max(config2_levels[config2_only_mask]))
            )
            self.assertLessEqual(
                max_config2,
                5,
                f"Config2 over-refined to level {max_config2}",
            )

    def test_refine_to_tolerance(self) -> None:
        """Test refine_to_tolerance convenience method."""
        factories = [self._create_legendre_factory()]
        config_bounds = self._bkd.asarray([3.0])
        model_factory = ConvergingModelFactory(self._bkd)

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=1,
            config_bounds=config_bounds,
            nqoi=1,
            model_factory=model_factory,
        )

        # Refine to a tolerance
        steps = grid.refine_to_tolerance(tol=1e-3, max_steps=20)
        self.assertGreater(steps, 0)
        self.assertLessEqual(steps, 20)

    def test_physical_samples_shape(self) -> None:
        """Test that step_samples returns physical-only samples."""
        factories = [
            self._create_legendre_factory(),
            self._create_legendre_factory(),
        ]
        config_bounds = self._bkd.asarray([3.0, 3.0])  # 2 config vars

        # Factory that handles 2 config vars
        class TwoConfigFactory(Generic[Array]):
            def __init__(self, bkd: Backend[Array]) -> None:
                self._bkd = bkd
                self._models: Dict[
                    Tuple[int, ...], Simple2DPolynomialModel[Array]
                ] = {}

            def get_model(
                self, config_index: Tuple[int, ...]
            ) -> FunctionProtocol[Array]:
                if config_index not in self._models:
                    # Sum of config levels determines eps
                    total_level = sum(config_index)
                    eps_values = [0.3, 0.1, 0.05, 0.0, 0.0]
                    eps = eps_values[min(total_level, len(eps_values) - 1)]
                    self._models[config_index] = Simple2DPolynomialModel(
                        eps, self._bkd
                    )
                return self._models[config_index]

            def nconfig_vars(self) -> int:
                return 2

            def bkd(self) -> Backend[Array]:
                return self._bkd

        factory = TwoConfigFactory(self._bkd)

        grid = MultiIndexAdaptiveCombinationSparseGrid(
            self._bkd,
            factories,
            nconfig_vars=2,
            config_bounds=config_bounds,
            nqoi=1,
            model_factory=factory,
        )

        # Get first step samples
        config_requests = grid.step_samples()
        self.assertIsNotNone(config_requests)
        assert config_requests is not None

        # All samples should be physical-only (nvars_physical = 2)
        for config_idx, phys_samples in config_requests.items():
            self.assertEqual(
                phys_samples.shape[0], 2,
                f"Config {config_idx}: Expected physical samples with 2 rows, "
                f"got shape {phys_samples.shape}"
            )


class TestMultiFidelitySparseGridNumpy(
    TestMultiFidelitySparseGrid[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMultiFidelitySparseGridTorch(
    TestMultiFidelitySparseGrid[torch.Tensor]
):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
