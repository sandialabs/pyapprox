"""Tests for ODE benchmark instances."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests

from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.instances.ode import (
    lotka_volterra_3species,
    coupled_springs_2mass,
    hastings_ecology_3species,
    chemical_reaction_surface,
)


class TestLotkaVolterra3SpeciesBenchmark(Generic[Array], unittest.TestCase):
    """Tests for lotka_volterra_3species benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = lotka_volterra_3species(self._bkd)
        self.assertEqual(benchmark.name(), "lotka_volterra_3species")

    def test_nstates(self) -> None:
        """Test number of states."""
        benchmark = lotka_volterra_3species(self._bkd)
        self.assertEqual(benchmark.nstates(), 3)

    def test_nparams(self) -> None:
        """Test number of parameters."""
        benchmark = lotka_volterra_3species(self._bkd)
        self.assertEqual(benchmark.nparams(), 12)

    def test_domain_nvars(self) -> None:
        """Test domain nvars matches nparams."""
        benchmark = lotka_volterra_3species(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 12)

    def test_domain_bounds(self) -> None:
        """Test domain bounds are [0.3, 0.7]^12."""
        benchmark = lotka_volterra_3species(self._bkd)
        bounds = benchmark.domain().bounds()
        expected = self._bkd.array([[0.3, 0.7]] * 12)
        self._bkd.assert_allclose(bounds, expected, atol=1e-14)

    def test_ground_truth_nstates(self) -> None:
        """Test ground truth nstates."""
        benchmark = lotka_volterra_3species(self._bkd)
        gt = benchmark.ground_truth()
        self.assertEqual(gt.nstates, 3)

    def test_ground_truth_nparams(self) -> None:
        """Test ground truth nparams."""
        benchmark = lotka_volterra_3species(self._bkd)
        gt = benchmark.ground_truth()
        self.assertEqual(gt.nparams, 12)

    def test_ground_truth_initial_condition(self) -> None:
        """Test ground truth initial condition has shape (nstates, 1)."""
        benchmark = lotka_volterra_3species(self._bkd)
        gt = benchmark.ground_truth()
        expected = self._bkd.array([[0.3], [0.4], [0.3]])
        self._bkd.assert_allclose(gt.initial_condition, expected, atol=1e-14)

    def test_time_config(self) -> None:
        """Test time configuration."""
        benchmark = lotka_volterra_3species(self._bkd)
        tc = benchmark.time_config()
        self.assertEqual(tc.init_time, 0.0)
        self.assertEqual(tc.final_time, 10.0)
        self.assertEqual(tc.deltat, 1.0)
        self.assertEqual(tc.ntimes(), 11)

    def test_prior_nvars(self) -> None:
        """Test prior has correct nvars."""
        benchmark = lotka_volterra_3species(self._bkd)
        prior = benchmark.prior()
        self.assertEqual(prior.nvars(), 12)

    def test_prior_samples_in_domain(self) -> None:
        """Test that samples from prior are in domain."""
        benchmark = lotka_volterra_3species(self._bkd)
        prior = benchmark.prior()
        bounds = benchmark.domain().bounds()

        samples = prior.rvs(100)
        self.assertEqual(samples.shape, (12, 100))

        # Check all samples are within bounds
        for i in range(12):
            self.assertTrue(
                self._bkd.all_bool(samples[i, :] >= bounds[i, 0])
            )
            self.assertTrue(
                self._bkd.all_bool(samples[i, :] <= bounds[i, 1])
            )

    def test_residual_evaluation(self) -> None:
        """Test residual can be evaluated at valid states."""
        benchmark = lotka_volterra_3species(self._bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        # Set nominal parameters
        param = self._bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Evaluate at initial condition (flatten from (nstates, 1) to (nstates,))
        state = self._bkd.flatten(gt.initial_condition)
        f = residual(state)
        self.assertEqual(f.shape, (3,))

    def test_residual_jacobian(self) -> None:
        """Test residual Jacobian has correct shape."""
        benchmark = lotka_volterra_3species(self._bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        param = self._bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Flatten from (nstates, 1) to (nstates,)
        state = self._bkd.flatten(gt.initial_condition)
        jac = residual.jacobian(state)
        self.assertEqual(jac.shape, (3, 3))

    def test_residual_param_jacobian(self) -> None:
        """Test residual parameter Jacobian has correct shape."""
        benchmark = lotka_volterra_3species(self._bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        param = self._bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Flatten from (nstates, 1) to (nstates,)
        state = self._bkd.flatten(gt.initial_condition)
        pjac = residual.param_jacobian(state)
        self.assertEqual(pjac.shape, (3, 12))


class TestLotkaVolterra3SpeciesBenchmarkNumpy(
    TestLotkaVolterra3SpeciesBenchmark[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLotkaVolterra3SpeciesBenchmarkTorch(
    TestLotkaVolterra3SpeciesBenchmark[torch.Tensor]
):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestCoupledSprings2MassBenchmark(Generic[Array], unittest.TestCase):
    """Tests for coupled_springs_2mass benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = coupled_springs_2mass(self._bkd)
        self.assertEqual(benchmark.name(), "coupled_springs_2mass")

    def test_nstates(self) -> None:
        """Test number of states."""
        benchmark = coupled_springs_2mass(self._bkd)
        self.assertEqual(benchmark.nstates(), 4)

    def test_nparams(self) -> None:
        """Test number of parameters."""
        benchmark = coupled_springs_2mass(self._bkd)
        self.assertEqual(benchmark.nparams(), 12)

    def test_domain_nvars(self) -> None:
        """Test domain nvars matches nparams."""
        benchmark = coupled_springs_2mass(self._bkd)
        self.assertEqual(benchmark.domain().nvars(), 12)

    def test_time_config(self) -> None:
        """Test time configuration."""
        benchmark = coupled_springs_2mass(self._bkd)
        tc = benchmark.time_config()
        self.assertEqual(tc.init_time, 0.0)
        self.assertEqual(tc.final_time, 10.0)
        self.assertEqual(tc.deltat, 0.1)
        self.assertEqual(tc.ntimes(), 101)

    def test_ground_truth_nstates(self) -> None:
        """Test ground truth nstates."""
        benchmark = coupled_springs_2mass(self._bkd)
        gt = benchmark.ground_truth()
        self.assertEqual(gt.nstates, 4)

    def test_prior_nvars(self) -> None:
        """Test prior has correct nvars."""
        benchmark = coupled_springs_2mass(self._bkd)
        prior = benchmark.prior()
        self.assertEqual(prior.nvars(), 12)

    def test_residual_evaluation(self) -> None:
        """Test residual can be evaluated at valid states."""
        benchmark = coupled_springs_2mass(self._bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        param = self._bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Flatten from (nstates, 1) to (nstates,)
        state = self._bkd.flatten(gt.initial_condition)
        f = residual(state)
        self.assertEqual(f.shape, (4,))


class TestCoupledSprings2MassBenchmarkNumpy(
    TestCoupledSprings2MassBenchmark[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCoupledSprings2MassBenchmarkTorch(
    TestCoupledSprings2MassBenchmark[torch.Tensor]
):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestHastingsEcology3SpeciesBenchmark(Generic[Array], unittest.TestCase):
    """Tests for hastings_ecology_3species benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = hastings_ecology_3species(self._bkd)
        self.assertEqual(benchmark.name(), "hastings_ecology_3species")

    def test_nstates(self) -> None:
        """Test number of states."""
        benchmark = hastings_ecology_3species(self._bkd)
        self.assertEqual(benchmark.nstates(), 3)

    def test_nparams(self) -> None:
        """Test number of parameters."""
        benchmark = hastings_ecology_3species(self._bkd)
        self.assertEqual(benchmark.nparams(), 9)

    def test_time_config(self) -> None:
        """Test time configuration."""
        benchmark = hastings_ecology_3species(self._bkd)
        tc = benchmark.time_config()
        self.assertEqual(tc.init_time, 0.0)
        self.assertEqual(tc.final_time, 100.0)
        self.assertEqual(tc.deltat, 2.5)
        self.assertEqual(tc.ntimes(), 41)

    def test_ground_truth_nstates(self) -> None:
        """Test ground truth nstates."""
        benchmark = hastings_ecology_3species(self._bkd)
        gt = benchmark.ground_truth()
        self.assertEqual(gt.nstates, 3)

    def test_ground_truth_reference(self) -> None:
        """Test benchmark reference."""
        benchmark = hastings_ecology_3species(self._bkd)
        self.assertIn("Hastings", benchmark.reference())

    def test_prior_nvars(self) -> None:
        """Test prior has correct nvars."""
        benchmark = hastings_ecology_3species(self._bkd)
        prior = benchmark.prior()
        self.assertEqual(prior.nvars(), 9)

    def test_residual_evaluation(self) -> None:
        """Test residual can be evaluated at valid states."""
        benchmark = hastings_ecology_3species(self._bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        param = self._bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Flatten from (nstates, 1) to (nstates,)
        state = self._bkd.flatten(gt.initial_condition)
        f = residual(state)
        self.assertEqual(f.shape, (3,))


class TestHastingsEcology3SpeciesBenchmarkNumpy(
    TestHastingsEcology3SpeciesBenchmark[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestHastingsEcology3SpeciesBenchmarkTorch(
    TestHastingsEcology3SpeciesBenchmark[torch.Tensor]
):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestChemicalReactionSurfaceBenchmark(Generic[Array], unittest.TestCase):
    """Tests for chemical_reaction_surface benchmark instance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_name(self) -> None:
        """Test benchmark name."""
        benchmark = chemical_reaction_surface(self._bkd)
        self.assertEqual(benchmark.name(), "chemical_reaction_surface")

    def test_nstates(self) -> None:
        """Test number of states."""
        benchmark = chemical_reaction_surface(self._bkd)
        self.assertEqual(benchmark.nstates(), 3)

    def test_nparams(self) -> None:
        """Test number of parameters."""
        benchmark = chemical_reaction_surface(self._bkd)
        self.assertEqual(benchmark.nparams(), 6)

    def test_time_config(self) -> None:
        """Test time configuration."""
        benchmark = chemical_reaction_surface(self._bkd)
        tc = benchmark.time_config()
        self.assertEqual(tc.init_time, 0.0)
        self.assertEqual(tc.final_time, 100.0)
        self.assertEqual(tc.deltat, 0.1)
        self.assertEqual(tc.ntimes(), 1001)

    def test_ground_truth_initial_condition_zeros(self) -> None:
        """Test initial condition is zeros (empty surface) with shape (nstates, 1)."""
        benchmark = chemical_reaction_surface(self._bkd)
        gt = benchmark.ground_truth()
        expected = self._bkd.array([[0.0], [0.0], [0.0]])
        self._bkd.assert_allclose(gt.initial_condition, expected, atol=1e-14)

    def test_ground_truth_reference(self) -> None:
        """Test benchmark reference."""
        benchmark = chemical_reaction_surface(self._bkd)
        self.assertIn("Vigil", benchmark.reference())

    def test_prior_nvars(self) -> None:
        """Test prior has correct nvars."""
        benchmark = chemical_reaction_surface(self._bkd)
        prior = benchmark.prior()
        self.assertEqual(prior.nvars(), 6)

    def test_residual_evaluation(self) -> None:
        """Test residual can be evaluated at valid states."""
        benchmark = chemical_reaction_surface(self._bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        param = self._bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Flatten from (nstates, 1) to (nstates,)
        state = self._bkd.flatten(gt.initial_condition)
        f = residual(state)
        self.assertEqual(f.shape, (3,))


class TestChemicalReactionSurfaceBenchmarkNumpy(
    TestChemicalReactionSurfaceBenchmark[NDArray[Any]]
):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestChemicalReactionSurfaceBenchmarkTorch(
    TestChemicalReactionSurfaceBenchmark[torch.Tensor]
):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestODEBenchmarkRegistry(unittest.TestCase):
    """Test registry for ODE benchmarks."""

    def test_lotka_volterra_registered(self) -> None:
        """Test lotka_volterra_3species is registered."""
        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("lotka_volterra_3species", bkd)
        self.assertEqual(benchmark.name(), "lotka_volterra_3species")

    def test_coupled_springs_registered(self) -> None:
        """Test coupled_springs_2mass is registered."""
        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("coupled_springs_2mass", bkd)
        self.assertEqual(benchmark.name(), "coupled_springs_2mass")

    def test_hastings_ecology_registered(self) -> None:
        """Test hastings_ecology_3species is registered."""
        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("hastings_ecology_3species", bkd)
        self.assertEqual(benchmark.name(), "hastings_ecology_3species")

    def test_chemical_reaction_registered(self) -> None:
        """Test chemical_reaction_surface is registered."""
        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("chemical_reaction_surface", bkd)
        self.assertEqual(benchmark.name(), "chemical_reaction_surface")

    def test_ode_category(self) -> None:
        """Test all ODE benchmarks are in ode category."""
        category_benchmarks = BenchmarkRegistry.list_category("ode")
        self.assertIn("lotka_volterra_3species", category_benchmarks)
        self.assertIn("coupled_springs_2mass", category_benchmarks)
        self.assertIn("hastings_ecology_3species", category_benchmarks)
        self.assertIn("chemical_reaction_surface", category_benchmarks)


if __name__ == "__main__":
    unittest.main()
