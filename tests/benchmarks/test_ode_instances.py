"""Tests for ODE benchmark instances."""

# TODO: this test class should be where function is defined
# not at this level which is for integration tests.

from pyapprox.benchmarks.instances.ode import (
    chemical_reaction_surface,
    coupled_springs_2mass,
    hastings_ecology_3species,
    lotka_volterra_3species,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.util.backends.numpy import NumpyBkd


class TestLotkaVolterra3SpeciesBenchmark:
    """Tests for lotka_volterra_3species benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = lotka_volterra_3species(bkd)
        assert benchmark.name() == "lotka_volterra_3species"

    def test_nstates(self, bkd) -> None:
        """Test number of states."""
        benchmark = lotka_volterra_3species(bkd)
        assert benchmark.nstates() == 3

    def test_nparams(self, bkd) -> None:
        """Test number of parameters."""
        benchmark = lotka_volterra_3species(bkd)
        assert benchmark.nparams() == 12

    def test_domain_nvars(self, bkd) -> None:
        """Test domain nvars matches nparams."""
        benchmark = lotka_volterra_3species(bkd)
        assert benchmark.domain().nvars() == 12

    def test_domain_bounds(self, bkd) -> None:
        """Test domain bounds are [0.3, 0.7]^12."""
        benchmark = lotka_volterra_3species(bkd)
        bounds = benchmark.domain().bounds()
        expected = bkd.array([[0.3, 0.7]] * 12)
        bkd.assert_allclose(bounds, expected, atol=1e-14)

    def test_ground_truth_nstates(self, bkd) -> None:
        """Test ground truth nstates."""
        benchmark = lotka_volterra_3species(bkd)
        gt = benchmark.ground_truth()
        assert gt.nstates == 3

    def test_ground_truth_nparams(self, bkd) -> None:
        """Test ground truth nparams."""
        benchmark = lotka_volterra_3species(bkd)
        gt = benchmark.ground_truth()
        assert gt.nparams == 12

    def test_ground_truth_initial_condition(self, bkd) -> None:
        """Test ground truth initial condition has shape (nstates, 1)."""
        benchmark = lotka_volterra_3species(bkd)
        gt = benchmark.ground_truth()
        expected = bkd.array([[0.3], [0.4], [0.3]])
        bkd.assert_allclose(gt.initial_condition, expected, atol=1e-14)

    def test_time_config(self, bkd) -> None:
        """Test time configuration."""
        benchmark = lotka_volterra_3species(bkd)
        tc = benchmark.time_config()
        assert tc.init_time == 0.0
        assert tc.final_time == 10.0
        assert tc.deltat == 1.0
        assert tc.ntimes() == 11

    def test_prior_nvars(self, bkd) -> None:
        """Test prior has correct nvars."""
        benchmark = lotka_volterra_3species(bkd)
        prior = benchmark.prior()
        assert prior.nvars() == 12

    def test_prior_samples_in_domain(self, bkd) -> None:
        """Test that samples from prior are in domain."""
        benchmark = lotka_volterra_3species(bkd)
        prior = benchmark.prior()
        bounds = benchmark.domain().bounds()

        samples = prior.rvs(100)
        assert samples.shape == (12, 100)

        # Check all samples are within bounds
        for i in range(12):
            assert bkd.all_bool(samples[i, :] >= bounds[i, 0])
            assert bkd.all_bool(samples[i, :] <= bounds[i, 1])

    def test_residual_evaluation(self, bkd) -> None:
        """Test residual can be evaluated at valid states."""
        benchmark = lotka_volterra_3species(bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        # Set nominal parameters
        param = bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Evaluate at initial condition (flatten from (nstates, 1) to (nstates,))
        state = bkd.flatten(gt.initial_condition)
        f = residual(state)
        assert f.shape == (3,)

    def test_residual_jacobian(self, bkd) -> None:
        """Test residual Jacobian has correct shape."""
        benchmark = lotka_volterra_3species(bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        param = bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Flatten from (nstates, 1) to (nstates,)
        state = bkd.flatten(gt.initial_condition)
        jac = residual.jacobian(state)
        assert jac.shape == (3, 3)

    def test_residual_param_jacobian(self, bkd) -> None:
        """Test residual parameter Jacobian has correct shape."""
        benchmark = lotka_volterra_3species(bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        param = bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Flatten from (nstates, 1) to (nstates,)
        state = bkd.flatten(gt.initial_condition)
        pjac = residual.param_jacobian(state)
        assert pjac.shape == (3, 12)


class TestCoupledSprings2MassBenchmark:
    """Tests for coupled_springs_2mass benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = coupled_springs_2mass(bkd)
        assert benchmark.name() == "coupled_springs_2mass"

    def test_nstates(self, bkd) -> None:
        """Test number of states."""
        benchmark = coupled_springs_2mass(bkd)
        assert benchmark.nstates() == 4

    def test_nparams(self, bkd) -> None:
        """Test number of parameters."""
        benchmark = coupled_springs_2mass(bkd)
        assert benchmark.nparams() == 12

    def test_domain_nvars(self, bkd) -> None:
        """Test domain nvars matches nparams."""
        benchmark = coupled_springs_2mass(bkd)
        assert benchmark.domain().nvars() == 12

    def test_time_config(self, bkd) -> None:
        """Test time configuration."""
        benchmark = coupled_springs_2mass(bkd)
        tc = benchmark.time_config()
        assert tc.init_time == 0.0
        assert tc.final_time == 10.0
        assert tc.deltat == 0.1
        assert tc.ntimes() == 101

    def test_ground_truth_nstates(self, bkd) -> None:
        """Test ground truth nstates."""
        benchmark = coupled_springs_2mass(bkd)
        gt = benchmark.ground_truth()
        assert gt.nstates == 4

    def test_prior_nvars(self, bkd) -> None:
        """Test prior has correct nvars."""
        benchmark = coupled_springs_2mass(bkd)
        prior = benchmark.prior()
        assert prior.nvars() == 12

    def test_residual_evaluation(self, bkd) -> None:
        """Test residual can be evaluated at valid states."""
        benchmark = coupled_springs_2mass(bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        param = bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Flatten from (nstates, 1) to (nstates,)
        state = bkd.flatten(gt.initial_condition)
        f = residual(state)
        assert f.shape == (4,)


class TestHastingsEcology3SpeciesBenchmark:
    """Tests for hastings_ecology_3species benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = hastings_ecology_3species(bkd)
        assert benchmark.name() == "hastings_ecology_3species"

    def test_nstates(self, bkd) -> None:
        """Test number of states."""
        benchmark = hastings_ecology_3species(bkd)
        assert benchmark.nstates() == 3

    def test_nparams(self, bkd) -> None:
        """Test number of parameters."""
        benchmark = hastings_ecology_3species(bkd)
        assert benchmark.nparams() == 9

    def test_time_config(self, bkd) -> None:
        """Test time configuration."""
        benchmark = hastings_ecology_3species(bkd)
        tc = benchmark.time_config()
        assert tc.init_time == 0.0
        assert tc.final_time == 100.0
        assert tc.deltat == 2.5
        assert tc.ntimes() == 41

    def test_ground_truth_nstates(self, bkd) -> None:
        """Test ground truth nstates."""
        benchmark = hastings_ecology_3species(bkd)
        gt = benchmark.ground_truth()
        assert gt.nstates == 3

    def test_ground_truth_reference(self, bkd) -> None:
        """Test benchmark reference."""
        benchmark = hastings_ecology_3species(bkd)
        assert "Hastings" in benchmark.reference()

    def test_prior_nvars(self, bkd) -> None:
        """Test prior has correct nvars."""
        benchmark = hastings_ecology_3species(bkd)
        prior = benchmark.prior()
        assert prior.nvars() == 9

    def test_residual_evaluation(self, bkd) -> None:
        """Test residual can be evaluated at valid states."""
        benchmark = hastings_ecology_3species(bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        param = bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Flatten from (nstates, 1) to (nstates,)
        state = bkd.flatten(gt.initial_condition)
        f = residual(state)
        assert f.shape == (3,)


class TestChemicalReactionSurfaceBenchmark:
    """Tests for chemical_reaction_surface benchmark instance."""

    def test_name(self, bkd) -> None:
        """Test benchmark name."""
        benchmark = chemical_reaction_surface(bkd)
        assert benchmark.name() == "chemical_reaction_surface"

    def test_nstates(self, bkd) -> None:
        """Test number of states."""
        benchmark = chemical_reaction_surface(bkd)
        assert benchmark.nstates() == 3

    def test_nparams(self, bkd) -> None:
        """Test number of parameters."""
        benchmark = chemical_reaction_surface(bkd)
        assert benchmark.nparams() == 6

    def test_time_config(self, bkd) -> None:
        """Test time configuration."""
        benchmark = chemical_reaction_surface(bkd)
        tc = benchmark.time_config()
        assert tc.init_time == 0.0
        assert tc.final_time == 100.0
        assert tc.deltat == 0.1
        assert tc.ntimes() == 1001

    def test_ground_truth_initial_condition_zeros(self, bkd) -> None:
        """Test initial condition is zeros (empty surface) with shape (nstates, 1)."""
        benchmark = chemical_reaction_surface(bkd)
        gt = benchmark.ground_truth()
        expected = bkd.array([[0.0], [0.0], [0.0]])
        bkd.assert_allclose(gt.initial_condition, expected, atol=1e-14)

    def test_ground_truth_reference(self, bkd) -> None:
        """Test benchmark reference."""
        benchmark = chemical_reaction_surface(bkd)
        assert "Vigil" in benchmark.reference()

    def test_prior_nvars(self, bkd) -> None:
        """Test prior has correct nvars."""
        benchmark = chemical_reaction_surface(bkd)
        prior = benchmark.prior()
        assert prior.nvars() == 6

    def test_residual_evaluation(self, bkd) -> None:
        """Test residual can be evaluated at valid states."""
        benchmark = chemical_reaction_surface(bkd)
        residual = benchmark.residual()
        gt = benchmark.ground_truth()

        param = bkd.asarray(gt.nominal_parameters)
        residual.set_param(param)

        # Flatten from (nstates, 1) to (nstates,)
        state = bkd.flatten(gt.initial_condition)
        f = residual(state)
        assert f.shape == (3,)


class TestODEBenchmarkRegistry:
    """Test registry for ODE benchmarks."""

    def test_lotka_volterra_registered(self) -> None:
        """Test lotka_volterra_3species is registered."""
        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("lotka_volterra_3species", bkd)
        assert benchmark.name() == "lotka_volterra_3species"

    def test_coupled_springs_registered(self) -> None:
        """Test coupled_springs_2mass is registered."""
        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("coupled_springs_2mass", bkd)
        assert benchmark.name() == "coupled_springs_2mass"

    def test_hastings_ecology_registered(self) -> None:
        """Test hastings_ecology_3species is registered."""
        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("hastings_ecology_3species", bkd)
        assert benchmark.name() == "hastings_ecology_3species"

    def test_chemical_reaction_registered(self) -> None:
        """Test chemical_reaction_surface is registered."""
        bkd = NumpyBkd()
        benchmark = BenchmarkRegistry.get("chemical_reaction_surface", bkd)
        assert benchmark.name() == "chemical_reaction_surface"

    def test_ode_category(self) -> None:
        """Test all ODE benchmarks are in ode category."""
        category_benchmarks = BenchmarkRegistry.list_category("ode")
        assert "lotka_volterra_3species" in category_benchmarks
        assert "coupled_springs_2mass" in category_benchmarks
        assert "hastings_ecology_3species" in category_benchmarks
        assert "chemical_reaction_surface" in category_benchmarks
