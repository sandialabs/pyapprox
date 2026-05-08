"""Tests for ODE forward UQ problems."""

from pyapprox_benchmarks.ode import (
    build_chemical_reaction_surface,
    build_coupled_springs_2mass,
    build_hastings_ecology_3species,
    build_lotka_volterra_3species,
)


class TestLotkaVolterra3Species:
    """Tests for build_lotka_volterra_3species."""

    def test_name(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        assert problem.name() == "lotka_volterra_3species"

    def test_nstates(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        assert problem.nstates() == 3

    def test_nparams(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        assert problem.nparams() == 12

    def test_domain_nvars(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        assert problem.domain().nvars() == 12

    def test_domain_bounds(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        bounds = problem.domain().bounds()
        expected = bkd.array([[0.3, 0.7]] * 12)
        bkd.assert_allclose(bounds, expected, atol=1e-14)

    def test_initial_condition(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        expected = bkd.array([[0.3], [0.4], [0.3]])
        bkd.assert_allclose(
            problem.initial_condition(), expected, atol=1e-14,
        )

    def test_time_config(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        tc = problem.time_config()
        assert tc.init_time == 0.0
        assert tc.final_time == 10.0
        assert tc.deltat == 1.0
        assert tc.ntimes() == 11

    def test_prior_nvars(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        assert problem.prior().nvars() == 12

    def test_prior_samples_in_domain(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        prior = problem.prior()
        bounds = problem.domain().bounds()

        samples = prior.rvs(100)
        assert samples.shape == (12, 100)

        for i in range(12):
            assert bkd.all_bool(samples[i, :] >= bounds[i, 0])
            assert bkd.all_bool(samples[i, :] <= bounds[i, 1])

    def test_residual_evaluation(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        residual = problem.residual()

        param = bkd.asarray(problem.nominal_parameters())
        residual.set_param(param)

        state = bkd.flatten(problem.initial_condition())
        f = residual(state)
        assert f.shape == (3,)

    def test_residual_jacobian(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        residual = problem.residual()

        param = bkd.asarray(problem.nominal_parameters())
        residual.set_param(param)

        state = bkd.flatten(problem.initial_condition())
        jac = residual.jacobian(state)
        assert jac.shape == (3, 3)

    def test_residual_param_jacobian(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        residual = problem.residual()

        param = bkd.asarray(problem.nominal_parameters())
        residual.set_param(param)

        state = bkd.flatten(problem.initial_condition())
        pjac = residual.param_jacobian(state)
        assert pjac.shape == (3, 12)

    def test_function(self, bkd) -> None:
        problem = build_lotka_volterra_3species(bkd)
        func = problem.function()
        assert func.nqoi() == 3
        assert func.nvars() == 12


class TestCoupledSprings2Mass:
    """Tests for build_coupled_springs_2mass."""

    def test_name(self, bkd) -> None:
        problem = build_coupled_springs_2mass(bkd)
        assert problem.name() == "coupled_springs_2mass"

    def test_nstates(self, bkd) -> None:
        problem = build_coupled_springs_2mass(bkd)
        assert problem.nstates() == 4

    def test_nparams(self, bkd) -> None:
        problem = build_coupled_springs_2mass(bkd)
        assert problem.nparams() == 12

    def test_domain_nvars(self, bkd) -> None:
        problem = build_coupled_springs_2mass(bkd)
        assert problem.domain().nvars() == 12

    def test_time_config(self, bkd) -> None:
        problem = build_coupled_springs_2mass(bkd)
        tc = problem.time_config()
        assert tc.init_time == 0.0
        assert tc.final_time == 10.0
        assert tc.deltat == 0.1
        assert tc.ntimes() == 101

    def test_prior_nvars(self, bkd) -> None:
        problem = build_coupled_springs_2mass(bkd)
        assert problem.prior().nvars() == 12

    def test_residual_evaluation(self, bkd) -> None:
        problem = build_coupled_springs_2mass(bkd)
        residual = problem.residual()

        param = bkd.asarray(problem.nominal_parameters())
        residual.set_param(param)

        state = bkd.flatten(problem.initial_condition())
        f = residual(state)
        assert f.shape == (4,)


class TestHastingsEcology3Species:
    """Tests for build_hastings_ecology_3species."""

    def test_name(self, bkd) -> None:
        problem = build_hastings_ecology_3species(bkd)
        assert problem.name() == "hastings_ecology_3species"

    def test_nstates(self, bkd) -> None:
        problem = build_hastings_ecology_3species(bkd)
        assert problem.nstates() == 3

    def test_nparams(self, bkd) -> None:
        problem = build_hastings_ecology_3species(bkd)
        assert problem.nparams() == 9

    def test_time_config(self, bkd) -> None:
        problem = build_hastings_ecology_3species(bkd)
        tc = problem.time_config()
        assert tc.init_time == 0.0
        assert tc.final_time == 100.0
        assert tc.deltat == 2.5
        assert tc.ntimes() == 41

    def test_reference(self, bkd) -> None:
        problem = build_hastings_ecology_3species(bkd)
        assert "Hastings" in problem.reference()

    def test_prior_nvars(self, bkd) -> None:
        problem = build_hastings_ecology_3species(bkd)
        assert problem.prior().nvars() == 9

    def test_residual_evaluation(self, bkd) -> None:
        problem = build_hastings_ecology_3species(bkd)
        residual = problem.residual()

        param = bkd.asarray(problem.nominal_parameters())
        residual.set_param(param)

        state = bkd.flatten(problem.initial_condition())
        f = residual(state)
        assert f.shape == (3,)


class TestChemicalReactionSurface:
    """Tests for build_chemical_reaction_surface."""

    def test_name(self, bkd) -> None:
        problem = build_chemical_reaction_surface(bkd)
        assert problem.name() == "chemical_reaction_surface"

    def test_nstates(self, bkd) -> None:
        problem = build_chemical_reaction_surface(bkd)
        assert problem.nstates() == 3

    def test_nparams(self, bkd) -> None:
        problem = build_chemical_reaction_surface(bkd)
        assert problem.nparams() == 6

    def test_time_config(self, bkd) -> None:
        problem = build_chemical_reaction_surface(bkd)
        tc = problem.time_config()
        assert tc.init_time == 0.0
        assert tc.final_time == 100.0
        assert tc.deltat == 0.1
        assert tc.ntimes() == 1001

    def test_initial_condition_zeros(self, bkd) -> None:
        problem = build_chemical_reaction_surface(bkd)
        expected = bkd.array([[0.0], [0.0], [0.0]])
        bkd.assert_allclose(
            problem.initial_condition(), expected, atol=1e-14,
        )

    def test_reference(self, bkd) -> None:
        problem = build_chemical_reaction_surface(bkd)
        assert "Vigil" in problem.reference()

    def test_prior_nvars(self, bkd) -> None:
        problem = build_chemical_reaction_surface(bkd)
        assert problem.prior().nvars() == 6

    def test_residual_evaluation(self, bkd) -> None:
        problem = build_chemical_reaction_surface(bkd)
        residual = problem.residual()

        param = bkd.asarray(problem.nominal_parameters())
        residual.set_param(param)

        state = bkd.flatten(problem.initial_condition())
        f = residual(state)
        assert f.shape == (3,)
