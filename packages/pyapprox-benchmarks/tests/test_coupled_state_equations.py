"""Tests for the coupled-state-equations forward UQ problem."""

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.optimization.implicitfunction.functionals.mean_squared_error import (
    MSEFunctional,
)
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox_benchmarks.problems.coupled_state_equations import (
    CoupledStateEquationsForwardUQ,
)


def _exact_qoi(a, b):
    """u_2 = sqrt((1-a)/(1+ab)) on the positive solution branch."""
    return np.sqrt((1.0 - a) / (1.0 + a * b))


def _exact_states(a, b):
    """(u_1, u_2) on the positive solution branch."""
    ratio = (1.0 - a) / (1.0 + a * b)
    return np.array([np.sqrt(1.0 + b * ratio), np.sqrt(ratio)])


class TestCoupledStateEquationsForwardUQ:
    def test_problem_metadata(self, numpy_bkd):
        problem = CoupledStateEquationsForwardUQ(numpy_bkd).problem()
        assert problem.name() == "coupled_state_equations"
        assert problem.description() != ""

    def test_function_satisfies_protocol(self, numpy_bkd):
        function = CoupledStateEquationsForwardUQ(numpy_bkd).problem().function()
        assert isinstance(function, FunctionProtocol)
        assert function.nvars() == 2
        assert function.nqoi() == 1

    def test_values_match_closed_form_on_prior_samples(self, numpy_bkd):
        problem = CoupledStateEquationsForwardUQ(numpy_bkd).problem()
        samples = problem.prior().rvs(8)
        values = problem.function()(samples)
        assert values.shape == (1, 8)
        samples_np = numpy_bkd.to_numpy(samples)
        expected = numpy_bkd.asarray(
            np.array(
                [
                    [
                        _exact_qoi(samples_np[0, ii], samples_np[1, ii])
                        for ii in range(samples_np.shape[1])
                    ]
                ]
            )
        )
        numpy_bkd.assert_allclose(values, expected, atol=1e-6)

    def test_prior_support_stays_inside_the_real_branch(self, numpy_bkd):
        """A real solution requires a < 1; the prior must not reach it."""
        prior = CoupledStateEquationsForwardUQ(numpy_bkd).problem().prior()
        samples = numpy_bkd.to_numpy(prior.rvs(200))
        assert samples[0].max() < 1.0
        assert samples[1].min() > 0.0

    def test_declares_adjoint_derivatives(self, numpy_bkd):
        function = CoupledStateEquationsForwardUQ(numpy_bkd).problem().function()
        derivatives = function.derivatives()
        assert derivatives.jacobian is not None
        assert derivatives.hvp is not None

    def test_derivatives_pass_finite_difference_check(self, numpy_bkd):
        function = CoupledStateEquationsForwardUQ(numpy_bkd).problem().function()
        checker = DerivativeChecker(function)
        errors = checker.check_derivatives(numpy_bkd.asarray([[0.5], [1.228]]))
        for error in errors:
            assert numpy_bkd.to_float(checker.error_ratio(error)) <= 1e-5

    def test_supplied_functional_reads_the_other_state(self, numpy_bkd):
        """Weights [1, 0] read u_1, which the closed form puts above 1."""
        functional = WeightedSumFunctional(
            numpy_bkd.asarray([[1.0], [0.0]]), 2, numpy_bkd
        )
        problem = CoupledStateEquationsForwardUQ(
            numpy_bkd, functional=functional
        ).problem()
        sample = numpy_bkd.asarray([[0.5], [1.228]])
        value = problem.function()(sample)
        expected = np.sqrt(1.0 + 1.228 * _exact_qoi(0.5, 1.228) ** 2)
        numpy_bkd.assert_allclose(
            value, numpy_bkd.asarray([[expected]]), atol=1e-6
        )

    def test_least_squares_functional_has_an_interior_minimum(self, numpy_bkd):
        """A misfit against states attainable inside the box turns the
        same solve into an estimation problem with an isolated minimum
        and a positive definite reduced Hessian."""
        target = numpy_bkd.asarray([[0.5], [1.0]])
        functional = MSEFunctional(2, 2, numpy_bkd)
        functional.set_observations(
            numpy_bkd.asarray(_exact_states(0.5, 1.0)[:, None])
        )
        function = CoupledStateEquationsForwardUQ(
            numpy_bkd, functional=functional
        ).problem().function()
        # the misfit vanishes and the gradient with it
        numpy_bkd.assert_allclose(
            function(target), numpy_bkd.zeros((1, 1)), atol=1e-12
        )
        numpy_bkd.assert_allclose(
            function.jacobian(target), numpy_bkd.zeros((1, 2)), atol=1e-6
        )
        hessian = numpy_bkd.hstack(
            [
                function.hvp(target, numpy_bkd.asarray([[1.0], [0.0]])),
                function.hvp(target, numpy_bkd.asarray([[0.0], [1.0]])),
            ]
        )
        eigvals = np.linalg.eigvalsh(numpy_bkd.to_numpy(hessian))
        assert eigvals.min() > 0.0

    def test_rejects_functional_without_second_order_terms(self, numpy_bkd):
        class _JacobianOnly:
            def nqoi(self):
                return 1

        with pytest.raises(TypeError, match="ParameterizedFunctional"):
            CoupledStateEquationsForwardUQ(numpy_bkd, functional=_JacobianOnly())
