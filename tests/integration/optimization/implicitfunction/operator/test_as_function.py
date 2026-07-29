"""Tests for exposing an implicitly constrained QoI as a function."""

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.optimization.implicitfunction.operator.as_function import (
    ImplicitFunctionOfParameters,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_jacobian import (
    AdjointOperatorWithJacobian,
)
from pyapprox_benchmarks.functions.algebraic.wildeys_nonlinear_state_equation import (
    NonLinearCoupledStateEquations,
)


def _exact_qoi(a, b):
    """u_2 = sqrt((1-a)/(1+ab)) for the coupled state equations."""
    return np.sqrt((1.0 - a) / (1.0 + a * b))


def _exact_gradient(a, b):
    """Analytic gradient of the closed-form QoI."""
    ratio = (1.0 - a) / (1.0 + a * b)
    dr_da = (-(1.0 + a * b) - (1.0 - a) * b) / (1.0 + a * b) ** 2
    dr_db = -(1.0 - a) * a / (1.0 + a * b) ** 2
    return np.array([dr_da, dr_db]) / (2.0 * np.sqrt(ratio))


def _build(bkd, second_order=True):
    state_eq = NonLinearCoupledStateEquations(bkd)
    functional = WeightedSumFunctional(
        bkd.asarray([[0.0], [1.0]]), state_eq.nparams(), bkd
    )
    operator = (
        AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        if second_order
        else AdjointOperatorWithJacobian(state_eq, functional)
    )
    init_state = bkd.full((state_eq.nstates(), 1), 0.75)
    return ImplicitFunctionOfParameters(operator, init_state)


class TestImplicitFunctionOfParameters:
    def test_satisfies_function_protocol(self, numpy_bkd):
        assert isinstance(_build(numpy_bkd), FunctionProtocol)

    def test_dimensions(self, numpy_bkd):
        function = _build(numpy_bkd)
        assert function.nvars() == 2
        assert function.nqoi() == 1

    def test_batch_values_match_closed_form(self, numpy_bkd):
        function = _build(numpy_bkd)
        samples = numpy_bkd.asarray([[0.5, 0.7, 0.3], [1.228, 0.9, 1.5]])
        values = function(samples)
        assert values.shape == (1, 3)
        expected = numpy_bkd.asarray(
            [[_exact_qoi(0.5, 1.228), _exact_qoi(0.7, 0.9), _exact_qoi(0.3, 1.5)]]
        )
        numpy_bkd.assert_allclose(values, expected, atol=1e-7)

    def test_jacobian_matches_closed_form(self, numpy_bkd):
        function = _build(numpy_bkd)
        sample = numpy_bkd.asarray([[0.5], [1.228]])
        jacobian = function.jacobian(sample)
        assert jacobian.shape == (1, 2)
        expected = numpy_bkd.asarray(_exact_gradient(0.5, 1.228)[None, :])
        numpy_bkd.assert_allclose(jacobian, expected, atol=1e-7)

    def test_hessian_is_symmetric(self, numpy_bkd):
        """Symmetry is not enforced by the four-solve recipe, so agreement
        to machine precision checks every term and sign."""
        function = _build(numpy_bkd)
        sample = numpy_bkd.asarray([[0.5], [1.228]])
        vec_u = numpy_bkd.asarray([[0.6], [-0.8]])
        vec_v = numpy_bkd.asarray([[0.3], [0.95]])
        hu_v = numpy_bkd.sum(function.hvp(sample, vec_u) * vec_v)
        hv_u = numpy_bkd.sum(function.hvp(sample, vec_v) * vec_u)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([hu_v]), numpy_bkd.asarray([hv_u]), atol=1e-12
        )

    def test_derivative_checker_passes(self, numpy_bkd):
        function = _build(numpy_bkd)
        checker = DerivativeChecker(function)
        errors = checker.check_derivatives(numpy_bkd.asarray([[0.5], [1.228]]))
        for error in errors:
            assert numpy_bkd.to_float(checker.error_ratio(error)) <= 1e-5

    def test_bundle_declares_second_order_only_when_supported(self, numpy_bkd):
        with_hvp = _build(numpy_bkd, second_order=True).derivatives()
        assert with_hvp.jacobian is not None
        assert with_hvp.hvp is not None
        without_hvp = _build(numpy_bkd, second_order=False).derivatives()
        assert without_hvp.jacobian is not None
        assert without_hvp.hvp is None

    def test_hvp_raises_without_second_order_operator(self, numpy_bkd):
        function = _build(numpy_bkd, second_order=False)
        sample = numpy_bkd.asarray([[0.5], [1.228]])
        with pytest.raises(RuntimeError, match="AdjointOperatorWithJacobianAndHVP"):
            function.hvp(sample, sample)

    def test_rejects_wrong_shaped_init_state(self, numpy_bkd):
        state_eq = NonLinearCoupledStateEquations(numpy_bkd)
        functional = WeightedSumFunctional(
            numpy_bkd.asarray([[0.0], [1.0]]), state_eq.nparams(), numpy_bkd
        )
        operator = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        with pytest.raises(ValueError):
            ImplicitFunctionOfParameters(operator, numpy_bkd.full((3, 1), 0.75))
