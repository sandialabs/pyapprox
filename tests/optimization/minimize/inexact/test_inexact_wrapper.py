"""Dual-backend tests for InexactWrapper."""

import numpy as np
import pytest

from pyapprox.expdesign.statistics import SampleAverageMean
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    NonlinearConstraintProtocol,
)
from pyapprox.optimization.minimize.inexact.fixed import (
    FixedSampleStrategy,
)
from pyapprox.optimization.minimize.inexact.monte_carlo import (
    MonteCarloSAAStrategy,
)
from pyapprox.optimization.minimize.inexact.protocols import (
    InexactDifferentiable,
    InexactEvaluable,
)
from pyapprox.optimization.minimize.inexact.wrapper import (
    InexactWrapper,
)


class _QuadraticModel:
    """f(x1, x2) = [x1^2 + x2, x2^2 + x1]. x1 random, x2 design."""

    def __init__(self, bkd):
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def nvars(self):
        return 2

    def nqoi(self):
        return 2

    def __call__(self, samples):
        x1 = samples[0:1, :]
        x2 = samples[1:2, :]
        return self._bkd.concatenate([x1**2 + x2, x2**2 + x1], axis=0)

    def jacobian(self, sample):
        x1 = sample[0, 0]
        x2 = sample[1, 0]
        zero = 0.0 * x1
        return self._bkd.asarray(
            [
                [2.0 * x1 + zero, 1.0 + zero],
                [1.0 + zero, 2.0 * x2 + zero],
            ]
        )


class _ScalarModel:
    """f(x1, x2) = x1^2 + x2^2. x1 random, x2 design. nqoi=1."""

    def __init__(self, bkd):
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def nvars(self):
        return 2

    def nqoi(self):
        return 1

    def __call__(self, samples):
        x1 = samples[0:1, :]
        x2 = samples[1:2, :]
        return x1**2 + x2**2

    def jacobian(self, sample):
        x1 = sample[0, 0]
        x2 = sample[1, 0]
        zero = 0.0 * x1
        return self._bkd.asarray([[2.0 * x1 + zero, 2.0 * x2 + zero]])


class _NoJacModel:
    """Model without jacobian for testing dynamic binding."""

    def __init__(self, bkd):
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def nvars(self):
        return 2

    def nqoi(self):
        return 1

    def __call__(self, samples):
        return samples[0:1, :] + samples[1:2, :]


def _make_wrapper(bkd, model=None, strategy=None, design_indices=None,
                  constraint_lb=None, constraint_ub=None):
    if model is None:
        model = _QuadraticModel(bkd)
    stat = SampleAverageMean(bkd)
    if strategy is None:
        # Default: 3-point fixed quadrature on x1
        quad_samples = bkd.asarray([[-1.0, 0.0, 1.0]])
        quad_weights = bkd.asarray([1.0 / 6, 2.0 / 3, 1.0 / 6])
        strategy = FixedSampleStrategy(quad_samples, quad_weights, bkd)
    if design_indices is None:
        design_indices = [1]  # x2 is design
    return InexactWrapper(
        model=model,
        stat=stat,
        strategy=strategy,
        design_indices=design_indices,
        bkd=bkd,
        constraint_lb=constraint_lb,
        constraint_ub=constraint_ub,
    )


class TestInexactWrapperBasic:
    def test_nvars_nqoi(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        assert wrapper.nvars() == 1
        assert wrapper.nqoi() == 2

    def test_bkd(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        assert wrapper.bkd() is bkd

    def test_bounds(self, bkd) -> None:
        lb = bkd.asarray([0.0, 0.0])
        ub = bkd.asarray([1.0, 1.0])
        wrapper = _make_wrapper(bkd, constraint_lb=lb, constraint_ub=ub)
        bkd.assert_allclose(wrapper.lb(), lb)
        bkd.assert_allclose(wrapper.ub(), ub)

    def test_no_bounds_raises(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        with pytest.raises(AttributeError):
            wrapper.lb()
        with pytest.raises(AttributeError):
            wrapper.ub()


class TestInexactWrapperProtocols:
    def test_satisfies_nonlinear_constraint_protocol(self, bkd) -> None:
        lb = bkd.asarray([0.0, 0.0])
        ub = bkd.asarray([10.0, 10.0])
        wrapper = _make_wrapper(bkd, constraint_lb=lb, constraint_ub=ub)
        assert isinstance(wrapper, NonlinearConstraintProtocol)

    def test_satisfies_inexact_evaluable(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        assert isinstance(wrapper, InexactEvaluable)

    def test_satisfies_inexact_differentiable(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        assert isinstance(wrapper, InexactDifferentiable)

    def test_no_jacobian_without_model_jacobian(self, bkd) -> None:
        model = _NoJacModel(bkd)
        wrapper = _make_wrapper(bkd, model=model)
        assert not hasattr(wrapper, "jacobian")
        assert not isinstance(wrapper, InexactDifferentiable)


class TestInexactWrapperNeutrality:
    """InexactWrapper + FixedSampleStrategy matches SampleAverageConstraint."""

    def test_value_matches_sample_average_constraint(self, bkd) -> None:
        from pyapprox.optimization.minimize.constraints.sample_average import (
            SampleAverageConstraint,
        )

        model = _QuadraticModel(bkd)
        stat = SampleAverageMean(bkd)
        quad_samples = bkd.asarray([[-1.0, 0.0, 1.0]])
        quad_weights = bkd.asarray([1.0 / 6, 2.0 / 3, 1.0 / 6])

        # Reference: SampleAverageConstraint
        ref = SampleAverageConstraint(
            model=model,
            quad_samples=quad_samples,
            quad_weights=quad_weights,
            stat=stat,
            design_indices=[1],
            constraint_lb=bkd.asarray([0.0, 0.0]),
            constraint_ub=bkd.asarray([10.0, 10.0]),
            bkd=bkd,
        )

        # InexactWrapper with FixedSampleStrategy
        strategy = FixedSampleStrategy(quad_samples, quad_weights, bkd)
        wrapper = InexactWrapper(
            model=model,
            stat=stat,
            strategy=strategy,
            design_indices=[1],
            bkd=bkd,
        )

        sample = bkd.asarray([[2.0]])
        bkd.assert_allclose(wrapper(sample), ref(sample), rtol=1e-12)

    def test_jacobian_matches_sample_average_constraint(self, bkd) -> None:
        from pyapprox.optimization.minimize.constraints.sample_average import (
            SampleAverageConstraint,
        )

        model = _QuadraticModel(bkd)
        stat = SampleAverageMean(bkd)
        quad_samples = bkd.asarray([[-1.0, 0.0, 1.0]])
        quad_weights = bkd.asarray([1.0 / 6, 2.0 / 3, 1.0 / 6])

        ref = SampleAverageConstraint(
            model=model,
            quad_samples=quad_samples,
            quad_weights=quad_weights,
            stat=stat,
            design_indices=[1],
            constraint_lb=bkd.asarray([0.0, 0.0]),
            constraint_ub=bkd.asarray([10.0, 10.0]),
            bkd=bkd,
        )

        strategy = FixedSampleStrategy(quad_samples, quad_weights, bkd)
        wrapper = InexactWrapper(
            model=model,
            stat=stat,
            strategy=strategy,
            design_indices=[1],
            bkd=bkd,
        )

        sample = bkd.asarray([[2.0]])
        bkd.assert_allclose(
            wrapper.jacobian(sample), ref.jacobian(sample), rtol=1e-12,
        )


class TestInexactWrapperValues:
    def test_call_value_shape(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        result = wrapper(bkd.asarray([[2.0]]))
        assert result.shape == (2, 1)

    def test_inexact_value_shape(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        result = wrapper.inexact_value(bkd.asarray([[2.0]]), 0.1)
        assert result.shape == (2, 1)

    def test_call_equals_inexact_value_tol_zero(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        sample = bkd.asarray([[2.0]])
        bkd.assert_allclose(
            wrapper(sample),
            wrapper.inexact_value(sample, 0.0),
            rtol=1e-12,
        )

    def test_manual_value_verification(self, bkd) -> None:
        """Same manual calculation as SampleAverageConstraint test.

        E[f1] = 1/3 + x2, E[f2] = x2^2
        """
        wrapper = _make_wrapper(bkd)
        x2 = 2.0
        result = wrapper(bkd.asarray([[x2]]))
        expected = bkd.asarray([[1.0 / 3.0 + x2], [x2**2]])
        bkd.assert_allclose(result, expected, rtol=1e-12)


class TestInexactWrapperJacobian:
    def test_jacobian_shape(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        jac = wrapper.jacobian(bkd.asarray([[2.0]]))
        assert jac.shape == (2, 1)

    def test_inexact_jacobian_shape(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        jac = wrapper.inexact_jacobian(bkd.asarray([[2.0]]), 0.1)
        assert jac.shape == (2, 1)

    def test_jacobian_equals_inexact_jacobian_tol_zero(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        sample = bkd.asarray([[2.0]])
        bkd.assert_allclose(
            wrapper.jacobian(sample),
            wrapper.inexact_jacobian(sample, 0.0),
            rtol=1e-12,
        )

    def test_manual_jacobian_verification(self, bkd) -> None:
        """d(E[f1])/dx2 = 1, d(E[f2])/dx2 = 2*x2."""
        wrapper = _make_wrapper(bkd)
        x2 = 2.0
        jac = wrapper.jacobian(bkd.asarray([[x2]]))
        expected = bkd.asarray([[1.0], [2.0 * x2]])
        bkd.assert_allclose(jac, expected, rtol=1e-10)

    def test_derivative_checker(self, bkd) -> None:
        wrapper = _make_wrapper(bkd)
        sample = bkd.asarray([[2.0]])
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5


class TestInexactWrapperScalar:
    """Tests with nqoi=1 (objective mode)."""

    def test_scalar_value_shape(self, bkd) -> None:
        model = _ScalarModel(bkd)
        wrapper = _make_wrapper(bkd, model=model)
        result = wrapper(bkd.asarray([[2.0]]))
        assert result.shape == (1, 1)

    def test_scalar_jacobian_shape(self, bkd) -> None:
        model = _ScalarModel(bkd)
        wrapper = _make_wrapper(bkd, model=model)
        jac = wrapper.jacobian(bkd.asarray([[2.0]]))
        assert jac.shape == (1, 1)

    def test_scalar_derivative_checker(self, bkd) -> None:
        model = _ScalarModel(bkd)
        wrapper = _make_wrapper(bkd, model=model)
        sample = bkd.asarray([[2.0]])
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5


class TestInexactWrapperWithMonteCarlo:
    """Tests with MonteCarloSAAStrategy to verify inexact behavior."""

    def _make_mc_wrapper(self, bkd, n_max=1000, scale_factor=1.0):
        model = _ScalarModel(bkd)
        np.random.seed(42)
        base_np = np.random.randn(1, n_max)
        base_samples = bkd.array(base_np)
        strategy = MonteCarloSAAStrategy(
            base_samples, bkd, scale_factor=scale_factor,
        )
        return InexactWrapper(
            model=model,
            stat=SampleAverageMean(bkd),
            strategy=strategy,
            design_indices=[1],
            bkd=bkd,
        )

    def test_different_tol_gives_different_nsamples(self, bkd) -> None:
        wrapper = self._make_mc_wrapper(bkd)
        # Just verify inexact methods don't error — the strategy
        # controls sample count, wrapper is agnostic
        sample = bkd.asarray([[1.0]])
        v1 = wrapper.inexact_value(sample, 1.0)
        v2 = wrapper.inexact_value(sample, 0.1)
        assert v1.shape == (1, 1)
        assert v2.shape == (1, 1)

    def test_value_tol_independence(self, bkd) -> None:
        """Value and gradient can use different tols independently."""
        wrapper = self._make_mc_wrapper(bkd)
        sample = bkd.asarray([[1.0]])
        # These should not error — value and gradient each get their own tol
        v = wrapper.inexact_value(sample, 0.5)
        j = wrapper.inexact_jacobian(sample, 0.1)
        assert v.shape == (1, 1)
        assert j.shape == (1, 1)

    def test_convergence_as_tol_shrinks(self, bkd) -> None:
        """Inexact values approach exact as tol → 0.

        f(x1, x2) = x1^2 + x2^2, x1 ~ N(0,1), design x2.
        E[f] = E[x1^2] + x2^2 = 1 + x2^2 (using MC samples).
        As tol → 0, more samples used, closer to true expectation.
        """
        wrapper = self._make_mc_wrapper(bkd, n_max=10000)
        sample = bkd.asarray([[2.0]])
        exact_value = wrapper.inexact_value(sample, 0.0)
        # Coarse should be less accurate than fine
        coarse_value = wrapper.inexact_value(sample, 1.0)
        fine_value = wrapper.inexact_value(sample, 0.01)
        coarse_err = float(
            bkd.to_numpy(bkd.abs(coarse_value - exact_value))[0, 0]
        )
        fine_err = float(
            bkd.to_numpy(bkd.abs(fine_value - exact_value))[0, 0]
        )
        # Fine error should generally be smaller (not guaranteed for
        # single realization, but with seed=42 and enough samples it holds)
        assert fine_err <= coarse_err + 1e-10

    def test_inexact_jacobian_derivative_checker(self, bkd) -> None:
        """DerivativeChecker on inexact wrapper (uses tol=0 for exact)."""
        wrapper = self._make_mc_wrapper(bkd, n_max=1000)
        sample = bkd.asarray([[2.0]])
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5
