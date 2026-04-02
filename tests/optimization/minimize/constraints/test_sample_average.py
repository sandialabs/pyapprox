"""Tests for SampleAverageConstraint.

Tests cover:
- Value correctness against manual computation
- Jacobian verification via DerivativeChecker
- Dynamic binding (jacobian present/absent)
- Integration with cantilever beam model + Gauss quadrature
- Dual-backend testing (NumPy and PyTorch)
"""


class _QuadraticModel:
    """Simple quadratic model: f(x1, x2) = [x1^2 + x2, x2^2 + x1].

    x1 is "random", x2 is "design". nvars=2, nqoi=2.
    """

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


class TestSampleAverageConstraint:
    """Base test class for SampleAverageConstraint."""

    def _make_constraint(
        self,
        bkd,
        model=None,
        quad_samples=None,
        quad_weights=None,
        stat=None,
        design_indices=None,
        constraint_lb=None,
        constraint_ub=None,
    ):
        from pyapprox.expdesign.statistics import SampleAverageMean
        from pyapprox.optimization.minimize.constraints.sample_average import (
            SampleAverageConstraint,
        )

        if model is None:
            model = _QuadraticModel(bkd)
        if quad_samples is None:
            # 3-point quadrature on x1 (random variable)
            quad_samples = bkd.asarray([[-1.0, 0.0, 1.0]])
        if quad_weights is None:
            quad_weights = bkd.asarray([1.0 / 6, 2.0 / 3, 1.0 / 6])
        if stat is None:
            stat = SampleAverageMean(bkd)
        if design_indices is None:
            design_indices = [1]  # x2 is design
        if constraint_lb is None:
            constraint_lb = bkd.asarray([0.0, 0.0])
        if constraint_ub is None:
            constraint_ub = bkd.asarray([float("inf"), float("inf")])

        return SampleAverageConstraint(
            model=model,
            quad_samples=quad_samples,
            quad_weights=quad_weights,
            stat=stat,
            design_indices=design_indices,
            constraint_lb=constraint_lb,
            constraint_ub=constraint_ub,
            bkd=bkd,
        )

    def test_nvars_nqoi(self, bkd):
        """Constraint exposes only design variables."""
        con = self._make_constraint(bkd)
        assert con.nvars() == 1
        assert con.nqoi() == 2

    def test_bounds_shape(self, bkd):
        """Bounds are 1D arrays of shape (nqoi,)."""
        con = self._make_constraint(bkd)
        assert con.lb().shape == (2,)
        assert con.ub().shape == (2,)

    def test_call_manual_verification(self, bkd):
        """Verify constraint value against manual computation.

        Model: f(x1, x2) = [x1^2 + x2, x2^2 + x1]
        Quadrature: 3-point rule on x1 with weights [1/6, 2/3, 1/6]
        Stat: mean

        E[f1] = w1*((-1)^2 + x2) + w2*(0^2 + x2) + w3*(1^2 + x2)
              = (1/6)*(1 + x2) + (2/3)*(x2) + (1/6)*(1 + x2)
              = 1/6 + x2/6 + 2*x2/3 + 1/6 + x2/6
              = 1/3 + x2

        E[f2] = w1*(x2^2 + (-1)) + w2*(x2^2 + 0) + w3*(x2^2 + 1)
              = x2^2 + (1/6)*(-1) + (1/6)*(1)
              = x2^2
        """
        con = self._make_constraint(bkd)

        x2 = 2.0
        sample = bkd.asarray([[x2]])
        result = con(sample)

        expected_f1 = 1.0 / 3.0 + x2
        expected_f2 = x2**2
        expected = bkd.asarray([[expected_f1], [expected_f2]])

        assert result.shape == (2, 1)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_jacobian_manual_verification(self, bkd):
        """Verify jacobian against manual computation.

        d(E[f1])/dx2 = 1
        d(E[f2])/dx2 = 2*x2
        """
        con = self._make_constraint(bkd)

        x2 = 2.0
        sample = bkd.asarray([[x2]])
        jac = con.jacobian(sample)

        expected = bkd.asarray([[1.0], [2.0 * x2]])
        assert jac.shape == (2, 1)
        bkd.assert_allclose(jac, expected, rtol=1e-10)

    def test_jacobian_derivative_checker(self, bkd):
        """Validate Jacobian via DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        con = self._make_constraint(bkd)
        sample = bkd.asarray([[2.0]])

        checker = DerivativeChecker(con)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5

    def test_dynamic_binding_with_jacobian(self, bkd):
        """Model with jacobian + stat with jacobian => constraint has jacobian."""
        con = self._make_constraint(bkd)
        assert hasattr(con, "jacobian")

    def test_dynamic_binding_without_model_jacobian(self, bkd):
        """Model without jacobian => constraint has no jacobian."""
        model = _NoJacModel(bkd)
        con = self._make_constraint(
            bkd,
            model=model,
            constraint_lb=bkd.asarray([0.0]),
            constraint_ub=bkd.asarray([float("inf")]),
        )
        assert not hasattr(con, "jacobian")

    def test_dynamic_binding_without_stat_jacobian(self, bkd):
        """Stat without jacobian => constraint has no jacobian."""

        class NoJacStat:
            def bkd(self):
                return self._bkd

            def jacobian_implemented(self):
                return False

            def __call__(self, values, weights):
                return values[:, 0:1]

        con = self._make_constraint(bkd, stat=NoJacStat())
        assert not hasattr(con, "jacobian")

    def test_satisfies_nonlinear_constraint_protocol(self, bkd):
        """Constraint satisfies NonlinearConstraintProtocol."""
        from pyapprox.optimization.minimize.constraints.protocols import (
            NonlinearConstraintProtocol,
        )

        con = self._make_constraint(bkd)
        assert isinstance(con, NonlinearConstraintProtocol)

    def test_1d_weights_reshaped(self, bkd):
        """1D quad weights are reshaped to (1, n_quad_pts) internally."""
        con = self._make_constraint(
            bkd,
            quad_weights=bkd.asarray([1.0 / 6, 2.0 / 3, 1.0 / 6]),
        )
        # Should work without errors
        sample = bkd.asarray([[2.0]])
        result = con(sample)
        assert result.shape == (2, 1)

    def test_with_mean_plus_stdev_stat(self, bkd):
        """Works with SampleAverageMeanPlusStdev statistic."""
        from pyapprox.expdesign.statistics import (
            SampleAverageMeanPlusStdev,
        )

        stat = SampleAverageMeanPlusStdev(2.0, bkd)
        con = self._make_constraint(bkd, stat=stat)

        sample = bkd.asarray([[2.0]])
        result = con(sample)
        assert result.shape == (2, 1)

        # Jacobian should be available since mean+stdev has jacobian
        assert hasattr(con, "jacobian")
        jac = con.jacobian(sample)
        assert jac.shape == (2, 1)

    def test_with_mean_plus_stdev_derivative_checker(self, bkd):
        """Validate mean+stdev jacobian via DerivativeChecker."""
        from pyapprox.expdesign.statistics import (
            SampleAverageMeanPlusStdev,
        )
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        stat = SampleAverageMeanPlusStdev(2.0, bkd)
        con = self._make_constraint(bkd, stat=stat)
        sample = bkd.asarray([[2.0]])

        checker = DerivativeChecker(con)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5

    def test_multi_design_variable(self, bkd):
        """Works with multiple design variables."""
        # Both variables are design, no random
        con = self._make_constraint(
            bkd,
            design_indices=[0, 1],
            quad_samples=bkd.asarray([]).reshape(0, 1),
            quad_weights=bkd.asarray([1.0]),
        )
        assert con.nvars() == 2

    def test_cantilever_beam_integration(self, bkd):
        """End-to-end test with cantilever beam model + Gauss quadrature.

        Uses CantileverBeam2DConstraints with uncertainty in X and Y
        (random variables) and design variables w and t.
        """
        from pyapprox.benchmarks.functions.algebraic.cantilever_beam_2d import (
            CantileverBeam2DAnalytical,
            CantileverBeam2DConstraints,
        )
        from pyapprox.expdesign.statistics import SampleAverageMean
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        # Build beam constraint model
        beam = CantileverBeam2DAnalytical(length=100.0, bkd=bkd)
        constraints_model = CantileverBeam2DConstraints(beam, 40000.0, 2.2535)

        # Simple 3-point quadrature on X, Y, E, R (random vars, indices 0-3)
        # Just use nominal values as "quadrature points" for simplicity
        quad_samples = bkd.asarray(
            [
                [500.0, 490.0, 510.0],  # X
                [1000.0, 990.0, 1010.0],  # Y
                [2.9e7, 2.9e7, 2.9e7],  # E
                [40000.0, 40000.0, 40000.0],  # R
            ]
        )
        quad_weights = bkd.asarray([1.0 / 3, 1.0 / 3, 1.0 / 3])

        stat = SampleAverageMean(bkd)
        design_indices = [4, 5]  # w and t

        con = self._make_constraint(
            bkd,
            model=constraints_model,
            quad_samples=quad_samples,
            quad_weights=quad_weights,
            stat=stat,
            design_indices=design_indices,
            constraint_lb=bkd.asarray([0.0, 0.0]),
            constraint_ub=bkd.asarray([float("inf"), float("inf")]),
        )

        assert con.nvars() == 2
        assert con.nqoi() == 2

        sample = bkd.asarray([[2.5], [3.0]])
        result = con(sample)
        assert result.shape == (2, 1)

        # Jacobian should work
        assert hasattr(con, "jacobian")
        jac = con.jacobian(sample)
        assert jac.shape == (2, 2)

        # DerivativeChecker
        checker = DerivativeChecker(con)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5
