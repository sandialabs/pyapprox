"""
Tests for ActiveSetFunction.

Tests cover:
- Variable fixing and evaluation
- Jacobian propagation and column extraction
- Dynamic binding (jacobian present/absent)
- Dual-backend testing (NumPy and PyTorch)
- Integration with CantileverBeam2DAnalytical
"""


class TestActiveSetFunction:
    """Base test class for ActiveSetFunction."""

    def _setup(self, bkd):
        from pyapprox.benchmarks.functions.algebraic.cantilever_beam_2d import (
            CantileverBeam2DAnalytical,
        )

        self._beam = CantileverBeam2DAnalytical(length=100.0, bkd=bkd)
        # Nominal: X=500, Y=1000, E=2.9e7, R=40000, w=2.5, t=3.0
        self._nominal = bkd.asarray([500.0, 1000.0, 2.9e7, 40000.0, 2.5, 3.0])
        # Keep only design variables w (idx=4) and t (idx=5)
        self._keep = [4, 5]

    def _make_asf(self, bkd, function=None, nominal=None, keep=None):
        from pyapprox.interface.functions.marginalize import (
            ActiveSetFunction,
        )

        return ActiveSetFunction(
            function or self._beam,
            nominal if nominal is not None else self._nominal,
            keep or self._keep,
            bkd,
        )

    def test_nvars_nqoi(self, bkd):
        """ActiveSetFunction exposes only kept variables."""
        self._setup(bkd)
        asf = self._make_asf(bkd)
        assert asf.nvars() == 2
        assert asf.nqoi() == 2

    def test_call_matches_full_model(self, bkd):
        """Evaluation matches full model with nominal values filled in."""
        self._setup(bkd)
        asf = self._make_asf(bkd)
        # Evaluate at nominal w, t
        reduced_sample = bkd.asarray([[2.5], [3.0]])
        result = asf(reduced_sample)

        # Compare with full model
        full_sample = bkd.asarray(
            [
                [500.0],
                [1000.0],
                [2.9e7],
                [40000.0],
                [2.5],
                [3.0],
            ]
        )
        expected = self._beam(full_sample)

        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_call_batch(self, bkd):
        """Batch evaluation works correctly."""
        self._setup(bkd)
        asf = self._make_asf(bkd)
        # Two samples: (w, t) = (2.5, 3.0) and (3.0, 4.0)
        samples = bkd.asarray([[2.5, 3.0], [3.0, 4.0]])
        result = asf(samples)
        assert result.shape == (2, 2)

        # Verify each sample
        for ii in range(2):
            single = samples[:, ii : ii + 1]
            expected = asf(single)
            bkd.assert_allclose(result[:, ii : ii + 1], expected, rtol=1e-12)

    def test_jacobian_shape(self, bkd):
        """Jacobian has shape (nqoi, n_keep)."""
        self._setup(bkd)
        asf = self._make_asf(bkd)
        sample = bkd.asarray([[2.5], [3.0]])
        jac = asf.jacobian(sample)
        assert jac.shape == (2, 2)

    def test_jacobian_extracts_correct_columns(self, bkd):
        """Jacobian matches columns 4,5 of the full Jacobian."""
        self._setup(bkd)
        asf = self._make_asf(bkd)
        sample = bkd.asarray([[2.5], [3.0]])
        jac_reduced = asf.jacobian(sample)

        # Full Jacobian
        full_sample = bkd.asarray(
            [
                [500.0],
                [1000.0],
                [2.9e7],
                [40000.0],
                [2.5],
                [3.0],
            ]
        )
        jac_full = self._beam.jacobian(full_sample)

        bkd.assert_allclose(jac_reduced, jac_full[:, [4, 5]], rtol=1e-12)

    def test_jacobian_derivative_checker(self, bkd):
        """Validate Jacobian via DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        self._setup(bkd)
        asf = self._make_asf(bkd)
        sample = bkd.asarray([[2.5], [3.0]])

        checker = DerivativeChecker(asf)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5

    def test_dynamic_binding_with_jacobian(self, bkd):
        """Function with jacobian gets jacobian bound."""
        self._setup(bkd)
        asf = self._make_asf(bkd)
        assert hasattr(asf, "jacobian")

    def test_dynamic_binding_without_jacobian(self, bkd):
        """Function without jacobian does not get jacobian bound."""
        self._setup(bkd)

        class NoJacFunction:
            def nvars(self):
                return 2

            def nqoi(self):
                return 1

            def __call__(self, samples):
                return samples[0:1, :] + samples[1:2, :]

        func = NoJacFunction()
        asf = self._make_asf(
            bkd,
            function=func,
            nominal=bkd.asarray([1.0, 2.0]),
            keep=[0],
        )
        assert not hasattr(asf, "jacobian")

    def test_with_constraints_model(self, bkd):
        """Works with CantileverBeam2DConstraints."""
        from pyapprox.benchmarks.functions.algebraic.cantilever_beam_2d import (
            CantileverBeam2DConstraints,
        )
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        self._setup(bkd)
        constraints = CantileverBeam2DConstraints(self._beam, 40000.0, 2.2535)
        asf = self._make_asf(bkd, function=constraints)

        sample = bkd.asarray([[2.5], [3.0]])
        result = asf(sample)
        assert result.shape == (2, 1)
        assert hasattr(asf, "jacobian")

        checker = DerivativeChecker(asf)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5

    def test_with_objective_model(self, bkd):
        """Works with CantileverBeam2DObjective."""
        from pyapprox.benchmarks.functions.algebraic.cantilever_beam_2d import (
            CantileverBeam2DObjective,
        )
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        self._setup(bkd)
        objective = CantileverBeam2DObjective(bkd)
        asf = self._make_asf(bkd, function=objective)

        sample = bkd.asarray([[2.5], [3.0]])
        result = asf(sample)
        assert result.shape == (1, 1)
        bkd.assert_allclose(result, bkd.asarray([[7.5]]), rtol=1e-12)

        checker = DerivativeChecker(asf)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5

    def test_single_keep_index(self, bkd):
        """Works with a single kept variable."""
        self._setup(bkd)
        asf = self._make_asf(bkd, keep=[4])
        assert asf.nvars() == 1
        sample = bkd.asarray([[2.5]])
        result = asf(sample)
        assert result.shape == (2, 1)

        jac = asf.jacobian(sample)
        assert jac.shape == (2, 1)
