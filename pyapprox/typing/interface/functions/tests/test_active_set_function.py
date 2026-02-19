"""
Tests for ActiveSetFunction.

Tests cover:
- Variable fixing and evaluation
- Jacobian propagation and column extraction
- Dynamic binding (jacobian present/absent)
- Dual-backend testing (NumPy and PyTorch)
- Integration with CantileverBeam2DAnalytical
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestActiveSetFunction(Generic[Array], unittest.TestCase):
    """Base test class for ActiveSetFunction."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        from pyapprox.typing.benchmarks.functions.algebraic.cantilever_beam_2d import (
            CantileverBeam2DAnalytical,
        )

        self._bkd = self.bkd()
        self._beam = CantileverBeam2DAnalytical(length=100.0, bkd=self._bkd)
        # Nominal: X=500, Y=1000, E=2.9e7, R=40000, w=2.5, t=3.0
        self._nominal = self._bkd.asarray(
            [500.0, 1000.0, 2.9e7, 40000.0, 2.5, 3.0]
        )
        # Keep only design variables w (idx=4) and t (idx=5)
        self._keep = [4, 5]

    def _make_asf(self, function=None, nominal=None, keep=None):
        from pyapprox.typing.interface.functions.marginalize import (
            ActiveSetFunction,
        )

        return ActiveSetFunction(
            function or self._beam,
            nominal if nominal is not None else self._nominal,
            keep or self._keep,
            self._bkd,
        )

    def test_nvars_nqoi(self):
        """ActiveSetFunction exposes only kept variables."""
        asf = self._make_asf()
        self.assertEqual(asf.nvars(), 2)
        self.assertEqual(asf.nqoi(), 2)

    def test_call_matches_full_model(self):
        """Evaluation matches full model with nominal values filled in."""
        bkd = self._bkd
        asf = self._make_asf()
        # Evaluate at nominal w, t
        reduced_sample = bkd.asarray([[2.5], [3.0]])
        result = asf(reduced_sample)

        # Compare with full model
        full_sample = bkd.asarray([
            [500.0], [1000.0], [2.9e7], [40000.0], [2.5], [3.0],
        ])
        expected = self._beam(full_sample)

        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_call_batch(self):
        """Batch evaluation works correctly."""
        bkd = self._bkd
        asf = self._make_asf()
        # Two samples: (w, t) = (2.5, 3.0) and (3.0, 4.0)
        samples = bkd.asarray([[2.5, 3.0], [3.0, 4.0]])
        result = asf(samples)
        self.assertEqual(result.shape, (2, 2))

        # Verify each sample
        for ii in range(2):
            single = samples[:, ii:ii + 1]
            expected = asf(single)
            bkd.assert_allclose(
                result[:, ii:ii + 1], expected, rtol=1e-12
            )

    def test_jacobian_shape(self):
        """Jacobian has shape (nqoi, n_keep)."""
        bkd = self._bkd
        asf = self._make_asf()
        sample = bkd.asarray([[2.5], [3.0]])
        jac = asf.jacobian(sample)
        self.assertEqual(jac.shape, (2, 2))

    def test_jacobian_extracts_correct_columns(self):
        """Jacobian matches columns 4,5 of the full Jacobian."""
        bkd = self._bkd
        asf = self._make_asf()
        sample = bkd.asarray([[2.5], [3.0]])
        jac_reduced = asf.jacobian(sample)

        # Full Jacobian
        full_sample = bkd.asarray([
            [500.0], [1000.0], [2.9e7], [40000.0], [2.5], [3.0],
        ])
        jac_full = self._beam.jacobian(full_sample)

        bkd.assert_allclose(
            jac_reduced, jac_full[:, [4, 5]], rtol=1e-12
        )

    def test_jacobian_derivative_checker(self):
        """Validate Jacobian via DerivativeChecker."""
        from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = self._bkd
        asf = self._make_asf()
        sample = bkd.asarray([[2.5], [3.0]])

        checker = DerivativeChecker(asf)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-5)

    def test_dynamic_binding_with_jacobian(self):
        """Function with jacobian gets jacobian bound."""
        asf = self._make_asf()
        self.assertTrue(hasattr(asf, "jacobian"))

    def test_dynamic_binding_without_jacobian(self):
        """Function without jacobian does not get jacobian bound."""

        class NoJacFunction:
            def nvars(self):
                return 2

            def nqoi(self):
                return 1

            def __call__(self, samples):
                return samples[0:1, :] + samples[1:2, :]

        func = NoJacFunction()
        asf = self._make_asf(
            function=func,
            nominal=self._bkd.asarray([1.0, 2.0]),
            keep=[0],
        )
        self.assertFalse(hasattr(asf, "jacobian"))

    def test_with_constraints_model(self):
        """Works with CantileverBeam2DConstraints."""
        from pyapprox.typing.benchmarks.functions.algebraic.cantilever_beam_2d import (
            CantileverBeam2DConstraints,
        )
        from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = self._bkd
        constraints = CantileverBeam2DConstraints(
            self._beam, 40000.0, 2.2535
        )
        asf = self._make_asf(function=constraints)

        sample = bkd.asarray([[2.5], [3.0]])
        result = asf(sample)
        self.assertEqual(result.shape, (2, 1))
        self.assertTrue(hasattr(asf, "jacobian"))

        checker = DerivativeChecker(asf)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-5)

    def test_with_objective_model(self):
        """Works with CantileverBeam2DObjective."""
        from pyapprox.typing.benchmarks.functions.algebraic.cantilever_beam_2d import (
            CantileverBeam2DObjective,
        )
        from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = self._bkd
        objective = CantileverBeam2DObjective(bkd)
        asf = self._make_asf(function=objective)

        sample = bkd.asarray([[2.5], [3.0]])
        result = asf(sample)
        self.assertEqual(result.shape, (1, 1))
        bkd.assert_allclose(result, bkd.asarray([[7.5]]), rtol=1e-12)

        checker = DerivativeChecker(asf)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-5)

    def test_single_keep_index(self):
        """Works with a single kept variable."""
        bkd = self._bkd
        asf = self._make_asf(keep=[4])
        self.assertEqual(asf.nvars(), 1)
        sample = bkd.asarray([[2.5]])
        result = asf(sample)
        self.assertEqual(result.shape, (2, 1))

        jac = asf.jacobian(sample)
        self.assertEqual(jac.shape, (2, 1))


class TestActiveSetFunctionNumpy(TestActiveSetFunction[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestActiveSetFunctionTorch(TestActiveSetFunction[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
