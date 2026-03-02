"""Tests for tensor product evaluation dispatch.

Verifies that all three dispatch strategies (Numba, torch.compile, vectorized)
produce results matching the original dense assembly approach.
"""

from typing import List

import numpy as np
import pytest
import torch

from pyapprox.surrogates.tensorproduct import TensorProductInterpolant
from pyapprox.surrogates.tensorproduct.compute import (
    tp_eval_vectorized,
)
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.cartesian import cartesian_product_indices
from pyapprox.util.optional_deps import package_available

if not package_available("numba"):
    pytest.skip("numba not installed", allow_module_level=True)

from pyapprox.surrogates.affine.univariate import (
    LagrangeBasis1D,
    LegendrePolynomial1D,
)
from pyapprox.surrogates.affine.univariate.piecewisepoly import (
    PiecewiseLinear,
)
from pyapprox.surrogates.affine.univariate.piecewisepoly.dynamic import (
    DynamicPiecewiseBasis,
    EquidistantNodeGenerator,
)
from pyapprox.surrogates.tensorproduct.compute_numba import (
    tp_eval_numba,
)
from pyapprox.surrogates.tensorproduct.compute_torch import (
    tp_eval_torch,
)


def _dense_eval(
    basis_vals_1d,
    values,
    tp_indices,
    bkd,
):
    """Original dense assembly approach for reference."""
    nvars = len(basis_vals_1d)
    interp_mat = basis_vals_1d[0][:, tp_indices[0, :]]
    for dd in range(1, nvars):
        interp_mat = interp_mat * basis_vals_1d[dd][:, tp_indices[dd, :]]
    return bkd.dot(values, interp_mat.T)


def _make_lagrange_bases(bkd, nvars: int) -> List:
    """Create separate LagrangeBasis1D instances per dimension."""
    bases = []
    for _ in range(nvars):
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(10)
        bases.append(LagrangeBasis1D(bkd, poly.gauss_quadrature_rule))
    return bases


def _make_piecewise_bases(bkd, nvars: int) -> List:
    """Create separate DynamicPiecewiseBasis instances per dimension."""
    bases = []
    for _ in range(nvars):
        node_gen = EquidistantNodeGenerator(bkd, (-1.0, 1.0))
        bases.append(DynamicPiecewiseBasis(bkd, PiecewiseLinear, node_gen))
    return bases


class TestTpEvalVectorized:
    """Test tp_eval_vectorized against original dense assembly."""

    def _run_comparison(
        self, bkd, interp, ntest: int = 100
    ) -> None:
        """Compare vectorized einsum against dense assembly."""
        nvars = interp.nvars()
        nterms_1d = interp._nterms_1d
        tp_indices = cartesian_product_indices(nterms_1d, bkd)

        np.random.seed(42)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        basis_vals_1d = interp._basis_vals_1d(test_samples)

        expected = _dense_eval(basis_vals_1d, interp._values, tp_indices, bkd)
        result = tp_eval_vectorized(basis_vals_1d, interp._values, nterms_1d, bkd)
        bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_lagrange_2d_symmetric(self, bkd) -> None:
        bases = _make_lagrange_bases(bkd, 2)
        interp = TensorProductInterpolant(bkd, bases, [4, 4])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 2 + samples[1:2, :] ** 2)
        self._run_comparison(bkd, interp)

    def test_lagrange_2d_asymmetric(self, bkd) -> None:
        bases = _make_lagrange_bases(bkd, 2)
        interp = TensorProductInterpolant(bkd, bases, [3, 5])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] * samples[1:2, :])
        self._run_comparison(bkd, interp)

    def test_lagrange_3d(self, bkd) -> None:
        bases = _make_lagrange_bases(bkd, 3)
        interp = TensorProductInterpolant(bkd, bases, [3, 4, 5])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] + samples[1:2, :] + samples[2:3, :])
        self._run_comparison(bkd, interp)

    def test_piecewise_2d(self, bkd) -> None:
        bases = _make_piecewise_bases(bkd, 2)
        interp = TensorProductInterpolant(bkd, bases, [10, 12])
        samples = interp.get_samples()
        interp.set_values(
            bkd.sin(samples[0:1, :]) + bkd.cos(samples[1:2, :])
        )
        self._run_comparison(bkd, interp)

    def test_multi_qoi(self, bkd) -> None:
        bases = _make_lagrange_bases(bkd, 2)
        interp = TensorProductInterpolant(bkd, bases, [4, 4])
        samples = interp.get_samples()
        q1 = samples[0:1, :] ** 2
        q2 = samples[1:2, :] ** 2
        interp.set_values(bkd.vstack([q1, q2]))
        self._run_comparison(bkd, interp)

    def test_1d(self, bkd) -> None:
        bases = _make_lagrange_bases(bkd, 1)
        interp = TensorProductInterpolant(bkd, bases, [6])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 3)
        self._run_comparison(bkd, interp)


class TestTpEvalNumba:
    """Test Numba kernel matches vectorized (NumPy only)."""

    def _run_numba_comparison(
        self,
        numpy_bkd,
        interp,
        ntest: int = 200,
    ) -> None:
        nvars = interp.nvars()
        nterms_1d = interp._nterms_1d

        np.random.seed(42)
        test_samples = numpy_bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        basis_vals_1d = interp._basis_vals_1d(test_samples)

        expected = tp_eval_vectorized(
            basis_vals_1d, interp._values, nterms_1d, numpy_bkd
        )

        # Prepare Numba inputs
        npoints = ntest
        nqoi = interp._values.shape[0]
        max_n1d = max(nterms_1d)
        basis_vals_pad = np.zeros((nvars, npoints, max_n1d))
        for d in range(nvars):
            basis_vals_pad[d, :, : nterms_1d[d]] = basis_vals_1d[d]
        nterms_arr = np.array(nterms_1d, dtype=np.int64)

        result = tp_eval_numba(
            np.asarray(interp._values),
            basis_vals_pad,
            nterms_arr,
            nvars,
            nqoi,
            npoints,
        )
        numpy_bkd.assert_allclose(numpy_bkd.asarray(result), expected, rtol=1e-10)

    def test_2d_symmetric(self, numpy_bkd) -> None:
        bases = _make_lagrange_bases(numpy_bkd, 2)
        interp = TensorProductInterpolant(numpy_bkd, bases, [4, 4])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 2 + samples[1:2, :] ** 2)
        self._run_numba_comparison(numpy_bkd, interp)

    def test_2d_asymmetric(self, numpy_bkd) -> None:
        bases = _make_lagrange_bases(numpy_bkd, 2)
        interp = TensorProductInterpolant(numpy_bkd, bases, [3, 7])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] * samples[1:2, :])
        self._run_numba_comparison(numpy_bkd, interp)

    def test_3d(self, numpy_bkd) -> None:
        bases = _make_lagrange_bases(numpy_bkd, 3)
        interp = TensorProductInterpolant(numpy_bkd, bases, [3, 4, 5])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] + samples[1:2, :] + samples[2:3, :])
        self._run_numba_comparison(numpy_bkd, interp)

    def test_5d(self, numpy_bkd) -> None:
        bases = _make_lagrange_bases(numpy_bkd, 5)
        interp = TensorProductInterpolant(numpy_bkd, bases, [3, 3, 3, 3, 4])
        samples = interp.get_samples()
        interp.set_values(numpy_bkd.sum(samples, axis=0, keepdims=True))
        self._run_numba_comparison(numpy_bkd, interp)

    def test_1d(self, numpy_bkd) -> None:
        bases = _make_lagrange_bases(numpy_bkd, 1)
        interp = TensorProductInterpolant(numpy_bkd, bases, [8])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 3)
        self._run_numba_comparison(numpy_bkd, interp)

    def test_multi_qoi(self, numpy_bkd) -> None:
        bases = _make_lagrange_bases(numpy_bkd, 2)
        interp = TensorProductInterpolant(numpy_bkd, bases, [4, 5])
        samples = interp.get_samples()
        q1 = samples[0:1, :] ** 2
        q2 = samples[1:2, :] ** 2
        interp.set_values(numpy_bkd.vstack([q1, q2]))
        self._run_numba_comparison(numpy_bkd, interp)


class TestTpEvalTorchCompile:
    """Test torch.compile path matches vectorized (Torch only)."""

    def _get_torch_bkd(self):
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

    def _run_torch_comparison(
        self,
        torch_bkd,
        interp,
        ntest: int = 200,
    ) -> None:
        nvars = interp.nvars()
        nterms_1d = interp._nterms_1d

        np.random.seed(42)
        test_samples = torch_bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        basis_vals_1d = interp._basis_vals_1d(test_samples)

        expected = tp_eval_vectorized(
            basis_vals_1d, interp._values, nterms_1d, torch_bkd
        )
        result = tp_eval_torch(basis_vals_1d, interp._values, nterms_1d)
        torch_bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_2d_symmetric(self) -> None:
        torch_bkd = self._get_torch_bkd()
        bases = _make_lagrange_bases(torch_bkd, 2)
        interp = TensorProductInterpolant(torch_bkd, bases, [4, 4])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 2 + samples[1:2, :] ** 2)
        self._run_torch_comparison(torch_bkd, interp)

    def test_2d_asymmetric(self) -> None:
        torch_bkd = self._get_torch_bkd()
        bases = _make_lagrange_bases(torch_bkd, 2)
        interp = TensorProductInterpolant(torch_bkd, bases, [3, 7])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] * samples[1:2, :])
        self._run_torch_comparison(torch_bkd, interp)

    def test_3d(self) -> None:
        torch_bkd = self._get_torch_bkd()
        bases = _make_lagrange_bases(torch_bkd, 3)
        interp = TensorProductInterpolant(torch_bkd, bases, [3, 4, 5])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] + samples[1:2, :] + samples[2:3, :])
        self._run_torch_comparison(torch_bkd, interp)

    def test_1d(self) -> None:
        torch_bkd = self._get_torch_bkd()
        bases = _make_lagrange_bases(torch_bkd, 1)
        interp = TensorProductInterpolant(torch_bkd, bases, [8])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 3)
        self._run_torch_comparison(torch_bkd, interp)

    def test_multi_qoi(self) -> None:
        torch_bkd = self._get_torch_bkd()
        bases = _make_lagrange_bases(torch_bkd, 2)
        interp = TensorProductInterpolant(torch_bkd, bases, [4, 5])
        samples = interp.get_samples()
        q1 = samples[0:1, :] ** 2
        q2 = samples[1:2, :] ** 2
        interp.set_values(torch_bkd.vstack([q1, q2]))
        self._run_torch_comparison(torch_bkd, interp)
