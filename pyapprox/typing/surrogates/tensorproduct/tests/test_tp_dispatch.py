"""Tests for tensor product evaluation dispatch.

Verifies that all three dispatch strategies (Numba, torch.compile, vectorized)
produce results matching the original dense assembly approach.
"""

import unittest
from typing import Any, Generic, List

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.util.cartesian import cartesian_product_indices
from pyapprox.typing.surrogates.tensorproduct import TensorProductInterpolant
from pyapprox.typing.surrogates.tensorproduct.compute import (
    tp_eval_vectorized,
)
from pyapprox.typing.surrogates.tensorproduct.compute_numba import (
    tp_eval_numba,
)
from pyapprox.typing.surrogates.tensorproduct.compute_torch import (
    tp_eval_torch,
)
from pyapprox.typing.surrogates.affine.univariate import (
    LagrangeBasis1D,
    LegendrePolynomial1D,
)
from pyapprox.typing.surrogates.affine.univariate.piecewisepoly import (
    PiecewiseLinear,
)
from pyapprox.typing.surrogates.affine.univariate.piecewisepoly.dynamic import (
    DynamicPiecewiseBasis,
    EquidistantNodeGenerator,
)


def _dense_eval(
    basis_vals_1d: List[Array],
    values: Array,
    tp_indices: Array,
    bkd: Backend[Array],
) -> Array:
    """Original dense assembly approach for reference."""
    nvars = len(basis_vals_1d)
    interp_mat = basis_vals_1d[0][:, tp_indices[0, :]]
    for dd in range(1, nvars):
        interp_mat = interp_mat * basis_vals_1d[dd][:, tp_indices[dd, :]]
    return bkd.dot(values, interp_mat.T)


def _make_lagrange_bases(bkd: Backend[Array], nvars: int) -> List:
    """Create separate LagrangeBasis1D instances per dimension."""
    bases = []
    for _ in range(nvars):
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(10)
        bases.append(LagrangeBasis1D(bkd, poly.gauss_quadrature_rule))
    return bases


def _make_piecewise_bases(bkd: Backend[Array], nvars: int) -> List:
    """Create separate DynamicPiecewiseBasis instances per dimension."""
    bases = []
    for _ in range(nvars):
        node_gen = EquidistantNodeGenerator(bkd, (-1.0, 1.0))
        bases.append(DynamicPiecewiseBasis(bkd, PiecewiseLinear, node_gen))
    return bases


class TestTpEvalVectorized(Generic[Array], unittest.TestCase):
    """Test tp_eval_vectorized against original dense assembly."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _run_comparison(
        self, interp: TensorProductInterpolant[Array], ntest: int = 100
    ) -> None:
        """Compare vectorized einsum against dense assembly."""
        nvars = interp.nvars()
        nterms_1d = interp._nterms_1d
        tp_indices = cartesian_product_indices(nterms_1d, self._bkd)

        np.random.seed(42)
        test_samples = self._bkd.asarray(
            np.random.uniform(-1, 1, (nvars, ntest))
        )
        basis_vals_1d = interp._basis_vals_1d(test_samples)

        expected = _dense_eval(
            basis_vals_1d, interp._values, tp_indices, self._bkd
        )
        result = tp_eval_vectorized(
            basis_vals_1d, interp._values, nterms_1d, self._bkd
        )
        self._bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_lagrange_2d_symmetric(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 2)
        interp = TensorProductInterpolant(self._bkd, bases, [4, 4])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 2 + samples[1:2, :] ** 2)
        self._run_comparison(interp)

    def test_lagrange_2d_asymmetric(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 2)
        interp = TensorProductInterpolant(self._bkd, bases, [3, 5])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] * samples[1:2, :])
        self._run_comparison(interp)

    def test_lagrange_3d(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 3)
        interp = TensorProductInterpolant(self._bkd, bases, [3, 4, 5])
        samples = interp.get_samples()
        interp.set_values(
            samples[0:1, :] + samples[1:2, :] + samples[2:3, :]
        )
        self._run_comparison(interp)

    def test_piecewise_2d(self) -> None:
        bases = _make_piecewise_bases(self._bkd, 2)
        interp = TensorProductInterpolant(self._bkd, bases, [10, 12])
        samples = interp.get_samples()
        interp.set_values(
            self._bkd.sin(samples[0:1, :]) + self._bkd.cos(samples[1:2, :])
        )
        self._run_comparison(interp)

    def test_multi_qoi(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 2)
        interp = TensorProductInterpolant(self._bkd, bases, [4, 4])
        samples = interp.get_samples()
        q1 = samples[0:1, :] ** 2
        q2 = samples[1:2, :] ** 2
        interp.set_values(self._bkd.vstack([q1, q2]))
        self._run_comparison(interp)

    def test_1d(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 1)
        interp = TensorProductInterpolant(self._bkd, bases, [6])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 3)
        self._run_comparison(interp)


class TestTpEvalVectorizedNumpy(TestTpEvalVectorized[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTpEvalVectorizedTorch(TestTpEvalVectorized[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestTpEvalNumba(unittest.TestCase):
    """Test Numba kernel matches vectorized (NumPy only)."""

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def _run_numba_comparison(
        self,
        interp: TensorProductInterpolant[NDArray[Any]],
        ntest: int = 200,
    ) -> None:
        nvars = interp.nvars()
        nterms_1d = interp._nterms_1d

        np.random.seed(42)
        test_samples = self._bkd.asarray(
            np.random.uniform(-1, 1, (nvars, ntest))
        )
        basis_vals_1d = interp._basis_vals_1d(test_samples)

        expected = tp_eval_vectorized(
            basis_vals_1d, interp._values, nterms_1d, self._bkd
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
        self._bkd.assert_allclose(
            self._bkd.asarray(result), expected, rtol=1e-10
        )

    def test_2d_symmetric(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 2)
        interp = TensorProductInterpolant(self._bkd, bases, [4, 4])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 2 + samples[1:2, :] ** 2)
        self._run_numba_comparison(interp)

    def test_2d_asymmetric(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 2)
        interp = TensorProductInterpolant(self._bkd, bases, [3, 7])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] * samples[1:2, :])
        self._run_numba_comparison(interp)

    def test_3d(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 3)
        interp = TensorProductInterpolant(self._bkd, bases, [3, 4, 5])
        samples = interp.get_samples()
        interp.set_values(
            samples[0:1, :] + samples[1:2, :] + samples[2:3, :]
        )
        self._run_numba_comparison(interp)

    def test_5d(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 5)
        interp = TensorProductInterpolant(self._bkd, bases, [3, 3, 3, 3, 4])
        samples = interp.get_samples()
        interp.set_values(self._bkd.sum(samples, axis=0, keepdims=True))
        self._run_numba_comparison(interp)

    def test_1d(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 1)
        interp = TensorProductInterpolant(self._bkd, bases, [8])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 3)
        self._run_numba_comparison(interp)

    def test_multi_qoi(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 2)
        interp = TensorProductInterpolant(self._bkd, bases, [4, 5])
        samples = interp.get_samples()
        q1 = samples[0:1, :] ** 2
        q2 = samples[1:2, :] ** 2
        interp.set_values(self._bkd.vstack([q1, q2]))
        self._run_numba_comparison(interp)


class TestTpEvalTorchCompile(unittest.TestCase):
    """Test torch.compile path matches vectorized (Torch only)."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def _run_torch_comparison(
        self,
        interp: TensorProductInterpolant[torch.Tensor],
        ntest: int = 200,
    ) -> None:
        nvars = interp.nvars()
        nterms_1d = interp._nterms_1d

        np.random.seed(42)
        test_samples = self._bkd.asarray(
            np.random.uniform(-1, 1, (nvars, ntest))
        )
        basis_vals_1d = interp._basis_vals_1d(test_samples)

        expected = tp_eval_vectorized(
            basis_vals_1d, interp._values, nterms_1d, self._bkd
        )
        result = tp_eval_torch(basis_vals_1d, interp._values, nterms_1d)
        self._bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_2d_symmetric(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 2)
        interp = TensorProductInterpolant(self._bkd, bases, [4, 4])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 2 + samples[1:2, :] ** 2)
        self._run_torch_comparison(interp)

    def test_2d_asymmetric(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 2)
        interp = TensorProductInterpolant(self._bkd, bases, [3, 7])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] * samples[1:2, :])
        self._run_torch_comparison(interp)

    def test_3d(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 3)
        interp = TensorProductInterpolant(self._bkd, bases, [3, 4, 5])
        samples = interp.get_samples()
        interp.set_values(
            samples[0:1, :] + samples[1:2, :] + samples[2:3, :]
        )
        self._run_torch_comparison(interp)

    def test_1d(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 1)
        interp = TensorProductInterpolant(self._bkd, bases, [8])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 3)
        self._run_torch_comparison(interp)

    def test_multi_qoi(self) -> None:
        bases = _make_lagrange_bases(self._bkd, 2)
        interp = TensorProductInterpolant(self._bkd, bases, [4, 5])
        samples = interp.get_samples()
        q1 = samples[0:1, :] ** 2
        q2 = samples[1:2, :] ** 2
        interp.set_values(self._bkd.vstack([q1, q2]))
        self._run_torch_comparison(interp)


if __name__ == "__main__":
    unittest.main()
