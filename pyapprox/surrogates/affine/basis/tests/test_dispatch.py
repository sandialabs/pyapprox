"""
Tests for three-tier dispatch of MultiIndexBasis tensor product assembly.

Tests verify that:
- Numba kernels match vectorized implementations (rtol=1e-12)
- Torch-native kernels match vectorized implementations (rtol=1e-12)
- Dispatch selects the correct implementation per backend type
- Dual-backend (NumPy + PyTorch) full round-trip through MultiIndexBasis
- Hessian symmetry is preserved across all tiers

Minor differences (~1e-14) from float64 arithmetic ordering are expected
with Numba parallel mode. Tests use rtol=1e-12 to accommodate this.
"""

import numpy as np
import pytest
import torch

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.optional_deps import package_available

if not package_available("numba"):
    pytest.skip("numba not installed", allow_module_level=True)

from pyapprox.probability import GaussianMarginal, UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.basis.compute import (
    basis_eval_vectorized,
    basis_hessian_vectorized,
    basis_jacobian_vectorized,
)
from pyapprox.surrogates.affine.basis.compute_numba import (
    basis_eval_numba,
    basis_hessian_numba,
    basis_jacobian_numba,
)
from pyapprox.surrogates.affine.basis.compute_torch import (
    basis_eval_torch,
    basis_hessian_torch,
    basis_jacobian_torch,
)
from pyapprox.surrogates.affine.basis.dispatch import (
    _stack_1d_arrays_for_numba,
    get_basis_eval_impl,
    get_basis_hessian_impl,
    get_basis_jacobian_impl,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d


def _make_basis_data_numpy(nvars, max_level, nsamples):
    """Create test data for kernel tests using NumPy."""
    bkd = NumpyBkd()
    np.random.seed(42)

    marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)

    # Ensure bases have enough terms
    for dd in range(nvars):
        max_deg = int(np.max(bkd.to_numpy(indices[dd, :])))
        if bases_1d[dd].nterms() < max_deg + 1:
            bases_1d[dd].set_nterms(max_deg + 1)

    samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (nvars, nsamples)))

    # Evaluate univariate bases
    vals_1d = [bases_1d[dd](samples[dd : dd + 1, :]) for dd in range(nvars)]
    derivs_1d = [
        bases_1d[dd].jacobian_batch(samples[dd : dd + 1, :]) for dd in range(nvars)
    ]
    hess_1d = [
        bases_1d[dd].hessian_batch(samples[dd : dd + 1, :]) for dd in range(nvars)
    ]

    return vals_1d, derivs_1d, hess_1d, indices, bkd


# ---------------------------------------------------------------------------
# Numba kernel tests
# ---------------------------------------------------------------------------


class TestNumbaKernels:
    """Test Numba kernels match vectorized implementations."""

    def test_eval_small(self):
        """Numba eval matches vectorized for 2D, level 3."""
        vals_1d, _, _, indices, bkd = _make_basis_data_numpy(2, 3, 20)
        nvars = 2

        result_vec = basis_eval_vectorized(vals_1d, indices, nvars, bkd)

        stacked = _stack_1d_arrays_for_numba(vals_1d)
        indices_np = np.asarray(indices)
        result_numba = basis_eval_numba(
            stacked,
            indices_np,
            nvars,
            vals_1d[0].shape[0],
            indices_np.shape[1],
        )

        bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_eval_higher_dim(self):
        """Numba eval matches vectorized for 5D, level 3."""
        vals_1d, _, _, indices, bkd = _make_basis_data_numpy(5, 3, 50)
        nvars = 5

        result_vec = basis_eval_vectorized(vals_1d, indices, nvars, bkd)

        stacked = _stack_1d_arrays_for_numba(vals_1d)
        indices_np = np.asarray(indices)
        result_numba = basis_eval_numba(
            stacked,
            indices_np,
            nvars,
            vals_1d[0].shape[0],
            indices_np.shape[1],
        )

        bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_jacobian_small(self):
        """Numba jacobian matches vectorized for 2D, level 3."""
        vals_1d, derivs_1d, _, indices, bkd = _make_basis_data_numpy(2, 3, 20)
        nvars = 2

        result_vec = basis_jacobian_vectorized(
            vals_1d,
            derivs_1d,
            indices,
            nvars,
            bkd,
        )

        stacked_vals = _stack_1d_arrays_for_numba(vals_1d)
        stacked_derivs = _stack_1d_arrays_for_numba(derivs_1d)
        indices_np = np.asarray(indices)
        result_numba = basis_jacobian_numba(
            stacked_vals,
            stacked_derivs,
            indices_np,
            nvars,
            vals_1d[0].shape[0],
            indices_np.shape[1],
        )

        bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_jacobian_higher_dim(self):
        """Numba jacobian matches vectorized for 4D, level 2."""
        vals_1d, derivs_1d, _, indices, bkd = _make_basis_data_numpy(4, 2, 30)
        nvars = 4

        result_vec = basis_jacobian_vectorized(
            vals_1d,
            derivs_1d,
            indices,
            nvars,
            bkd,
        )

        stacked_vals = _stack_1d_arrays_for_numba(vals_1d)
        stacked_derivs = _stack_1d_arrays_for_numba(derivs_1d)
        indices_np = np.asarray(indices)
        result_numba = basis_jacobian_numba(
            stacked_vals,
            stacked_derivs,
            indices_np,
            nvars,
            vals_1d[0].shape[0],
            indices_np.shape[1],
        )

        bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_hessian_small(self):
        """Numba hessian matches vectorized for 2D, level 3."""
        vals_1d, derivs_1d, hess_1d, indices, bkd = _make_basis_data_numpy(
            2,
            3,
            15,
        )
        nvars = 2

        result_vec = basis_hessian_vectorized(
            vals_1d,
            derivs_1d,
            hess_1d,
            indices,
            nvars,
            bkd,
        )

        stacked_vals = _stack_1d_arrays_for_numba(vals_1d)
        stacked_derivs = _stack_1d_arrays_for_numba(derivs_1d)
        stacked_hess = _stack_1d_arrays_for_numba(hess_1d)
        indices_np = np.asarray(indices)
        result_numba = basis_hessian_numba(
            stacked_vals,
            stacked_derivs,
            stacked_hess,
            indices_np,
            nvars,
            vals_1d[0].shape[0],
            indices_np.shape[1],
        )

        bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_hessian_higher_dim(self):
        """Numba hessian matches vectorized for 3D, level 2."""
        vals_1d, derivs_1d, hess_1d, indices, bkd = _make_basis_data_numpy(
            3,
            2,
            10,
        )
        nvars = 3

        result_vec = basis_hessian_vectorized(
            vals_1d,
            derivs_1d,
            hess_1d,
            indices,
            nvars,
            bkd,
        )

        stacked_vals = _stack_1d_arrays_for_numba(vals_1d)
        stacked_derivs = _stack_1d_arrays_for_numba(derivs_1d)
        stacked_hess = _stack_1d_arrays_for_numba(hess_1d)
        indices_np = np.asarray(indices)
        result_numba = basis_hessian_numba(
            stacked_vals,
            stacked_derivs,
            stacked_hess,
            indices_np,
            nvars,
            vals_1d[0].shape[0],
            indices_np.shape[1],
        )

        bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_hessian_symmetry(self):
        """Numba hessian is symmetric."""
        vals_1d, derivs_1d, hess_1d, indices, bkd = _make_basis_data_numpy(
            3,
            2,
            10,
        )
        nvars = 3

        stacked_vals = _stack_1d_arrays_for_numba(vals_1d)
        stacked_derivs = _stack_1d_arrays_for_numba(derivs_1d)
        stacked_hess = _stack_1d_arrays_for_numba(hess_1d)
        indices_np = np.asarray(indices)
        result = basis_hessian_numba(
            stacked_vals,
            stacked_derivs,
            stacked_hess,
            indices_np,
            nvars,
            vals_1d[0].shape[0],
            indices_np.shape[1],
        )

        for dd in range(nvars):
            for kk in range(dd + 1, nvars):
                bkd.assert_allclose(result[:, :, dd, kk], result[:, :, kk, dd])


# ---------------------------------------------------------------------------
# Torch kernel tests
# ---------------------------------------------------------------------------


class TestTorchKernels:
    """Test torch-native kernels match vectorized implementations."""

    def _to_torch(self, arrays):
        """Convert list of numpy arrays to torch tensors."""
        return [torch.from_numpy(np.asarray(a)) for a in arrays]

    def test_eval_matches_vectorized(self):
        """Torch eval matches vectorized for 3D, level 3."""
        vals_1d, _, _, indices, bkd = _make_basis_data_numpy(3, 3, 25)
        nvars = 3

        result_vec = basis_eval_vectorized(vals_1d, indices, nvars, bkd)

        vals_torch = self._to_torch(vals_1d)
        indices_torch = torch.from_numpy(np.asarray(indices))
        result_torch = basis_eval_torch(vals_torch, indices_torch)

        bkd.assert_allclose(
            result_vec,
            result_torch.numpy(),
            rtol=1e-12,
        )

    def test_jacobian_matches_vectorized(self):
        """Torch jacobian matches vectorized for 3D, level 2."""
        vals_1d, derivs_1d, _, indices, bkd = _make_basis_data_numpy(3, 2, 20)
        nvars = 3

        result_vec = basis_jacobian_vectorized(
            vals_1d,
            derivs_1d,
            indices,
            nvars,
            bkd,
        )

        vals_torch = self._to_torch(vals_1d)
        derivs_torch = self._to_torch(derivs_1d)
        indices_torch = torch.from_numpy(np.asarray(indices))
        result_torch = basis_jacobian_torch(
            vals_torch,
            derivs_torch,
            indices_torch,
        )

        bkd.assert_allclose(
            result_vec,
            result_torch.numpy(),
            rtol=1e-12,
        )

    def test_hessian_matches_vectorized(self):
        """Torch hessian matches vectorized for 2D, level 3."""
        vals_1d, derivs_1d, hess_1d, indices, bkd = _make_basis_data_numpy(
            2,
            3,
            15,
        )
        nvars = 2

        result_vec = basis_hessian_vectorized(
            vals_1d,
            derivs_1d,
            hess_1d,
            indices,
            nvars,
            bkd,
        )

        vals_torch = self._to_torch(vals_1d)
        derivs_torch = self._to_torch(derivs_1d)
        hess_torch = self._to_torch(hess_1d)
        indices_torch = torch.from_numpy(np.asarray(indices))
        result_torch = basis_hessian_torch(
            vals_torch,
            derivs_torch,
            hess_torch,
            indices_torch,
        )

        bkd.assert_allclose(
            result_vec,
            result_torch.numpy(),
            rtol=1e-12,
        )

    def test_hessian_symmetry(self):
        """Torch hessian is symmetric."""
        vals_1d, derivs_1d, hess_1d, indices, _ = _make_basis_data_numpy(
            3,
            2,
            10,
        )
        nvars = 3
        bkd_torch = TorchBkd()

        vals_torch = self._to_torch(vals_1d)
        derivs_torch = self._to_torch(derivs_1d)
        hess_torch = self._to_torch(hess_1d)
        indices_torch = torch.from_numpy(np.asarray(indices))
        result = basis_hessian_torch(
            vals_torch,
            derivs_torch,
            hess_torch,
            indices_torch,
        )

        for dd in range(nvars):
            for kk in range(dd + 1, nvars):
                bkd_torch.assert_allclose(
                    result[:, :, dd, kk],
                    result[:, :, kk, dd],
                )


# ---------------------------------------------------------------------------
# Dispatch selection tests
# ---------------------------------------------------------------------------


class TestDispatchSelection:
    """Test dispatch selects correct implementation per backend type."""

    def test_numpy_gets_numba_eval(self):
        """NumPy backend should get Numba closure for eval."""
        bkd = NumpyBkd()
        impl = get_basis_eval_impl(bkd)
        assert impl is not basis_eval_vectorized

    def test_numpy_gets_numba_jacobian(self):
        """NumPy backend should get Numba closure for jacobian."""
        bkd = NumpyBkd()
        impl = get_basis_jacobian_impl(bkd)
        assert impl is not basis_jacobian_vectorized

    def test_numpy_gets_numba_hessian(self):
        """NumPy backend should get Numba closure for hessian."""
        bkd = NumpyBkd()
        impl = get_basis_hessian_impl(bkd)
        assert impl is not basis_hessian_vectorized

    def test_torch_gets_compiled_eval(self):
        """Torch backend should get compiled closure for eval."""
        bkd = TorchBkd()
        impl = get_basis_eval_impl(bkd)
        assert impl is not basis_eval_vectorized

    def test_torch_gets_compiled_jacobian(self):
        """Torch backend should get compiled closure for jacobian."""
        bkd = TorchBkd()
        impl = get_basis_jacobian_impl(bkd)
        assert impl is not basis_jacobian_vectorized

    def test_torch_gets_compiled_hessian(self):
        """Torch backend should get compiled closure for hessian."""
        bkd = TorchBkd()
        impl = get_basis_hessian_impl(bkd)
        assert impl is not basis_hessian_vectorized


# ---------------------------------------------------------------------------
# Integration tests (dual-backend, full round-trip through MultiIndexBasis)
# ---------------------------------------------------------------------------


class TestBasisDispatchIntegration:
    """Dual-backend integration tests for dispatch through MultiIndexBasis.

    Verifies that eval, jacobian_batch, and hessian_batch produce correct
    results when the dispatch layer is active.
    """

    def _create_basis(self, bkd, nvars, max_level):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    @pytest.mark.slow_on("TorchBkd")
    def test_eval_shape(self, bkd):
        """Dispatched eval returns correct shape."""
        np.random.seed(42)
        basis = self._create_basis(bkd, 3, 3)
        nsamples = 20
        samples = bkd.asarray(
            np.random.uniform(-1, 1, (3, nsamples)),
        )
        result = basis(samples)
        assert result.shape == (nsamples, basis.nterms())

    def test_jacobian_shape(self, bkd):
        """Dispatched jacobian returns correct shape."""
        np.random.seed(42)
        basis = self._create_basis(bkd, 3, 2)
        nsamples = 15
        samples = bkd.asarray(
            np.random.uniform(-1, 1, (3, nsamples)),
        )
        jac = basis.jacobian_batch(samples)
        assert jac.shape == (nsamples, basis.nterms(), 3)

    def test_hessian_shape(self, bkd):
        """Dispatched hessian returns correct shape."""
        np.random.seed(42)
        basis = self._create_basis(bkd, 2, 3)
        nsamples = 10
        samples = bkd.asarray(
            np.random.uniform(-1, 1, (2, nsamples)),
        )
        hess = basis.hessian_batch(samples)
        assert hess.shape == (nsamples, basis.nterms(), 2, 2)

    @pytest.mark.slow_on("TorchBkd")
    def test_eval_orthonormality(self, bkd):
        """Dispatched eval preserves orthonormality via quadrature."""
        np.random.seed(42)
        basis = self._create_basis(bkd, 2, 3)
        npoints_1d = [8, 8]
        pts, wts = basis.tensor_product_quadrature(npoints_1d)

        values = basis(pts)
        weighted_values = values * bkd.reshape(wts, (-1, 1))
        gram = bkd.dot(values.T, weighted_values)

        expected = bkd.eye(basis.nterms())
        bkd.assert_allclose(gram, expected, atol=1e-10)

    @pytest.mark.slow_on("TorchBkd")
    def test_jacobian_finite_difference(self, bkd):
        """Dispatched jacobian matches finite differences."""
        np.random.seed(42)
        basis = self._create_basis(bkd, 2, 3)
        nsamples = 5
        samples = bkd.asarray(
            np.random.uniform(-0.9, 0.9, (2, nsamples)),
        )

        jac = basis.jacobian_batch(samples)

        eps = 1e-7
        for dd in range(2):
            samples_plus = bkd.copy(samples)
            samples_minus = bkd.copy(samples)
            samples_plus[dd, :] += eps
            samples_minus[dd, :] -= eps

            fd_jac = (basis(samples_plus) - basis(samples_minus)) / (2 * eps)
            bkd.assert_allclose(
                jac[:, :, dd],
                fd_jac,
                rtol=1e-5,
                atol=1e-7,
            )

    @pytest.mark.slow_on("TorchBkd")
    def test_hessian_finite_difference(self, bkd):
        """Dispatched hessian matches finite differences on jacobian."""
        np.random.seed(42)
        basis = self._create_basis(bkd, 2, 2)
        nsamples = 3
        samples = bkd.asarray(
            np.random.uniform(-0.9, 0.9, (2, nsamples)),
        )

        hess = basis.hessian_batch(samples)

        eps = 1e-6
        for dd in range(2):
            samples_plus = bkd.copy(samples)
            samples_minus = bkd.copy(samples)
            samples_plus[dd, :] += eps
            samples_minus[dd, :] -= eps

            jac_plus = basis.jacobian_batch(samples_plus)
            jac_minus = basis.jacobian_batch(samples_minus)

            fd_hess_row = (jac_plus - jac_minus) / (2 * eps)
            bkd.assert_allclose(
                hess[:, :, dd, :],
                fd_hess_row,
                rtol=1e-4,
                atol=1e-6,
            )

    @pytest.mark.slow_on("TorchBkd")
    def test_hessian_symmetry(self, bkd):
        """Dispatched hessian is symmetric."""
        np.random.seed(42)
        basis = self._create_basis(bkd, 3, 2)
        nsamples = 5
        samples = bkd.asarray(
            np.random.uniform(-1, 1, (3, nsamples)),
        )

        hess = basis.hessian_batch(samples)

        for dd in range(3):
            for kk in range(dd + 1, 3):
                bkd.assert_allclose(
                    hess[:, :, dd, kk],
                    hess[:, :, kk, dd],
                )

    def test_mixed_bases(self, bkd):
        """Dispatch works with mixed Legendre + Hermite bases."""
        np.random.seed(42)
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
        ]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(2, 3, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

        nsamples = 10
        samples = bkd.asarray(
            np.column_stack(
                [
                    np.random.uniform(-1, 1, nsamples),
                    np.random.normal(0, 1, nsamples),
                ]
            ).T
        )

        values = basis(samples)
        assert values.shape == (nsamples, basis.nterms())

        jac = basis.jacobian_batch(samples)
        assert jac.shape == (nsamples, basis.nterms(), 2)
