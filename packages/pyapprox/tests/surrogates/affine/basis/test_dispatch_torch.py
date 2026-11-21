"""Tests of the torch-native tensor-product basis kernels and their dispatch.

Tests verify that:
- Torch-native kernels match vectorized implementations (rtol=1e-12)
- Dispatch selects a compiled implementation for TorchBkd
- The torch Hessian is symmetric

These test the torch kernels themselves, so the module needs torch.
"""

import pytest

# The kernels under test are torch functions.
pytest.importorskip("torch")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pyapprox.surrogates.affine.basis.compute import (  # noqa: E402
    basis_eval_vectorized,
    basis_hessian_vectorized,
    basis_jacobian_vectorized,
)
from pyapprox.surrogates.affine.basis.compute_torch import (  # noqa: E402
    basis_eval_torch,
    basis_hessian_torch,
    basis_jacobian_torch,
)
from pyapprox.surrogates.affine.basis.dispatch import (  # noqa: E402
    get_basis_eval_impl,
    get_basis_hessian_impl,
    get_basis_jacobian_impl,
)
from pyapprox.util.backends.torch import TorchBkd  # noqa: E402
from tests._helpers.surrogates.basis_kernels import (  # noqa: E402
    make_basis_kernel_data,
)

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
        vals_1d, _, _, indices, bkd = make_basis_kernel_data(3, 3, 25)
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
        vals_1d, derivs_1d, _, indices, bkd = make_basis_kernel_data(3, 2, 20)
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
        vals_1d, derivs_1d, hess_1d, indices, bkd = make_basis_kernel_data(
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
        vals_1d, derivs_1d, hess_1d, indices, _ = make_basis_kernel_data(
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


class TestTorchDispatchSelection:
    """Test dispatch selects a compiled implementation for TorchBkd."""

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
