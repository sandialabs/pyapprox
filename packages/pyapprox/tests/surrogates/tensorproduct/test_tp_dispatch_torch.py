"""Tests of the torch.compile tensor-product evaluation kernel.

Verifies that ``tp_eval_torch`` matches the vectorized implementation.
This tests the torch kernel itself, so the module needs torch.
"""

import pytest

# The kernel under test is a torch function.
pytest.importorskip("torch")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pyapprox.surrogates.tensorproduct import TensorProductInterpolant  # noqa: E402
from pyapprox.surrogates.tensorproduct.compute import (  # noqa: E402
    tp_eval_vectorized,
)
from pyapprox.surrogates.tensorproduct.compute_torch import (  # noqa: E402
    tp_eval_torch,
)
from pyapprox.util.backends.torch import TorchBkd  # noqa: E402
from tests._helpers.surrogates.tensorproduct_bases import (  # noqa: E402
    make_lagrange_bases,
)


def _run_torch_comparison(
    torch_bkd: TorchBkd,
    interp: TensorProductInterpolant[torch.Tensor],
    ntest: int = 200,
) -> None:
    nvars = interp.nvars()
    nterms_1d = interp._nterms_1d
    values = interp.get_values()
    assert values is not None

    np.random.seed(42)
    test_samples = torch_bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
    basis_vals_1d = interp._basis_vals_1d(test_samples)

    expected = tp_eval_vectorized(basis_vals_1d, values, nterms_1d, torch_bkd)
    result = tp_eval_torch(basis_vals_1d, values, nterms_1d)
    torch_bkd.assert_allclose(result, expected, rtol=1e-10)


class TestTpEvalTorchCompile:
    """Test torch.compile path matches vectorized."""

    def test_2d_symmetric(self, torch_bkd: TorchBkd) -> None:
        bases = make_lagrange_bases(torch_bkd, 2)
        interp = TensorProductInterpolant(torch_bkd, bases, [4, 4])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 2 + samples[1:2, :] ** 2)
        _run_torch_comparison(torch_bkd, interp)

    def test_2d_asymmetric(self, torch_bkd: TorchBkd) -> None:
        bases = make_lagrange_bases(torch_bkd, 2)
        interp = TensorProductInterpolant(torch_bkd, bases, [3, 7])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] * samples[1:2, :])
        _run_torch_comparison(torch_bkd, interp)

    def test_3d(self, torch_bkd: TorchBkd) -> None:
        bases = make_lagrange_bases(torch_bkd, 3)
        interp = TensorProductInterpolant(torch_bkd, bases, [3, 4, 5])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] + samples[1:2, :] + samples[2:3, :])
        _run_torch_comparison(torch_bkd, interp)

    def test_1d(self, torch_bkd: TorchBkd) -> None:
        bases = make_lagrange_bases(torch_bkd, 1)
        interp = TensorProductInterpolant(torch_bkd, bases, [8])
        samples = interp.get_samples()
        interp.set_values(samples[0:1, :] ** 3)
        _run_torch_comparison(torch_bkd, interp)

    def test_multi_qoi(self, torch_bkd: TorchBkd) -> None:
        bases = make_lagrange_bases(torch_bkd, 2)
        interp = TensorProductInterpolant(torch_bkd, bases, [4, 5])
        samples = interp.get_samples()
        q1 = samples[0:1, :] ** 2
        q2 = samples[1:2, :] ** 2
        interp.set_values(torch_bkd.vstack([q1, q2]))
        _run_torch_comparison(torch_bkd, interp)
