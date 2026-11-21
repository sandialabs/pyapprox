"""SUPN input Jacobian against torch autograd (torch-only, interim).

The SUPN input path evaluates through the auto-selected ``torch.compile``
basis kernels, whose compiled backward uses donated buffers, so
``bkd.jacobian`` (which retains the graph) cannot run through it. Until
those kernels become injected implementations and the generic kernel is
the default, this check takes one gradient without retaining the graph,
which the compiled graph allows. The module is skipped without torch.
"""

import numpy as np
import pytest

# Torch is optional; skip the module without it, so the imports below are
# ordinary ones.
pytest.importorskip("torch")

import torch

from pyapprox.surrogates.supn import create_supn
from pyapprox.util.backends.torch import TorchBkd


def test_input_jacobian_matches_autograd(torch_bkd: TorchBkd) -> None:
    bkd = torch_bkd
    np.random.seed(42)
    supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
    supn = supn.with_params(bkd.asarray(np.random.randn(supn.nparams())))
    sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))

    sample_ag: torch.Tensor = sample.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(supn(sample_ag)[0, 0], [sample_ag])
    bkd.assert_allclose(supn.jacobian(sample), gradient.T, rtol=1e-10)
