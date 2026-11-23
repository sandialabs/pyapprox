"""Torch conformance beyond values: every differentiable row of the
table in ``tests/_helpers/backend_conformance.py`` passes ``gradcheck``,
so a generic kernel built from protocol methods needs no torch-specific
copy to be differentiated. (Compiled rows run on every backend, through
``bkd.jit``, in ``test_backend_conformance.py``.)

Torch is optional, so the whole module is skipped without it.
"""

from typing import Any, Iterator, List

import pytest

# Skip the module without torch; the imports below are then ordinary ones.
pytest.importorskip("torch")

import torch

from pyapprox.util.backends.torch import TorchBkd
from tests._helpers.backend_conformance import (
    CASES,
    Case,
    case_id,
    prepare,
)

_DIFFERENTIABLE = [c for c in CASES if c.differentiable]


def _leaves(value: Any) -> List[torch.Tensor]:
    """The floating tensors in ``value``, also inside tuples."""
    if isinstance(value, tuple):
        return [leaf for v in value for leaf in _leaves(v)]
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return [value]
    return []


def _rebuild(value: Any, leaves: Iterator[torch.Tensor]) -> Any:
    """``value`` with its floating tensors replaced, in order, by
    ``leaves``."""
    if isinstance(value, tuple):
        return tuple(_rebuild(v, leaves) for v in value)
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return next(leaves)
    return value


@pytest.mark.parametrize("case", _DIFFERENTIABLE, ids=case_id)
def test_gradcheck(torch_bkd: TorchBkd, case: Case) -> None:
    method, args = prepare(torch_bkd, case)
    inputs = tuple(leaf.detach().clone().requires_grad_(True) for leaf in _leaves(args))
    assert inputs, "a differentiable row needs a floating input"

    def function(*leaves: torch.Tensor) -> Any:
        return method(*_rebuild(args, iter(leaves)))

    assert torch.autograd.gradcheck(function, inputs)
