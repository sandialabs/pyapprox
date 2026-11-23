"""Torch conformance beyond values: every traceable row of the table in
``tests/_helpers/backend_conformance.py`` compiles into one graph under
``torch.compile(fullgraph=True)``, which fails on any graph break, and
every differentiable row passes ``gradcheck``. So a generic kernel built
from protocol methods needs no torch-specific copy to compile or to be
differentiated.

Torch is optional, so the whole module is skipped without it.
"""

from typing import Any, Callable, Iterator, List, Tuple

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
    run_case,
)

_TRACEABLE = [c for c in CASES if not c.eager_only and c.raises is None]
_DIFFERENTIABLE = [c for c in CASES if c.differentiable]


def _compiled(method: Callable[..., Any], args: Tuple[Any, ...]) -> Any:
    return torch.compile(method, fullgraph=True, dynamic=False)(*args)


@pytest.mark.parametrize("case", _TRACEABLE, ids=case_id)
def test_compiles_fullgraph(torch_bkd: TorchBkd, case: Case) -> None:
    torch._dynamo.reset()
    run_case(torch_bkd, case, call=_compiled)


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
