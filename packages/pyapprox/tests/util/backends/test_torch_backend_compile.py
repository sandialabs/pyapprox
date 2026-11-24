"""Torch backend methods compile into one graph under torch.compile, and
what ``TorchBkd.jit`` adds: compilation off by default, a cache per
kernel and options, ``fullgraph=True``, pickling without the cache.

Torch is optional, so the whole module is skipped without it.
"""

import functools
import pickle

import pytest

# Skip the module without torch; the imports below are then ordinary ones.
pytest.importorskip("torch")

import torch

from pyapprox.util.backends.torch import TorchBkd


class TestBooleanReductionsCompile:
    """``any`` and ``all`` stay inside the graph (no host sync), so a
    function using them compiles with ``fullgraph=True``."""

    def test_any_and_all_compile_fullgraph(self, torch_bkd: TorchBkd) -> None:
        bkd = torch_bkd

        def masked(array: torch.Tensor) -> torch.Tensor:
            rows_ok = bkd.all(array > 0.0, axis=1, keepdims=True)
            some = bkd.any(array < -1.0)
            return bkd.where(rows_ok & ~some, array, bkd.zeros_like(array))

        compiled = torch.compile(masked, fullgraph=True)
        array = bkd.asarray([[1.0, 2.0], [-0.5, 3.0]])
        bkd.assert_allclose(compiled(array), masked(array))
        bkd.assert_allclose(compiled(array), bkd.asarray([[1.0, 2.0], [0.0, 0.0]]))


def _scaled(x: torch.Tensor, *, bkd: TorchBkd) -> torch.Tensor:
    return bkd.exp(x) * 2.0


def _host_branch(x: torch.Tensor, *, bkd: TorchBkd) -> torch.Tensor:
    # to_bool leaves the graph (a host sync): a graph break.
    if bkd.to_bool(bkd.any(x > 0.0)):
        return x
    return -x


@pytest.fixture
def compiling_bkd() -> TorchBkd:
    torch._dynamo.reset()
    return TorchBkd(compile=True)


class TestTorchJit:
    """What ``TorchBkd.jit`` does beyond the other backends."""

    def test_uncompiled_by_default(self, torch_bkd: TorchBkd) -> None:
        # compile=False keeps every generic kernel twice differentiable.
        kernel = torch_bkd.jit(_scaled)
        assert isinstance(kernel, functools.partial)
        assert kernel.keywords == {"bkd": torch_bkd}

    def test_compiles_once_per_kernel_and_options(
        self, compiling_bkd: TorchBkd
    ) -> None:
        first = compiling_bkd.jit(_scaled)
        assert not isinstance(first, functools.partial)
        assert compiling_bkd.jit(_scaled) is first
        assert compiling_bkd.jit(_scaled, dynamic=True) is not first
        x = compiling_bkd.asarray([0.0, 1.0])
        compiling_bkd.assert_allclose(first(x), _scaled(x, bkd=compiling_bkd))

    def test_graph_break_raises(self, compiling_bkd: TorchBkd) -> None:
        # torch 2.7 reports the host sync as Unsupported (UserError is a
        # subclass); later releases raise UserError outside that hierarchy.
        errors = (torch._dynamo.exc.Unsupported, torch._dynamo.exc.UserError)
        with pytest.raises(errors):
            compiling_bkd.jit(_host_branch)(compiling_bkd.asarray([1.0, -2.0]))

    def test_pickles_without_its_cache(self, compiling_bkd: TorchBkd) -> None:
        compiling_bkd.jit(_scaled)
        restored = pickle.loads(pickle.dumps(compiling_bkd))
        x = restored.asarray([0.5])
        restored.assert_allclose(restored.jit(_scaled)(x), _scaled(x, bkd=restored))
