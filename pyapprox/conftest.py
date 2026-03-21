"""Pytest configuration and shared fixtures."""

import os
import warnings

import numpy as np
import pytest

from pyapprox.util.backends.numpy import NumpyBkd

# ---------- env vars ----------

_RUN_SLOW = os.environ.get("PYAPPROX_RUN_SLOW", "").lower() in ("1", "true", "yes")
_RUN_SLOWER = os.environ.get("PYAPPROX_RUN_SLOWER", "").lower() in (
    "1",
    "true",
    "yes",
)
_RUN_SLOWEST = os.environ.get("PYAPPROX_RUN_SLOWEST", "").lower() in (
    "1",
    "true",
    "yes",
)


# ---------- backend discovery ----------


def _backend_factories():
    factories = [("NumpyBkd", NumpyBkd)]
    try:
        import torch  # noqa: F401

        from pyapprox.util.backends.torch import TorchBkd

        factories.append(("TorchBkd", TorchBkd))
    except ImportError:
        warnings.warn(
            "PyTorch not installed -- Torch backend tests will be skipped. "
            "Install with: pip install pyapprox[torch]",
            stacklevel=1,
        )
    return factories


_BACKEND_FACTORIES = _backend_factories()


# ---------- fixtures ----------


@pytest.fixture(params=_BACKEND_FACTORIES, ids=lambda pair: pair[0])
def bkd(request):
    """Parametrized backend fixture -- fresh instance per test."""
    name, factory = request.param
    if name == "TorchBkd":
        import torch

        torch.set_default_dtype(torch.float64)
    return factory()


@pytest.fixture
def numpy_bkd():
    """Single-backend fixture for backend-independent tests."""
    return NumpyBkd()


@pytest.fixture
def torch_bkd():
    """For tests that only make sense on Torch (e.g., autograd)."""
    torch = pytest.importorskip("torch")
    from pyapprox.util.backends.torch import TorchBkd

    torch.set_default_dtype(torch.float64)
    return TorchBkd()


@pytest.fixture
def torch_float32_bkd():
    """TorchBkd with float32 dtype for targeted precision tests."""
    torch = pytest.importorskip("torch")
    from pyapprox.util.backends.torch import TorchBkd

    return TorchBkd(dtype=torch.float32)


@pytest.fixture
def torch_mps_bkd():
    """TorchBkd on MPS device (skipped if MPS unavailable)."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    from pyapprox.util.backends.torch import TorchBkd

    return TorchBkd(device="mps", dtype=torch.float32)


@pytest.fixture(autouse=True)
def _reproducible_rng():
    """Reset NumPy (and Torch if available) RNG before every test.

    Ensures deterministic behavior regardless of test execution order.
    Tests that call np.random.seed() in their body override this.
    """
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except ImportError:
        pass


# ---------- per-backend slow-test skip logic ----------

# Mapping: marker name -> (env enabled, tier label, threshold description)
_SLOW_TIERS = {
    "slow_on": (
        _RUN_SLOW or _RUN_SLOWER or _RUN_SLOWEST,
        "Slow (>5s)",
        "PYAPPROX_RUN_SLOW",
    ),
    "slower_on": (
        _RUN_SLOWER or _RUN_SLOWEST,
        "Slower (>30s)",
        "PYAPPROX_RUN_SLOWER",
    ),
    "slowest_on": (
        _RUN_SLOWEST,
        "Slowest (>60s)",
        "PYAPPROX_RUN_SLOWEST",
    ),
}


def _extract_backend_name(item):
    """Extract backend name from a test item's parametrize ID, or None."""
    nodeid = item.nodeid
    for name, _ in _BACKEND_FACTORIES:
        if f"[{name}" in nodeid:  # handles [NumpyBkd] and [NumpyBkd-...] combos
            return name
    return None


def pytest_collection_modifyitems(config, items):
    """Skip slow tests based on env vars and per-backend markers."""
    for item in items:
        backend_name = _extract_backend_name(item)
        for marker_name, (enabled, tier_label, env_var) in _SLOW_TIERS.items():
            marker = item.get_closest_marker(marker_name)
            if marker is None:
                continue
            # marker.args contains backend names, e.g. ("*",) or ("TorchBkd",)
            target_backends = marker.args if marker.args else ("*",)
            # Check if this test's backend matches any target
            applies = "*" in target_backends or (
                backend_name is not None and backend_name in target_backends
            )
            if applies and not enabled:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{tier_label} on {backend_name or 'all'}. "
                        f"Set {env_var}=1 to run."
                    )
                )
