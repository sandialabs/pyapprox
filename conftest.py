"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union, cast

import numpy as np
import pytest

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.precision import Precision
from tests._helpers.markers import matching_target

if TYPE_CHECKING:
    from pyapprox.util.backends.jax import JaxBkd
    from pyapprox.util.backends.torch import TorchBkd

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


def _backend_factories() -> List[Tuple[str, type]]:
    factories: List[Tuple[str, type]] = [("NumpyBkd", NumpyBkd)]
    try:
        import torch  # noqa: F401

        from pyapprox.util.backends.torch import TorchBkd

        factories.append(("TorchBkd", TorchBkd))
    except ImportError:
        warnings.warn(
            "PyTorch not installed -- Torch backend tests will be skipped. "
            "Install with: pip install torch",
            stacklevel=1,
        )
    return factories


_BACKEND_FACTORIES = _backend_factories()


# ---------- fixtures ----------


@pytest.fixture(params=_BACKEND_FACTORIES, ids=lambda pair: pair[0])
def bkd(request: pytest.FixtureRequest) -> Union[NumpyBkd, TorchBkd]:
    """Parametrized backend fixture -- fresh instance per test."""
    _, factory = cast(Tuple[str, type], request.param)
    return factory()  # type: ignore[no-any-return]


def _configured_bkd(
    name: str, precision: Precision, device: str
) -> Union[NumpyBkd, TorchBkd, JaxBkd]:
    if name == "NumpyBkd":
        return NumpyBkd(precision=precision)
    if name == "JaxBkd":
        return _configured_jax_bkd(precision, device)
    torch = pytest.importorskip("torch")
    from pyapprox.util.backends.torch import TorchBkd

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    return TorchBkd(precision=precision, device=device)


def _configured_jax_bkd(precision: Precision, device: str) -> JaxBkd:
    jax = pytest.importorskip("jax")
    from pyapprox.util.backends.jax import JaxBkd

    if device != "cpu" and not any(d.platform == device for d in jax.devices()):
        pytest.skip(f"no JAX {device} device")
    # 64-bit types are process-wide in JAX and the caller's to enable;
    # the test session enables them once, as a user would.
    jax.config.update("jax_enable_x64", True)
    return JaxBkd(precision=precision, device=device)


# The configurations config_bkd runs over: backend x precision x device.
# MPS has no float64, so it runs single precision only.
_CONFIGURATIONS: List[Tuple[str, Precision, str]] = [
    (name, precision, "cpu")
    for name in ("NumpyBkd", "TorchBkd", "JaxBkd")
    for precision in (Precision.DOUBLE, Precision.SINGLE)
] + [
    ("TorchBkd", Precision.DOUBLE, "cuda"),
    ("TorchBkd", Precision.SINGLE, "cuda"),
    ("TorchBkd", Precision.SINGLE, "mps"),
]


def _configuration_id(config: Tuple[str, Precision, str]) -> str:
    # The CPU is the default, so its configurations carry no device.
    name, precision, device = config
    suffix = "" if device == "cpu" else f"-{device}"
    return f"{name}-{precision.value}{suffix}"


@pytest.fixture(params=_CONFIGURATIONS, ids=_configuration_id)
def config_bkd(request: pytest.FixtureRequest) -> Union[NumpyBkd, TorchBkd, JaxBkd]:
    """Opt-in backend fixture over every configuration (backend x
    precision x device); a library or device that is not available is
    skipped. The default ``bkd`` fixture stays double precision on the
    CPU."""
    name, precision, device = cast(Tuple[str, Precision, str], request.param)
    return _configured_bkd(name, precision, device)


@pytest.fixture
def numpy_bkd() -> NumpyBkd:
    """Single-backend fixture for backend-independent tests."""
    return NumpyBkd()


def _make_torch_bkd() -> "TorchBkd":
    """Float64 TorchBkd; skips the test when torch is not installed."""
    pytest.importorskip("torch")
    from pyapprox.util.backends.torch import TorchBkd

    return TorchBkd()


# Backends the ``autodiff_bkd`` fixture runs on: each satisfies
# AutodiffBackend, and each factory skips when its library is missing.
_AUTODIFF_BACKEND_FACTORIES: List[Tuple[str, Callable[[], "TorchBkd"]]] = [
    ("TorchBkd", _make_torch_bkd),
]


@pytest.fixture(params=_AUTODIFF_BACKEND_FACTORIES, ids=lambda pair: pair[0])
def autodiff_bkd(request: pytest.FixtureRequest) -> "TorchBkd":
    """Parametrized over the autodiff backends.

    For tests that need ``bkd.jacobian`` or ``bkd.hvp`` and nothing
    specific to one library; annotate the argument
    ``AutodiffBackend[Array]``. Tests that are torch-only by construction
    (``torch.compile``, ``torch.nn``, a torch-bound class) take
    ``torch_bkd`` instead.
    """
    _, factory = cast(Tuple[str, Callable[[], "TorchBkd"]], request.param)
    return factory()


@pytest.fixture
def torch_bkd() -> "TorchBkd":
    """For tests that only make sense on Torch (``torch.compile``,
    ``torch.nn``, a torch-bound class). Tests that only need autodiff take
    ``autodiff_bkd``."""
    return _make_torch_bkd()


@pytest.fixture
def torch_float32_bkd() -> "TorchBkd":
    """TorchBkd at single precision for targeted precision tests."""
    pytest.importorskip("torch")
    from pyapprox.util.backends.torch import TorchBkd

    return TorchBkd(precision=Precision.SINGLE)


@pytest.fixture
def torch_mps_bkd() -> "TorchBkd":
    """TorchBkd on MPS device (skipped if MPS unavailable)."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    from pyapprox.util.backends.torch import TorchBkd

    return TorchBkd(device="mps", precision=Precision.SINGLE)


@pytest.fixture(autouse=True)
def _reproducible_rng() -> None:
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


def _parametrize_id(item: pytest.Item) -> Optional[str]:
    """The item's combined parametrize id, or None if not parametrized."""
    callspec = getattr(item, "callspec", None)
    return None if callspec is None else str(callspec.id)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: List[pytest.Item],
) -> None:
    """Skip slow tests based on env vars and per-backend markers."""
    for item in items:
        for marker_name, (enabled, tier_label, env_var) in _SLOW_TIERS.items():
            marker = item.get_closest_marker(marker_name)
            if marker is None:
                continue
            # marker.args names backends or stack cases, e.g. ("*",),
            # ("TorchBkd",) or ("dense-torch", "mock").
            targets = tuple(marker.args) if marker.args else ("*",)
            matched = (
                None
                if "*" in targets
                else matching_target(_parametrize_id(item), targets)
            )
            if ("*" in targets or matched is not None) and not enabled:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{tier_label} on {matched or 'all'}. "
                        f"Set {env_var}=1 to run."
                    )
                )
