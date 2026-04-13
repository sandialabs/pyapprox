"""Tests verifying graceful behaviour when umbridge is not installed."""

import importlib
import sys

import pytest


@pytest.fixture()
def _hide_umbridge(monkeypatch: pytest.MonkeyPatch):
    """Temporarily make ``import umbridge`` fail, then reload client module."""
    monkeypatch.setitem(sys.modules, "umbridge", None)

    # Clear the package_available cache so import_optional_dependency
    # re-evaluates umbridge availability.
    from pyapprox.util.optional_deps import _package_cache

    _cached = _package_cache.pop("umbridge", None)

    import pyapprox.interface.umbridge.client as cmod

    importlib.reload(cmod)

    yield cmod

    # Restore cache and reload so subsequent tests see the real umbridge
    if _cached is not None:
        _package_cache["umbridge"] = _cached
    importlib.reload(cmod)


class TestUMBridgeMissing:

    @pytest.mark.usefixtures("_hide_umbridge")
    def test_module_import_succeeds_without_umbridge(self) -> None:
        """Importing the client module should never fail, even without umbridge."""

    def test_instantiation_raises_importerror_without_umbridge(
        self, _hide_umbridge,
    ) -> None:
        """UMBridgeModel() should raise ImportError with a helpful message."""
        cmod = _hide_umbridge
        from pyapprox.util.backends.numpy import NumpyBkd

        bkd = NumpyBkd()
        with pytest.raises(ImportError, match="umbridge"):
            cmod.UMBridgeModel("http://localhost:4242", "test", bkd)
