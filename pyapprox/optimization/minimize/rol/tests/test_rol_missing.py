"""Tests verifying graceful behaviour when pyrol is not installed."""

import importlib
import sys

import pytest


@pytest.fixture()
def _hide_pyrol(monkeypatch: pytest.MonkeyPatch):
    """Temporarily make ``import pyrol`` fail, then reload ROL modules."""
    monkeypatch.setitem(sys.modules, "pyrol", None)
    monkeypatch.setitem(sys.modules, "pyrol.vectors", None)

    import pyapprox.optimization.minimize.rol.rol_wrappers as wmod
    import pyapprox.optimization.minimize.rol.rol_optimizer as omod
    import pyapprox.optimization.minimize.rol.rol_result as rmod

    importlib.reload(wmod)
    importlib.reload(omod)
    importlib.reload(rmod)

    yield wmod, omod, rmod

    # Reload so subsequent tests in the session see the real pyrol
    importlib.reload(wmod)
    importlib.reload(omod)


class TestROLMissing:

    @pytest.mark.usefixtures("_hide_pyrol")
    def test_module_import_succeeds_without_pyrol(self) -> None:
        """Importing the ROL modules should never fail, even without pyrol."""

    def test_instantiation_raises_importerror_without_pyrol(
        self, _hide_pyrol,  # type: ignore[no-untyped-def]
    ) -> None:
        """ROLOptimizer() should raise ImportError with a helpful message."""
        _wmod, omod, _rmod = _hide_pyrol
        with pytest.raises(ImportError, match="rol-python"):
            omod.ROLOptimizer(verbosity=0)

    def test_require_pyrol_message(
        self, _hide_pyrol,  # type: ignore[no-untyped-def]
    ) -> None:
        """_require_pyrol should mention 'pip install rol-python'."""
        wmod, _omod, _rmod = _hide_pyrol
        with pytest.raises(ImportError, match="pip install rol-python"):
            wmod._require_pyrol()
