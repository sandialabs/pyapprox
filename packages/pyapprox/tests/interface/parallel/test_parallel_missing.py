"""Tests verifying graceful behaviour when joblib/mpire are not installed."""

import importlib
import sys

import pytest


@pytest.fixture()
def _hide_joblib(monkeypatch: pytest.MonkeyPatch):
    """Temporarily make ``import joblib`` fail, then reload backend module."""
    monkeypatch.setitem(sys.modules, "joblib", None)

    from pyapprox.util.optional_deps import _package_cache

    _cached = _package_cache.pop("joblib", None)

    import pyapprox.interface.parallel.joblib_backend as jmod

    importlib.reload(jmod)

    yield jmod

    if _cached is not None:
        _package_cache["joblib"] = _cached
    importlib.reload(jmod)


@pytest.fixture()
def _hide_mpire(monkeypatch: pytest.MonkeyPatch):
    """Temporarily make ``import mpire`` fail, then reload backend module."""
    monkeypatch.setitem(sys.modules, "mpire", None)

    from pyapprox.util.optional_deps import _package_cache

    _cached = _package_cache.pop("mpire", None)

    import pyapprox.interface.parallel.mpire_backend as mmod

    importlib.reload(mmod)

    yield mmod

    if _cached is not None:
        _package_cache["mpire"] = _cached
    importlib.reload(mmod)


class TestJoblibMissing:

    @pytest.mark.usefixtures("_hide_joblib")
    def test_module_import_succeeds_without_joblib(self) -> None:
        """Importing joblib_backend should never fail, even without joblib."""

    def test_map_raises_importerror_without_joblib(
        self, _hide_joblib,
    ) -> None:
        """JoblibBackend().map() should raise ImportError matching 'joblib'."""
        jmod = _hide_joblib
        backend = jmod.JoblibBackend(n_jobs=1)
        with pytest.raises(ImportError, match="joblib"):
            backend.map(lambda x: x, [1, 2])


class TestMpireMissing:

    @pytest.mark.usefixtures("_hide_mpire")
    def test_module_import_succeeds_without_mpire(self) -> None:
        """Importing mpire_backend should never fail, even without mpire."""

    def test_map_raises_importerror_without_mpire(
        self, _hide_mpire,
    ) -> None:
        """MpireBackend().map() should raise ImportError matching 'mpire'."""
        mmod = _hide_mpire
        backend = mmod.MpireBackend(n_jobs=1)
        with pytest.raises(ImportError, match="mpire"):
            backend.map(lambda x: x, [1, 2])
