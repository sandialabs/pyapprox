"""Tests for keeping one OpenMP runtime per process on macOS.

The platform, the environment and torch's location are faked, so every case
runs on any operating system and none depends on what is installed beyond
numba for the numba cases.
"""

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import pytest

from pyapprox.util import openmp

NUMBA_VARS = ("NUMBA_THREADING_LAYER", "NUMBA_THREADING_LAYER_PRIORITY")


def _fake_torch(
    monkeypatch: pytest.MonkeyPatch, folder: Optional[Path], with_header: bool
) -> None:
    """Make ``find_spec("torch")`` report a package in ``folder`` (None: absent)."""
    spec: Optional[importlib.machinery.ModuleSpec] = None
    if folder is not None:
        if with_header:
            (folder / "include").mkdir()
            (folder / "include" / "omp.h").write_text("")
        spec = importlib.machinery.ModuleSpec(
            "torch", None, origin=str(folder / "__init__.py")
        )

    def find_spec(name: str) -> Optional[importlib.machinery.ModuleSpec]:
        return spec if name == "torch" else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


class TestUseTorchOpenmp:
    def test_sets_prefix_to_torch_folder_on_macos(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("OMP_PREFIX", raising=False)
        _fake_torch(monkeypatch, tmp_path, with_header=True)
        openmp.use_torch_openmp()
        assert os.environ["OMP_PREFIX"] == str(tmp_path)

    def test_keeps_user_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("OMP_PREFIX", "/user/choice")
        _fake_torch(monkeypatch, tmp_path, with_header=True)
        openmp.use_torch_openmp()
        assert os.environ["OMP_PREFIX"] == "/user/choice"

    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_does_nothing_off_macos(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str
    ) -> None:
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.delenv("OMP_PREFIX", raising=False)
        _fake_torch(monkeypatch, tmp_path, with_header=True)
        openmp.use_torch_openmp()
        assert "OMP_PREFIX" not in os.environ

    def test_does_nothing_without_torch_header(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A torch built without its bundled OpenMP (e.g. conda's) has no
        # include/omp.h; pointing inductor there would break compilation.
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("OMP_PREFIX", raising=False)
        _fake_torch(monkeypatch, tmp_path, with_header=False)
        openmp.use_torch_openmp()
        assert "OMP_PREFIX" not in os.environ

    def test_does_nothing_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("OMP_PREFIX", raising=False)
        _fake_torch(monkeypatch, None, with_header=False)
        openmp.use_torch_openmp()
        assert "OMP_PREFIX" not in os.environ


class TestKeepNumbaOffOpenmp:
    @pytest.fixture
    def numba_config(self, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        """numba's config, restored after the test; numba variables unset."""
        numba = pytest.importorskip("numba")
        monkeypatch.setattr(
            numba.config,
            "THREADING_LAYER_PRIORITY",
            list(numba.config.THREADING_LAYER_PRIORITY),
        )
        for name in NUMBA_VARS:
            monkeypatch.delenv(name, raising=False)
        config: ModuleType = numba.config
        return config

    def test_ranks_omp_last_on_macos_with_torch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        numba_config: ModuleType,
    ) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        _fake_torch(monkeypatch, tmp_path, with_header=True)
        openmp.keep_numba_off_openmp()
        assert numba_config.THREADING_LAYER_PRIORITY == [
            "tbb",
            "workqueue",
            "omp",
        ]
        # Also in the environment, which numba's compiler reloads from.
        assert os.environ["NUMBA_THREADING_LAYER_PRIORITY"] == "tbb workqueue omp"

    @pytest.mark.parametrize("name", NUMBA_VARS)
    def test_keeps_user_choice(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        numba_config: ModuleType,
        name: str,
    ) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv(name, "omp")
        _fake_torch(monkeypatch, tmp_path, with_header=True)
        before = list(numba_config.THREADING_LAYER_PRIORITY)
        openmp.keep_numba_off_openmp()
        assert numba_config.THREADING_LAYER_PRIORITY == before
        assert os.environ[name] == "omp"

    def test_does_nothing_off_macos(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        numba_config: ModuleType,
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        _fake_torch(monkeypatch, tmp_path, with_header=True)
        before = list(numba_config.THREADING_LAYER_PRIORITY)
        openmp.keep_numba_off_openmp()
        assert numba_config.THREADING_LAYER_PRIORITY == before
        assert "NUMBA_THREADING_LAYER_PRIORITY" not in os.environ

    def test_does_nothing_without_torch(
        self, monkeypatch: pytest.MonkeyPatch, numba_config: ModuleType
    ) -> None:
        # Without torch there is no second runtime to clash with, so numba
        # keeps its own default order.
        monkeypatch.setattr(sys, "platform", "darwin")
        _fake_torch(monkeypatch, None, with_header=False)
        before = list(numba_config.THREADING_LAYER_PRIORITY)
        openmp.keep_numba_off_openmp()
        assert numba_config.THREADING_LAYER_PRIORITY == before
