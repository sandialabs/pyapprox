"""The alias finder for moved modules, on a throwaway package moved in a
fixture: the old path gives the same module objects as the new one, so
identity, isinstance, mock.patch strings and old pickles keep working."""

import importlib
import pickle
import sys
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest

import pyapprox
from pyapprox._moved import MOVED, AliasFinder, install

_MOD = """
class Widget:
    def __init__(self, size: int) -> None:
        self.size = size


def measure() -> str:
    return "new"


def call_measure() -> str:
    return measure()


def _helper() -> int:
    return 7
"""


@dataclass
class _Moved:
    old: str
    new: str
    finder: AliasFinder


def _cleanup(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if name.split(".")[0] in prefixes:
            del sys.modules[name]


@pytest.fixture
def moved(tmp_path: Path) -> Iterator[_Moved]:
    """``old_<id>`` moved to ``new_<id>``; the old directory keeps only a
    stub and no ``__init__.py``, as after a real move."""
    tag = uuid.uuid4().hex[:8]
    old, new = f"old_{tag}", f"new_{tag}"
    (tmp_path / new / "sub").mkdir(parents=True)
    (tmp_path / new / "__init__.py").write_text("")
    (tmp_path / new / "mod.py").write_text(_MOD)
    (tmp_path / new / "sub" / "__init__.py").write_text("")
    (tmp_path / new / "sub" / "deep.py").write_text("VALUE = 3\n")
    (tmp_path / old).mkdir()
    (tmp_path / old / "mod.pyi").write_text(f"from {new}.mod import Widget\n")
    sys.path.insert(0, str(tmp_path))
    table = {old: new}
    finder = install(table)
    try:
        yield _Moved(old, new, finder)
    finally:
        sys.meta_path.remove(finder)
        sys.path.remove(str(tmp_path))
        _cleanup((old, new))


def test_old_paths_give_the_same_module_objects(moved: _Moved) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_mod = importlib.import_module(f"{moved.old}.mod")
        old_deep = importlib.import_module(f"{moved.old}.sub.deep")
        old_pkg = importlib.import_module(moved.old)
    assert old_mod is importlib.import_module(f"{moved.new}.mod")
    assert old_deep is importlib.import_module(f"{moved.new}.sub.deep")
    assert old_pkg is importlib.import_module(moved.new)
    assert old_deep.VALUE == 3 and old_mod._helper() == 7
    widget = old_mod.Widget(2)
    assert isinstance(widget, importlib.import_module(f"{moved.new}.mod").Widget)


def test_one_warning_per_prefix(moved: _Moved) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module(f"{moved.new}.mod")
        assert not caught
        importlib.import_module(f"{moved.old}.mod")
        importlib.import_module(f"{moved.old}.sub.deep")
    deprecations = [w for w in caught if w.category is DeprecationWarning]
    assert len(deprecations) == 1
    assert moved.new in str(deprecations[0].message)


def test_patching_the_old_target_reaches_the_new_module(moved: _Moved) -> None:
    new_mod = importlib.import_module(f"{moved.new}.mod")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with mock.patch(f"{moved.old}.mod.measure", return_value="patched"):
            assert new_mod.call_measure() == "patched"
    assert new_mod.call_measure() == "new"


def test_objects_pickled_under_the_old_path_unpickle(moved: _Moved) -> None:
    new_mod = importlib.import_module(f"{moved.new}.mod")
    widget_class = new_mod.Widget
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # Pickle as code from before the move would have: the class's
        # module recorded as the old path.
        widget_class.__module__ = f"{moved.old}.mod"
        try:
            data = pickle.dumps(widget_class(5))
        finally:
            widget_class.__module__ = f"{moved.new}.mod"
        assert f"{moved.old}.mod".encode() in data
        restored = pickle.loads(data)
    assert type(restored) is widget_class and restored.size == 5


def test_finder_must_come_first(moved: _Moved) -> None:
    # Appended after the path finder, the finder never sees the old name:
    # the stub-only old directory imports as an empty namespace package.
    sys.meta_path.remove(moved.finder)
    sys.meta_path.append(moved.finder)
    namespace = importlib.import_module(moved.old)
    assert namespace is not importlib.import_module(moved.new)
    assert not hasattr(namespace, "__file__") or namespace.__file__ is None


def test_unrelated_names_pass_through(moved: _Moved) -> None:
    assert moved.finder.find_spec(moved.old + "x", None) is None
    assert moved.finder.find_spec(moved.new + ".mod", None) is None


def test_longest_prefix_wins(moved: _Moved) -> None:
    # The shorter prefix points nowhere; only the longer one resolves.
    finder = AliasFinder(
        {moved.old: "missing_package", f"{moved.old}.sub": f"{moved.new}.sub"}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        spec = finder.find_spec(f"{moved.old}.sub.deep", None)
    assert spec is not None and spec.loader is not None
    module = spec.loader.create_module(spec)
    assert module is importlib.import_module(f"{moved.new}.sub.deep")


def test_pyapprox_installs_its_table_once() -> None:
    assert pyapprox.name == "pyapprox"
    ours = install(MOVED)
    assert [f for f in sys.meta_path if f is ours] == [ours]
    path_finders = [
        i for i, f in enumerate(sys.meta_path) if f is importlib.machinery.PathFinder
    ]
    assert sys.meta_path.index(ours) < path_finders[0]
