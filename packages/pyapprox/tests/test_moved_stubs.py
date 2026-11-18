"""Stubs at the old paths of moved modules: generated from source, kept
in sync with the alias table, and enough for mypy --strict to type code
that imports through an old path, private names included."""

import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

import pyapprox
from pyapprox._moved import MOVED
from pyapprox._moved_stubs import HEADER, check, expected_stubs, write

_MOD = """
from typing import Optional

CONST: int = 3


class Widget:
    def __init__(self, size: int) -> None:
        self.size = size


def _helper() -> int:
    return 7


try:
    import json

    LOADED = True
except ImportError:
    LOADED = False

if True:
    def maybe(x: Optional[int]) -> int:
        return 0 if x is None else x

first, second = 1, 2
"""


@dataclass
class _Tree:
    root: Path
    old: str
    new: str

    def table(self) -> dict[str, str]:
        return {self.old: self.new, f"{self.old}_one": f"{self.new}_one"}


@pytest.fixture
def tree(tmp_path: Path) -> _Tree:
    """A moved package and a moved single module under ``tmp_path``; the
    old locations are empty."""
    tag = uuid.uuid4().hex[:8]
    old, new = f"old_{tag}", f"new_{tag}"
    (tmp_path / new / "sub").mkdir(parents=True)
    (tmp_path / new / "__init__.py").write_text("from .mod import Widget as Widget\n")
    (tmp_path / new / "mod.py").write_text(_MOD)
    (tmp_path / new / "sub" / "__init__.py").write_text(
        "import os.path\nfrom .deep import VALUE\nfrom ..mod import CONST as BASE\n"
        "__all__ = ['VALUE', 'BASE']\n"
    )
    (tmp_path / new / "sub" / "deep.py").write_text("VALUE: int = 3\n")
    (tmp_path / f"{new}_one.py").write_text("def single() -> str:\n    return 's'\n")
    return _Tree(tmp_path, old, new)


def test_stubs_reexport_every_module_level_name(tree: _Tree) -> None:
    stubs = expected_stubs(tree.table(), tree.root)
    mod = stubs[tree.root / tree.old / "mod.pyi"]
    assert mod.startswith(HEADER)
    for name in ("CONST", "Widget", "_helper", "LOADED", "maybe", "first", "second"):
        assert f"from {tree.new}.mod import {name} as {name}\n" in mod
    # Imported names are bound from where they come from, since a module
    # does not export its imports under --strict, and in the plain form,
    # since ``name as name`` would export them.
    assert "from typing import Optional\n" in mod
    assert "import json\n" in mod
    # Exported: what the module defines; imports stay bound but unexported.
    assert mod.endswith(
        "__all__ = ['CONST', 'Widget', '_helper', 'LOADED', 'maybe', 'first', "
        "'second']\n"
    )
    init = stubs[tree.root / tree.old / "__init__.pyi"]
    assert f"from {tree.new}.mod import Widget\n" in init
    assert init.endswith("__all__ = ['Widget']\n")
    sub = stubs[tree.root / tree.old / "sub" / "__init__.pyi"]
    assert "import os.path\n" in sub
    assert f"from {tree.new}.sub.deep import VALUE\n" in sub
    assert f"from {tree.new}.mod import CONST as BASE\n" in sub
    assert sub.endswith("__all__ = ['VALUE', 'BASE']\n")
    assert tree.root / tree.old / "sub" / "deep.pyi" in stubs
    assert (
        f"from {tree.new}_one import single as single\n"
        in (stubs[tree.root / f"{tree.old}_one.pyi"])
    )


def test_check_reports_missing_edited_and_stale_stubs(tree: _Tree) -> None:
    table = tree.table()
    assert all(p.startswith("missing") for p in check(table, tree.root))
    write(table, tree.root)
    assert check(table, tree.root) == []
    deep = tree.root / tree.old / "sub" / "deep.pyi"
    deep.write_text(deep.read_text() + "x = 1\n")
    assert check(table, tree.root) == [f"out of date: {deep}"]
    write(table, tree.root)
    # Dropping a table entry leaves its stubs stale until written again.
    single_only = {f"{tree.old}_one": f"{tree.new}_one"}
    assert any(p.startswith("stale") for p in check(single_only, tree.root))
    write(single_only, tree.root)
    assert check(single_only, tree.root) == []
    assert not (tree.root / tree.old / "mod.pyi").exists()
    # A hand-written stub is never touched.
    own = tree.root / "own.pyi"
    own.write_text("x: int\n")
    write({}, tree.root)
    assert own.exists() and check({}, tree.root) == []


def test_star_imports_are_refused(tree: _Tree) -> None:
    (tree.root / tree.new / "mod.py").write_text("from math import *\n")
    with pytest.raises(ValueError, match="import \\*"):
        expected_stubs(tree.table(), tree.root)
    with pytest.raises(FileNotFoundError):
        expected_stubs({"nowhere_old": "nowhere_new"}, tree.root)


def test_mypy_types_code_importing_old_paths(
    tree: _Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = pytest.importorskip("mypy.api")
    write(tree.table(), tree.root)
    lines = [
        "from {p}.mod import CONST, Widget, _helper",
        "from {p}.sub.deep import VALUE",
        "from {p}_one import single",
        "from {p} import Widget as Exported",
        "from {p}.sub import BASE",
        "widget: Widget = Exported(2)",
        "total: int = _helper() + CONST + VALUE + BASE + widget.size",
        "text: str = single()",
        "wrong: str = _helper()",
        # Bound but not exported by the new module, so an error both ways.
        "from {p}.mod import Optional",
        "from {p}.sub import os",
    ]
    for prefix in (tree.old, tree.new):
        user = tree.root / f"user_{prefix}.py"
        user.write_text("\n".join(line.format(p=prefix) for line in lines) + "\n")
    config = tree.root / "mypy.ini"
    config.write_text("[mypy]\n")
    monkeypatch.setenv("MYPYPATH", str(tree.root))
    out, _, status = api.run(
        [
            "--strict",
            "--config-file",
            str(config),
            "--cache-dir",
            str(tree.root / ".mypy_cache"),
            str(tree.root / f"user_{tree.old}.py"),
            str(tree.root / f"user_{tree.new}.py"),
        ]
    )
    # The old path types exactly as the new one: the same three errors
    # (the deliberate mistake and the two unexported names), at the same
    # lines, and nothing else.
    errors = [line for line in out.splitlines() if ": error:" in line]
    for prefix in (tree.old, tree.new):
        at = sorted(int(e.split(":")[1]) for e in errors if f"user_{prefix}.py:" in e)
        assert at == [9, 10, 11], out
    assert status == 1 and len(errors) == 6, out


def test_repository_stubs_match_the_table() -> None:
    root = Path(pyapprox.__file__).resolve().parent.parent
    assert check(MOVED, root) == []
