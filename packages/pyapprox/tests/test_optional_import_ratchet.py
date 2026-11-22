"""The optional-import ratchet: test modules that import torch, JAX or
Numba at module level without skipping first are counted and held at the
baselines in ratchet_baselines.txt."""

from pathlib import Path

import pytest

import pyapprox
from pyapprox._moved_ratchet import BASELINE_FILE
from pyapprox._optional_import_ratchet import (
    OPTIONAL,
    _library_imports,
    bound_modules,
    measure,
    ratchet_notices,
    ratchet_problems,
    read_baseline,
    unguarded_test_modules,
)


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _library(root: Path) -> None:
    src = "packages/pkg/src/pkg"
    _write(root, f"{src}/__init__.py", "")
    _write(root, f"{src}/direct.py", "import torch\n")
    _write(root, f"{src}/through.py", "from pkg.direct import f\n")
    _write(root, f"{src}/lazy.py", "def f():\n    import torch\n")
    _write(
        root,
        f"{src}/guarded.py",
        "try:\n    import torch\nexcept ImportError:\n    pass\n",
    )
    _write(
        root,
        f"{src}/reraised.py",
        "try:\n    import torch\nexcept ImportError as err:\n"
        "    raise ImportError('pip install pkg[torch]') from err\n",
    )
    _write(root, f"{src}/fast.py", "from numba import njit\n")


def test_bound_modules_follow_module_level_imports(tmp_path: Path) -> None:
    _library(tmp_path)
    imports = _library_imports(tmp_path)
    # Imports in functions and in try blocks that recover do not bind a
    # module; a try that re-raises does.
    assert bound_modules("torch", imports) == {
        "pkg.direct",
        "pkg.through",
        "pkg.reraised",
    }
    assert bound_modules("numba", imports) == {"pkg.fast"}
    assert bound_modules("jax", imports) == frozenset()


def test_counts_unguarded_test_modules(tmp_path: Path) -> None:
    _library(tmp_path)
    tests = "packages/pkg/tests"
    _write(tmp_path, f"{tests}/test_raw.py", "import numpy\nimport torch\n")
    _write(tmp_path, f"{tests}/test_bound.py", "from pkg.through import f\n")
    _write(
        tmp_path,
        f"{tests}/test_skip.py",
        'import pytest\npytest.importorskip("torch")\nimport torch\n',
    )
    _write(
        tmp_path,
        f"{tests}/test_numba.py",
        "import pytest\nfrom pkg.util import package_available\n"
        "if not package_available('numba'):\n"
        "    pytest.skip('no numba', allow_module_level=True)\n"
        "from pkg.fast import f\n",
    )
    _write(
        tmp_path,
        f"{tests}/test_fixture.py",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch\n"
        "def test_x(torch_bkd):\n    import torch\n",
    )
    _write(tmp_path, "packages/pkg/scripts/run.py", "import torch\n")  # not a test
    imports = _library_imports(tmp_path)
    bound = {lib: bound_modules(lib, imports) for lib in ("torch", "numba")}
    found = unguarded_test_modules(tmp_path, bound)
    names = {lib: sorted(p.name for p in paths) for lib, paths in found.items()}
    assert names == {"torch": ["test_bound.py", "test_raw.py"], "numba": []}


def test_ratchet_holds_counts_at_their_baselines(tmp_path: Path) -> None:
    _write(
        tmp_path,
        BASELINE_FILE,
        "mypy_src=1\nsuperseded-method:a=3\n"
        "optional-test-import:torch=1\noptional-test-import:gone=0\n",
    )
    baseline = read_baseline(tmp_path / BASELINE_FILE)
    assert baseline == {"torch": 1, "gone": 0}
    problems = ratchet_problems(
        ("torch", "jax"), {"torch": ["t/a.py", "t/b.py"], "jax": []}, baseline
    )
    assert problems == [
        "torch: 2 test modules import it at module level, baseline 1; reach "
        "it through a backend fixture, or skip the module first with "
        'pytest.importorskip("torch"):\n  t/a.py\n  t/b.py',
        "no baseline for jax: add optional-test-import:jax=0",
        "optional-test-import:gone is stale: gone is not checked",
    ]
    # A count below its baseline does not fail; it is a notice to lower it.
    assert ratchet_problems(("torch",), {"torch": []}, {"torch": 1}) == []
    fell = ratchet_notices(("torch",), {"torch": []}, {"torch": 1})
    assert fell == ["torch: down to 0; set optional-test-import:torch=0"]
    assert ratchet_problems(("torch",), {"torch": ["t/a.py"]}, {"torch": 1}) == []
    assert ratchet_notices(("torch",), {"torch": ["t/a.py"]}, {"torch": 1}) == []
    # Neither a rise nor a missing baseline is a notice.
    assert (
        ratchet_notices(("torch", "jax"), {"torch": ["t/a.py", "t/b.py"]}, baseline)
        == []
    )


def test_repository_holds_its_baselines() -> None:
    # src/pyapprox/__init__.py -> the repository root holding the baselines.
    root = Path(pyapprox.__file__).resolve().parents[4]
    if not (root / BASELINE_FILE).exists():
        pytest.skip("not a source checkout")
    modules, baseline = measure(root)
    assert ratchet_problems(OPTIONAL, modules, baseline) == []
