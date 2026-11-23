"""The optional-import ratchet: test modules that import torch, JAX or
Numba at module level, and tests and fixtures that import one in their
body, without skipping first are counted and held at the baselines in
ratchet_baselines.txt; skipping fixtures redefined outside the root
conftest.py are listed."""

from pathlib import Path

import pytest

import pyapprox
from pyapprox._moved_ratchet import BASELINE_FILE
from pyapprox._optional_import_ratchet import (
    FUNCTIONS,
    MODULES,
    Findings,
    _library_imports,
    bound_modules,
    check,
    ratchet_notices,
    ratchet_problems,
    read_baseline,
    scan_tests,
    shadowing_problems,
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
    # A skip inside a function does not skip the module.
    _write(
        tmp_path,
        f"{tests}/test_late.py",
        "import pytest\ndef test_a():\n    pytest.importorskip('torch')\n"
        "import torch\n",
    )
    _write(tmp_path, "packages/pkg/scripts/run.py", "import torch\n")  # not a test
    findings = _scan(tmp_path)
    names = {
        lib: sorted(Path(p).name for p in found)
        for lib, found in findings.modules.items()
    }
    assert names == {
        "torch": ["test_bound.py", "test_late.py", "test_raw.py"],
        "numba": [],
    }
    # The import in test_x follows the torch_bkd fixture, which skips.
    assert findings.functions == {"torch": [], "numba": []}


def _scan(root: Path) -> Findings:
    imports = _library_imports(root)
    return scan_tests(
        root, {lib: bound_modules(lib, imports) for lib in ("torch", "numba")}
    )


# Each body imports torch (or a module bound to it) in a test or fixture;
# the first group fails without torch, the second skips first.
_UNGUARDED = {
    "test_param.py": "def test_a(bkd):\n    import torch\n",
    "test_branch.py": (
        "class TestC:\n    def test_m(self, bkd):\n"
        "        if bkd.__class__.__name__ != 'NumpyBkd':\n"
        "            import torch\n"
    ),
    "test_fixture.py": (
        "import pytest\n@pytest.fixture\ndef model():\n    from pkg.through import f\n"
    ),
    "test_nested.py": ("def test_n(bkd):\n    def inner():\n        import torch.nn\n"),
    "test_late.py": (
        "import pytest\ndef test_late():\n    import torch\n"
        "    pytest.importorskip('torch')\n"
    ),
}
_GUARDED = {
    "test_torch_bkd.py": "def test_a(torch_bkd):\n    import torch\n",
    "test_autodiff.py": "def test_a(autodiff_bkd):\n    import torch\n",
    "test_importorskip.py": (
        "import pytest\ndef test_a():\n"
        "    torch = pytest.importorskip('torch')\n    import torch.nn\n"
    ),
    "test_available.py": (
        "import pytest\nfrom pkg.util import package_available\n"
        "def test_a():\n    if not package_available('torch'):\n"
        "        pytest.skip('no torch')\n    import torch\n"
    ),
    "test_skipif.py": (
        "import pytest\n@pytest.mark.skipif(not HAS_TORCH, reason='')\n"
        "def test_a():\n    import torch\n"
    ),
    "test_class_skipif.py": (
        "import pytest\n@pytest.mark.skipif(not HAS_TORCH, reason='')\n"
        "class TestC:\n    def test_m(self):\n        import torch\n"
    ),
    "test_module_skip.py": (
        "import pytest\ndef test_a():\n    import torch\npytest.importorskip('torch')\n"
    ),
    "test_recovers.py": (
        "import pytest\ndef test_a():\n    try:\n        import torch\n"
        "    except ImportError:\n        pytest.skip('no torch')\n"
    ),
    "test_helper.py": "def _make():\n    import torch\n",
    "test_lazy.py": "def test_a(bkd):\n    from pkg.lazy import f\n",
}


def test_counts_unguarded_tests_and_fixtures(tmp_path: Path) -> None:
    _library(tmp_path)
    for name, text in {**_UNGUARDED, **_GUARDED}.items():
        _write(tmp_path, f"packages/pkg/tests/{name}", text)
    _write(
        tmp_path,
        "packages/pkg/tests/test_numba.py",
        "def test_a(torch_bkd):\n    from pkg.fast import f\n",
    )
    # Counted at module level, so its functions are not counted again.
    _write(
        tmp_path,
        "packages/pkg/tests/test_both.py",
        "import torch\ndef test_a(bkd):\n    import torch\n",
    )
    findings = _scan(tmp_path)
    assert findings.modules["torch"] == ["packages/pkg/tests/test_both.py"]
    assert findings.functions == {
        "torch": [
            "packages/pkg/tests/test_branch.py::TestC.test_m",
            "packages/pkg/tests/test_fixture.py::model",
            "packages/pkg/tests/test_late.py::test_late",
            "packages/pkg/tests/test_nested.py::test_n",
            "packages/pkg/tests/test_param.py::test_a",
        ],
        # torch_bkd skips without torch, not without numba.
        "numba": ["packages/pkg/tests/test_numba.py::test_a"],
    }


def test_lists_skipping_fixtures_outside_the_root_conftest(tmp_path: Path) -> None:
    fixture = "import pytest\n@pytest.fixture\ndef {name}():\n    pass\n"
    _write(tmp_path, "conftest.py", fixture.format(name="torch_bkd"))
    _write(
        tmp_path, "packages/pkg/tests/sub/conftest.py", fixture.format(name="torch_bkd")
    )
    _write(tmp_path, "packages/pkg/conftest.py", fixture.format(name="autodiff_bkd"))
    _write(
        tmp_path,
        "packages/pkg/tests/test_a.py",
        fixture.format(name="torch_float32_bkd") + "def torch_mps_bkd():\n    pass\n",
    )  # not a fixture
    _write(tmp_path, "packages/pkg/tests/test_b.py", fixture.format(name="bkd"))
    shadowing = _scan(tmp_path).shadowing
    assert sorted(shadowing) == [
        "packages/pkg/conftest.py::autodiff_bkd",
        "packages/pkg/tests/sub/conftest.py::torch_bkd",
        "packages/pkg/tests/test_a.py::torch_float32_bkd",
    ]
    assert shadowing_problems(["t/conftest.py::torch_bkd"]) == [
        "t/conftest.py::torch_bkd redefines a root conftest.py fixture that "
        "skips without its library; use the root one"
    ]


def test_ratchet_holds_counts_at_their_baselines(tmp_path: Path) -> None:
    _write(
        tmp_path,
        BASELINE_FILE,
        "mypy_src=1\nsuperseded-method:a=3\n"
        "optional-test-import:torch=1\noptional-test-import:gone=0\n"
        "optional-test-import-in-function:torch=4\n",
    )
    baseline = read_baseline(tmp_path / BASELINE_FILE, MODULES)
    assert baseline == {"torch": 1, "gone": 0}
    assert read_baseline(tmp_path / BASELINE_FILE, FUNCTIONS) == {"torch": 4}
    problems = ratchet_problems(
        MODULES, ("torch", "jax"), {"torch": ["t/a.py", "t/b.py"], "jax": []}, baseline
    )
    assert problems == [
        "torch: 2 test modules import it at module level, baseline 1; reach "
        "it through a backend fixture, or skip the module first with "
        'pytest.importorskip("torch"):\n  t/a.py\n  t/b.py',
        "no baseline for jax: add optional-test-import:jax=0",
        "optional-test-import:gone is stale: gone is not checked",
    ]
    rose = ratchet_problems(
        FUNCTIONS, ("torch",), {"torch": ["t/a.py::f"]}, {"torch": 0}
    )
    assert rose == [
        "torch: 1 tests or fixtures import it without skipping first, "
        "baseline 0; take a backend fixture that skips without it "
        '(torch_bkd, autodiff_bkd), or call pytest.importorskip("torch") '
        "before the import:\n  t/a.py::f"
    ]
    # A count below its baseline does not fail; it is a notice to lower it.
    assert ratchet_problems(MODULES, ("torch",), {"torch": []}, {"torch": 1}) == []
    fell = ratchet_notices(MODULES, ("torch",), {"torch": []}, {"torch": 1})
    assert fell == ["torch: down to 0; set optional-test-import:torch=0"]
    fell = ratchet_notices(FUNCTIONS, ("torch",), {"torch": []}, {"torch": 1})
    assert fell == ["torch: down to 0; set optional-test-import-in-function:torch=0"]
    one = {"torch": ["t/a.py"]}
    assert ratchet_problems(MODULES, ("torch",), one, {"torch": 1}) == []
    assert ratchet_notices(MODULES, ("torch",), one, {"torch": 1}) == []
    # Neither a rise nor a missing baseline is a notice.
    two = {"torch": ["t/a.py", "t/b.py"]}
    assert ratchet_notices(MODULES, ("torch", "jax"), two, baseline) == []


def test_repository_holds_its_baselines() -> None:
    # src/pyapprox/__init__.py -> the repository root holding the baselines.
    root = Path(pyapprox.__file__).resolve().parents[4]
    if not (root / BASELINE_FILE).exists():
        pytest.skip("not a source checkout")
    problems, _ = check(root)
    assert problems == []
