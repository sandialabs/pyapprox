"""Keeping test modules collectable without an optional library.

Torch, JAX and Numba are optional: a test module that imports one at
module level, without first skipping when it is missing, fails collection
in an install without it, and so does one that imports a library module
which imports it at load time (``pyapprox.util.backends.torch``, the
``*_torch.py`` and ``*_numba.py`` kernels, ...). Tests reach a backend
through the ``bkd``, ``autodiff_bkd`` and ``torch_bkd`` fixtures instead;
a module that needs the library by construction skips at module level
before importing it, with ``pytest.importorskip("<lib>")`` or
``if not package_available("<lib>"): pytest.skip(..., allow_module_level=True)``.

:data:`OPTIONAL` names the libraries. :func:`bound_modules` finds the
library modules under ``packages/*/src`` that import one at load time,
transitively. :func:`unguarded_test_modules` lists the test modules whose
module-level imports reach it before such a skip. Imports inside
functions, ``if`` blocks (``TYPE_CHECKING``) and ``try`` blocks that
recover do not count; a ``try`` whose handlers re-raise does. Each count
is recorded in ``ratchet_baselines.txt`` as
``optional-test-import:<lib>=<count>``; under that file's rule a commit
that lowers a count lowers its baseline in the same commit, and no commit
raises one.

``python -m pyapprox._optional_import_ratchet`` exits nonzero when a
count exceeds its baseline (listing the modules), a library has none, or
a baseline is stale. A count below its baseline is printed as a notice to
lower it and does not fail, since merging branches lowers counts no
single branch could record.
"""

import ast
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pyapprox._moved_ratchet import BASELINE_FILE, _files

OPTIONAL = ("torch", "jax", "numba")
BASELINE_KEY = "optional-test-import:"


def _reraises(node: ast.Try) -> bool:
    """Whether every handler of ``node`` ends by raising, so the import in
    its body still fails the module when the library is missing."""
    return bool(node.handlers) and all(
        isinstance(handler.body[-1], ast.Raise) for handler in node.handlers
    )


def _module_imports(nodes: Iterable[ast.stmt]) -> list[str]:
    """Dotted names imported by the import statements among ``nodes``,
    including those in a ``try`` whose handlers re-raise (the
    :func:`~pyapprox.util.optional_deps.missing_dependency_message`
    guard)."""
    names: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Try) and _reraises(node):
            names += _module_imports(node.body)
        elif isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
            names += [f"{node.module}.{alias.name}" for alias in node.names]
    return names


def _reaches(name: str, lib: str, bound: Iterable[str]) -> bool:
    return name == lib or name.startswith(f"{lib}.") or name in bound


def _library_imports(root: Path) -> dict[str, list[str]]:
    """Dotted module name -> its module-level imports, for ``packages/*/src``."""
    imports: dict[str, list[str]] = {}
    for src in sorted((root / "packages").glob("*/src")):
        for path in sorted(src.rglob("*.py")):
            parts = list(path.relative_to(src).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if parts:
                tree = ast.parse(path.read_text(errors="ignore"))
                imports[".".join(parts)] = _module_imports(tree.body)
    return imports


def bound_modules(
    lib: str, library_imports: Mapping[str, Sequence[str]]
) -> frozenset[str]:
    """Library modules that import ``lib`` when they are imported, directly
    or through another such module."""
    bound: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, names in library_imports.items():
            if name not in bound and any(_reaches(n, lib, bound) for n in names):
                bound.add(name)
                changed = True
    return frozenset(bound)


def _guards(segment: str, lib: str) -> bool:
    """Whether a module-level statement skips the module without ``lib``."""
    text = segment.replace("'", '"')
    return f'importorskip("{lib}")' in text or (
        f'package_available("{lib}")' in text and "skip(" in text
    )


def _unguarded(tree: ast.Module, source: str, lib: str, bound: frozenset[str]) -> bool:
    for node in tree.body:
        if _guards(ast.get_source_segment(source, node) or "", lib):
            return False
        if any(_reaches(n, lib, bound) for n in _module_imports([node])):
            return True
    return False


def unguarded_test_modules(
    root: Path, bound: Mapping[str, frozenset[str]]
) -> dict[str, list[Path]]:
    """Per library in ``bound``, the test modules under ``root`` (files of a
    ``tests`` directory) whose module-level imports reach it before a
    module-level skip."""
    found: dict[str, list[Path]] = {lib: [] for lib in bound}
    for path in _files(root):
        if path.suffix != ".py" or "tests" not in path.relative_to(root).parts[:-1]:
            continue
        source = path.read_text(errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for lib, modules in bound.items():
            if _unguarded(tree, source, lib, modules):
                found[lib].append(path)
    return found


def read_baseline(path: Path) -> dict[str, int]:
    """The ``optional-test-import:<lib>=<count>`` entries of a baseline file."""
    baseline: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if line.startswith(BASELINE_KEY) and "=" in line:
            key, value = line[len(BASELINE_KEY) :].split("=", 1)
            baseline[key.strip()] = int(value)
    return baseline


def ratchet_problems(
    libs: Sequence[str],
    modules: Mapping[str, Sequence[str]],
    baseline: Mapping[str, int],
) -> list[str]:
    """What fails the ratchet, given the unguarded modules (relative paths,
    per library): a count above its baseline, a library without one, and
    baselines of libraries no longer in ``libs``. A count below its
    baseline is a notice (:func:`ratchet_notices`)."""
    problems = []
    for lib in libs:
        found = modules.get(lib, [])
        count, key = len(found), f"{BASELINE_KEY}{lib}"
        if lib not in baseline:
            problems.append(f"no baseline for {lib}: add {key}={count}")
        elif count > baseline[lib]:
            listed = "".join(f"\n  {p}" for p in found)
            problems.append(
                f"{lib}: {count} test modules import it at module level, "
                f"baseline {baseline[lib]}; reach it through a backend "
                f"fixture, or skip the module first with "
                f'pytest.importorskip("{lib}"):{listed}'
            )
    problems += [
        f"{BASELINE_KEY}{lib} is stale: {lib} is not checked"
        for lib in baseline
        if lib not in libs
    ]
    return problems


def ratchet_notices(
    libs: Sequence[str],
    modules: Mapping[str, Sequence[str]],
    baseline: Mapping[str, int],
) -> list[str]:
    """Counts below their baselines, each with the baseline to set."""
    return [
        f"{lib}: down to {len(modules.get(lib, []))}; "
        f"set {BASELINE_KEY}{lib}={len(modules.get(lib, []))}"
        for lib in libs
        if lib in baseline and len(modules.get(lib, [])) < baseline[lib]
    ]


def measure(root: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    """The unguarded test modules per library (paths relative to ``root``)
    and the baselines, for the checkout at ``root``."""
    library_imports = _library_imports(root)
    bound = {lib: bound_modules(lib, library_imports) for lib in OPTIONAL}
    found = unguarded_test_modules(root, bound)
    relative = {
        lib: [p.relative_to(root).as_posix() for p in paths]
        for lib, paths in found.items()
    }
    return relative, read_baseline(root / BASELINE_FILE)


def main(argv: Sequence[str]) -> int:
    root = Path(__file__).resolve().parents[4]
    modules, baseline = measure(root)
    problems = ratchet_problems(OPTIONAL, modules, baseline)
    for problem in problems:
        print(f"optional-test-import ratchet: {problem}")
    # GitHub Actions shows a ::notice:: line as an annotation.
    for notice in ratchet_notices(OPTIONAL, modules, baseline):
        print(f"::notice::optional-test-import ratchet: {notice}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
