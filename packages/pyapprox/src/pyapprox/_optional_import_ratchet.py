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

An import inside a test body fails that test instead, on every backend
parameter that does not skip: a test taking the parametrized ``bkd``
that imports torch errors on its NumPy parameter without torch. So does a
fixture of the same name as a root one (``torch_bkd``, ...) that shadows
it without skipping.

:data:`OPTIONAL` names the libraries. :func:`bound_modules` finds the
library modules under ``packages/*/src`` that import one at load time,
transitively. :func:`scan_tests` counts, per library,

- the test modules whose module-level imports reach it before a
  module-level skip. Imports inside functions, ``if`` blocks
  (``TYPE_CHECKING``) and ``try`` blocks that recover do not count; a
  ``try`` whose handlers re-raise does;
- in the other test modules that do not skip at module level, the tests
  (``test*`` functions and methods) and fixtures whose bodies, nested
  functions included, import it before a ``pytest.importorskip("<lib>")``
  or a ``package_available("<lib>")`` skip. A function that takes a
  fixture of :data:`SKIPPING_FIXTURES`, or whose ``skipif`` decorator (or
  its class's) names the library, does not count. Helper functions are
  not scanned; their callers are.

Each count is recorded in ``ratchet_baselines.txt``, as
``optional-test-import:<lib>=<modules>`` and
``optional-test-import-in-function:<lib>=<functions>``; under that
file's rule a commit that lowers a count lowers its baseline in the same
commit, and no commit raises one. The two are kept apart because a
module-level import fails collection of the whole module, an import in a
function fails one test, and one total would let removing the first pay
for adding several of the second. It also lists the fixtures of
:data:`SKIPPING_FIXTURES` defined anywhere but the root ``conftest.py``,
which has no baseline: the exemption above trusts those names to skip.

``python -m pyapprox._optional_import_ratchet`` exits nonzero when a
count exceeds its baseline (listing the modules or functions), a library
has none, a baseline is stale, or a skipping fixture is redefined. A
count below its baseline is printed as a notice to lower it and does not
fail, since merging branches lowers counts no single branch could record.
"""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from pyapprox._moved_ratchet import BASELINE_FILE, _files

OPTIONAL = ("torch", "jax", "numba")

# The root conftest.py fixtures that skip when a library is missing.
# ``autodiff_bkd`` skips without torch while TorchBkd is its only
# backend; once it runs on another one, a torch import in a test taking
# it fails there, and it leaves this table.
SKIPPING_FIXTURES: Mapping[str, frozenset[str]] = {
    "torch": frozenset(
        {"torch_bkd", "autodiff_bkd", "torch_float32_bkd", "torch_mps_bkd"}
    ),
}


@dataclass(frozen=True)
class Count:
    """One counted quantity: its baseline-key prefix and how a rise is
    reported."""

    key: str
    found: str
    fix: str


MODULES = Count(
    "optional-test-import:",
    "test modules import it at module level",
    "reach it through a backend fixture, or skip the module first with "
    'pytest.importorskip("{lib}")',
)
FUNCTIONS = Count(
    "optional-test-import-in-function:",
    "tests or fixtures import it without skipping first",
    "take a backend fixture that skips without it (torch_bkd, "
    'autodiff_bkd), or call pytest.importorskip("{lib}") before the import',
)


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


def _module_guard(node: ast.stmt, source: str, lib: str) -> bool:
    """Whether module-level ``node`` skips the module without ``lib``; a
    skip inside a function or class body skips only when it runs."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    return _guards(ast.get_source_segment(source, node) or "", lib)


def _unguarded(tree: ast.Module, source: str, lib: str, bound: frozenset[str]) -> bool:
    for node in tree.body:
        if _module_guard(node, source, lib):
            return False
        if any(_reaches(n, lib, bound) for n in _module_imports([node])):
            return True
    return False


_Function = ast.FunctionDef | ast.AsyncFunctionDef


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _guard_lines(function: _Function, lib: str) -> list[int]:
    """Lines in ``function`` that skip it without ``lib``: an
    ``importorskip("<lib>")`` call, or an ``if`` testing
    ``package_available("<lib>")`` whose body calls ``skip``."""
    lines = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and _call_name(node) == "importorskip"
            and _constant_arg(node) == lib
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.If) and any(
            isinstance(test, ast.Call)
            and _call_name(test) == "package_available"
            and _constant_arg(test) == lib
            for test in ast.walk(node.test)
        ):
            if any(
                isinstance(call, ast.Call) and _call_name(call) == "skip"
                for stmt in node.body
                for call in ast.walk(stmt)
            ):
                lines.append(node.lineno)
    return lines


def _constant_arg(node: ast.Call) -> object:
    first = node.args[0] if node.args else None
    return first.value if isinstance(first, ast.Constant) else None


def _imports_among(nodes: Iterable[ast.AST]) -> Iterator[tuple[int, list[str]]]:
    """(line, dotted names) of each import statement in ``nodes`` and their
    descendants, except in the body of a ``try`` that recovers."""
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node.lineno, _module_imports([node])
        elif isinstance(node, ast.Try) and not _reraises(node):
            yield from _imports_among([*node.handlers, *node.orelse, *node.finalbody])
        else:
            yield from _imports_among(ast.iter_child_nodes(node))


def _skips_without(decorators: Sequence[ast.expr], lib: str) -> bool:
    """Whether a ``skipif`` decorator names ``lib`` (``not _HAS_NUMBA``,
    ``not package_available("numba")``)."""
    return any(
        "skipif" in (text := ast.unparse(decorator)) and lib in text.lower()
        for decorator in decorators
    )


def _is_fixture(function: _Function) -> bool:
    return any("fixture" in ast.unparse(d) for d in function.decorator_list)


def _unguarded_function(function: _Function, lib: str, bound: frozenset[str]) -> bool:
    arguments = function.args
    taken = {
        a.arg for a in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    }
    if taken & SKIPPING_FIXTURES.get(lib, frozenset()) or _skips_without(
        function.decorator_list, lib
    ):
        return False
    first_guard = min(_guard_lines(function, lib), default=None)
    return any(
        (first_guard is None or line < first_guard)
        and any(_reaches(n, lib, bound) for n in names)
        for line, names in _imports_among(function.body)
    )


def _unguarded_functions(
    body: Sequence[ast.stmt], lib: str, bound: frozenset[str], prefix: str = ""
) -> Iterator[str]:
    """Qualified names of the tests and fixtures in ``body`` (and in its
    classes, unless a class ``skipif`` names ``lib``) that import ``lib``
    without skipping first."""
    for node in body:
        if isinstance(node, ast.ClassDef):
            if not _skips_without(node.decorator_list, lib):
                yield from _unguarded_functions(
                    node.body, lib, bound, f"{prefix}{node.name}."
                )
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (node.name.startswith("test") or _is_fixture(node))
            and _unguarded_function(node, lib, bound)
        ):
            yield f"{prefix}{node.name}"


def _shadowing_fixtures(tree: ast.Module) -> Iterator[str]:
    names = frozenset().union(*SKIPPING_FIXTURES.values())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
            and _is_fixture(node)
        ):
            yield node.name


@dataclass(frozen=True)
class Findings:
    """What :func:`scan_tests` counts, as paths relative to the scanned
    root: test modules per library, tests and fixtures per library
    (``path::Class.test``), and redefined skipping fixtures
    (``path::name``)."""

    modules: Mapping[str, Sequence[str]]
    functions: Mapping[str, Sequence[str]]
    shadowing: Sequence[str]


def scan_tests(root: Path, bound: Mapping[str, frozenset[str]]) -> Findings:
    """Per library in ``bound`` (its bound modules), the test modules and
    the tests and fixtures under ``root`` that import it without skipping
    first, and the skipping fixtures defined outside ``root/conftest.py``.
    Test modules are the files of a ``tests`` directory; a module counted
    at module level is not scanned for functions, since its collection
    already fails."""
    modules: dict[str, list[str]] = {lib: [] for lib in bound}
    functions: dict[str, list[str]] = {lib: [] for lib in bound}
    shadowing: list[str] = []
    for path in _files(root):
        relative = path.relative_to(root)
        in_tests = "tests" in relative.parts[:-1]
        if path.suffix != ".py" or not (in_tests or path.name == "conftest.py"):
            continue
        source = path.read_text(errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        name = relative.as_posix()
        if name != "conftest.py":
            shadowing += [f"{name}::{f}" for f in _shadowing_fixtures(tree)]
        if not in_tests:
            continue
        for lib, lib_bound in bound.items():
            if any(_module_guard(node, source, lib) for node in tree.body):
                continue
            if _unguarded(tree, source, lib, lib_bound):
                modules[lib].append(name)
            else:
                functions[lib] += [
                    f"{name}::{f}"
                    for f in _unguarded_functions(tree.body, lib, lib_bound)
                ]
    return Findings(modules, functions, shadowing)


def read_baseline(path: Path, count: Count = MODULES) -> dict[str, int]:
    """The ``<count.key><lib>=<value>`` entries of a baseline file."""
    baseline: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if line.startswith(count.key) and "=" in line:
            key, value = line[len(count.key) :].split("=", 1)
            baseline[key.strip()] = int(value)
    return baseline


def ratchet_problems(
    count: Count,
    libs: Sequence[str],
    found: Mapping[str, Sequence[str]],
    baseline: Mapping[str, int],
) -> list[str]:
    """What fails the ratchet for ``count``, given what was found per
    library: a count above its baseline, a library without one, and
    baselines of libraries no longer in ``libs``. A count below its
    baseline is a notice (:func:`ratchet_notices`)."""
    problems = []
    for lib in libs:
        items = found.get(lib, [])
        number, key = len(items), f"{count.key}{lib}"
        if lib not in baseline:
            problems.append(f"no baseline for {lib}: add {key}={number}")
        elif number > baseline[lib]:
            listed = "".join(f"\n  {item}" for item in items)
            problems.append(
                f"{lib}: {number} {count.found}, baseline {baseline[lib]}; "
                f"{count.fix.format(lib=lib)}:{listed}"
            )
    problems += [
        f"{count.key}{lib} is stale: {lib} is not checked"
        for lib in baseline
        if lib not in libs
    ]
    return problems


def ratchet_notices(
    count: Count,
    libs: Sequence[str],
    found: Mapping[str, Sequence[str]],
    baseline: Mapping[str, int],
) -> list[str]:
    """Counts below their baselines, each with the baseline to set."""
    return [
        f"{lib}: down to {len(found.get(lib, []))}; "
        f"set {count.key}{lib}={len(found.get(lib, []))}"
        for lib in libs
        if lib in baseline and len(found.get(lib, [])) < baseline[lib]
    ]


def shadowing_problems(shadowing: Sequence[str]) -> list[str]:
    """Each skipping fixture defined outside the root ``conftest.py``."""
    return [
        f"{item} redefines a root conftest.py fixture that skips without "
        f"its library; use the root one"
        for item in shadowing
    ]


def measure(root: Path) -> Findings:
    """:func:`scan_tests` for the checkout at ``root``."""
    library_imports = _library_imports(root)
    bound = {lib: bound_modules(lib, library_imports) for lib in OPTIONAL}
    return scan_tests(root, bound)


def check(root: Path) -> tuple[list[str], list[str]]:
    """The problems and the notices for the checkout at ``root``, against
    its baseline file."""
    findings = measure(root)
    problems = shadowing_problems(findings.shadowing)
    notices = []
    for count, found in ((MODULES, findings.modules), (FUNCTIONS, findings.functions)):
        baseline = read_baseline(root / BASELINE_FILE, count)
        problems += ratchet_problems(count, OPTIONAL, found, baseline)
        notices += ratchet_notices(count, OPTIONAL, found, baseline)
    return problems, notices


def main(argv: Sequence[str]) -> int:
    root = Path(__file__).resolve().parents[4]
    problems, notices = check(root)
    for problem in problems:
        print(f"optional-test-import ratchet: {problem}")
    # GitHub Actions shows a ::notice:: line as an annotation.
    for notice in notices:
        print(f"::notice::optional-test-import ratchet: {notice}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
