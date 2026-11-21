"""Importing a pyapprox package must not load torch, jax or numba.

All three are optional and slow to import (seconds each). A package
``__init__`` that loads one makes every user of that package pay for it,
and once torch is optional, breaks the package outright for a user without
it. Torch and JAX are backends, loaded only when a ``TorchBkd`` or
``JaxBkd`` is requested; numba is an accelerator, loaded only when a kernel
first runs.

The check reads the source rather than importing it, so it costs about a
second and reports the import chain that reaches the offending module. It
follows every statement that runs at import time: module-level imports and
calls, including those in ``try``, ``if`` and class bodies, but not function
bodies or ``if TYPE_CHECKING:`` blocks. A module-level
``package_available(name)`` call counts as importing ``name``, since it
imports the package when installed. Importing ``a.b.c`` also runs
``a/__init__`` and ``a/b/__init__``.

``KNOWN_HEAVY`` lists the modules reachable from a package ``__init__``
that load a heavy dependency today. It must equal what the scan finds:
fixing one means deleting its entry, and a new one fails with its chain.
"""

import ast
from collections import deque
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pytest

import pyapprox

HEAVY = ("torch", "jax", "numba")

SRC = Path(pyapprox.__file__).resolve().parent.parent

# The library modules whose import-time code loads a heavy dependency and
# that some package __init__ reaches. Only allowed to shrink.
KNOWN_HEAVY: Dict[str, Set[str]] = {"torch": set(), "jax": set(), "numba": set()}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _import_time_nodes(body: List[ast.stmt]) -> Iterator[ast.AST]:
    """Yield the nodes of ``body`` that execute when the module is imported."""
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Decorators and defaults run at import; the body does not.
            yield from stmt.decorator_list
            yield from stmt.args.defaults
            continue
        if isinstance(stmt, ast.If) and _is_type_checking(stmt.test):
            yield from _import_time_nodes(stmt.orelse)
            continue
        if isinstance(stmt, ast.ClassDef):
            yield from stmt.decorator_list
            yield from stmt.bases
            yield from _import_time_nodes(stmt.body)
            continue
        nested = [
            getattr(stmt, field)
            for field in ("body", "orelse", "finalbody")
            if isinstance(getattr(stmt, field, None), list)
        ]
        handlers = getattr(stmt, "handlers", [])
        if nested or handlers:
            for block in nested:
                yield from _import_time_nodes(block)
            for handler in handlers:
                yield from _import_time_nodes(handler.body)
            if isinstance(stmt, ast.If):
                yield stmt.test
            continue
        yield stmt


def _resolve(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    base = module.split(".")
    if not is_package:
        base.pop()
    if node.level > 1:
        base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _imports(path: Path, modules: Set[str]) -> Tuple[Set[str], Set[str]]:
    """Return (pyapprox modules, heavy packages) imported when ``path`` runs."""
    module = _module_name(path)
    is_package = path.name == "__init__.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    targets: Set[str] = set()
    heavy: Set[str] = set()
    for top in _import_time_nodes(tree.body):
        for node in ast.walk(top):
            names: List[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = _resolve(module, is_package, node)
                names = [base] + [f"{base}.{alias.name}" for alias in node.names]
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "package_available"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names = [node.args[0].value]
            for name in names:
                root = name.split(".")[0]
                if root in HEAVY:
                    heavy.add(root)
                elif root == "pyapprox":
                    parts = name.split(".")
                    # The module and the package __init__s above it.
                    for end in range(1, len(parts) + 1):
                        prefix = ".".join(parts[:end])
                        if prefix in modules:
                            targets.add(prefix)
    targets.discard(module)
    return targets, heavy


def _scan() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], List[str]]:
    paths = {
        _module_name(p): p
        for p in (SRC / "pyapprox").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    modules = set(paths)
    graph: Dict[str, Set[str]] = {}
    heavy: Dict[str, Set[str]] = {}
    for name, path in paths.items():
        graph[name], heavy[name] = _imports(path, modules)
    packages = sorted(n for n, p in paths.items() if p.name == "__init__.py")
    return graph, heavy, packages


def _chain(graph: Dict[str, Set[str]], start: str, goal: str) -> Optional[List[str]]:
    previous: Dict[str, str] = {}
    queue = deque([start])
    seen = {start}
    while queue:
        node = queue.popleft()
        if node == goal:
            chain = [node]
            while chain[-1] != start:
                chain.append(previous[chain[-1]])
            return chain[::-1]
        for nxt in sorted(graph.get(node, ())):
            if nxt not in seen:
                seen.add(nxt)
                previous[nxt] = node
                queue.append(nxt)
    return None


def heavy_modules_reached() -> Dict[str, Dict[str, List[str]]]:
    """For each heavy package: offending module -> one chain from an __init__."""
    graph, heavy, packages = _scan()
    found: Dict[str, Dict[str, List[str]]] = {dep: {} for dep in HEAVY}
    offenders = {dep: {m for m, h in heavy.items() if dep in h} for dep in HEAVY}
    for package in packages:
        reached: Set[str] = set()
        queue = deque([package])
        while queue:
            node = queue.popleft()
            if node in reached:
                continue
            reached.add(node)
            queue.extend(graph.get(node, ()))
        for dep in HEAVY:
            for module in sorted(reached & offenders[dep]):
                if module not in found[dep]:
                    chain = _chain(graph, package, module)
                    found[dep][module] = chain if chain is not None else [module]
    return found


@pytest.fixture(scope="module")
def reached() -> Dict[str, Dict[str, List[str]]]:
    return heavy_modules_reached()


@pytest.mark.parametrize("dep", HEAVY)
def test_no_new_heavy_import_on_package_import(
    reached: Dict[str, Dict[str, List[str]]], dep: str
) -> None:
    new = sorted(set(reached[dep]) - KNOWN_HEAVY[dep])
    chains = "\n".join(" -> ".join(reached[dep][m]) for m in new)
    assert not new, (
        f"importing a package now loads {dep}; defer the import into the "
        f"function that needs it, or stop re-exporting the module:\n{chains}"
    )


@pytest.mark.parametrize("dep", HEAVY)
def test_known_heavy_list_is_current(
    reached: Dict[str, Dict[str, List[str]]], dep: str
) -> None:
    fixed = sorted(KNOWN_HEAVY[dep] - set(reached[dep]))
    assert not fixed, f"no longer reached; delete from KNOWN_HEAVY[{dep!r}]: {fixed}"
