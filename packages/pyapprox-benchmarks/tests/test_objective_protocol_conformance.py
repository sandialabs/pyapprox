"""Every benchmark model class must expose a ``derivatives()`` bundle.

Generic wrappers (timed, make_parallel, ...) require
ObjectiveProtocol — evaluation plus ``derivatives()``. Per the
derivatives-bundle convention, absence of capability is expressed by
``Derivatives.none()``, never by a missing method. A benchmark model
without ``derivatives()`` cannot be wrapped and breaks tutorials at
render time; this sweep catches that statically, without executing any
model or tutorial.
"""

import ast
import pathlib

import pyapprox_benchmarks


def _iter_model_classes():
    """Yield (path, ClassDef, methods) for every model-like class.

    Model-like = defines both ``__call__`` and ``nqoi`` methods.
    """
    root = pathlib.Path(pyapprox_benchmarks.__file__).parent
    for py in sorted(root.rglob("*.py")):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "__call__" in methods and "nqoi" in methods:
                yield py, node, methods


def _is_protocol(node: ast.ClassDef) -> bool:
    return any("Protocol" in ast.unparse(base) for base in node.bases)


class TestObjectiveProtocolConformance:
    def test_all_model_classes_define_derivatives(self):
        missing = [
            f"{py.name}::{node.name}"
            for py, node, methods in _iter_model_classes()
            if not _is_protocol(node) and "derivatives" not in methods
        ]
        assert not missing, (
            "Benchmark model classes missing derivatives(); add\n"
            "    def derivatives(self) -> Derivatives[Array]:\n"
            "        return Derivatives.none()\n"
            "(or a populated bundle) to: " + ", ".join(missing)
        )

    def test_sweep_finds_models(self):
        """Guard the sweep itself: it must find a healthy population."""
        n = sum(1 for _ in _iter_model_classes())
        assert n >= 10, f"conformance sweep only found {n} model classes"
