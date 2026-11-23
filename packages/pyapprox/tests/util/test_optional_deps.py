"""Optional-dependency errors: the message names the feature and the pip
extra, and every library module that imports torch or numba at load time
raises it when the library is missing."""

import importlib
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

import pyapprox
from pyapprox._optional_import_ratchet import _library_imports, bound_modules
from pyapprox.util.optional_deps import (
    import_optional_dependency,
    missing_dependency_message,
)

_ROOT = Path(pyapprox.__file__).resolve().parents[4]
# Optional libraries whose pip extra has the library's name.
_GUARDED = ("torch", "numba", "jax")


def _importers(lib: str) -> List[str]:
    """Modules of the pyapprox package whose own module-level imports
    include ``lib``. (Tutorial figure modules raise their own message, about
    rendering the tutorial, not a pyapprox extra.)"""
    if not (_ROOT / "packages").is_dir():
        return []
    imports = _library_imports(_ROOT)
    return sorted(
        name
        for name, names in imports.items()
        if name.startswith("pyapprox.")
        and any(n == lib or n.startswith(f"{lib}.") for n in names)
    )


def _guard_cases() -> List[Tuple[str, str]]:
    return [(lib, module) for lib in _GUARDED for module in _importers(lib)]


def test_message_names_feature_and_extra() -> None:
    assert missing_dependency_message("torch", "TorchBkd", "torch") == (
        "TorchBkd requires the optional dependency 'torch'. Install it with: "
        "pip install pyapprox[torch] or pip install torch"
    )
    # Feature and extra default to the package name.
    assert missing_dependency_message("cvxpy") == (
        "cvxpy requires the optional dependency 'cvxpy'. Install it with: "
        "pip install pyapprox[cvxpy] or pip install cvxpy"
    )


def test_import_optional_dependency_raises_the_message() -> None:
    name = "pyapprox_no_such_package"
    with pytest.raises(ImportError) as info:
        import_optional_dependency(name, "SomeFeature", "extra")
    assert str(info.value) == missing_dependency_message(name, "SomeFeature", "extra")
    assert isinstance(info.value.__cause__, ImportError)
    assert import_optional_dependency("math").__name__ == "math"


@pytest.mark.parametrize("lib, module", _guard_cases())
def test_importer_raises_install_message_without_library(
    lib: str, module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``lib`` hidden, importing the module raises the standard message
    (from it, or from a bound module it imports first)."""
    bound = bound_modules(lib, _library_imports(_ROOT))
    for name in list(sys.modules):
        if name == lib or name.startswith(f"{lib}.") or name in bound:
            monkeypatch.delitem(sys.modules, name)
    # A None entry makes ``import <lib>`` raise ImportError.
    monkeypatch.setitem(sys.modules, lib, None)
    with pytest.raises(ImportError, match=rf"pip install pyapprox\[{lib}\]"):
        importlib.import_module(module)


@pytest.mark.parametrize(
    "module, selector",
    [
        ("pyapprox.surrogates.affine.basis.dispatch", "get_basis_eval_impl"),
        ("pyapprox.surrogates.affine.basis.dispatch", "get_basis_jacobian_impl"),
        ("pyapprox.surrogates.affine.basis.dispatch", "get_basis_hessian_impl"),
        ("pyapprox.polynomials.univariate.lagrange_dispatch", "get_lagrange_eval_impl"),
        ("pyapprox.surrogates.tensorproduct.dispatch", "get_tp_eval_impl"),
        ("pyapprox.expdesign.likelihood.dispatch", "get_logpdf_matrix_impl"),
        ("pyapprox.expdesign.likelihood.dispatch", "get_evidence_jacobian_impl"),
    ],
)
def test_numpy_dispatch_needs_no_torch(
    module: str, selector: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Choosing a kernel for a NumPy backend neither needs torch nor
    imports the torch backend (which would load torch when installed)."""
    from pyapprox.util.backends.numpy import NumpyBkd

    choose = getattr(importlib.import_module(module), selector)
    bound = bound_modules("torch", _library_imports(_ROOT))
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch.") or name in bound:
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert callable(choose(NumpyBkd()))
    assert "pyapprox.util.backends.torch" not in sys.modules


def test_importers_are_found() -> None:
    """The scan that parametrizes the guard test finds the known importers,
    so an empty parametrization cannot pass silently."""
    if not (_ROOT / "packages").is_dir():
        pytest.skip("not a source checkout")
    torch_importers = _importers("torch")
    assert "pyapprox.util.backends.torch" in torch_importers
    assert "pyapprox.surrogates.tensorproduct.compute_torch" in torch_importers
    assert _importers("numba") == ["pyapprox.util.numba_compat"]
    assert _importers("jax") == ["pyapprox.util.backends.jax"]
