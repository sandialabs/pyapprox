"""Optional-dependency errors: the message names the feature and the pip
extra, and every library module that imports torch at load time raises it
when torch is missing."""

import importlib
import sys
from pathlib import Path
from typing import List

import pytest

import pyapprox
from pyapprox._optional_import_ratchet import _library_imports, bound_modules
from pyapprox.util.optional_deps import (
    import_optional_dependency,
    missing_dependency_message,
)

_ROOT = Path(pyapprox.__file__).resolve().parents[4]


def _torch_importers() -> List[str]:
    """Modules of the pyapprox package whose own module-level imports
    include torch. (Tutorial figure modules raise their own message, about
    rendering the tutorial, not a pyapprox extra.)"""
    if not (_ROOT / "packages").is_dir():
        return []
    imports = _library_imports(_ROOT)
    return sorted(
        name
        for name, names in imports.items()
        if name.startswith("pyapprox.")
        and any(n == "torch" or n.startswith("torch.") for n in names)
    )


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


@pytest.mark.parametrize("module", _torch_importers())
def test_torch_importer_raises_install_message_without_torch(
    module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With torch hidden, importing the module raises the standard message
    (from it, or from a torch-bound module it imports first)."""
    bound = bound_modules("torch", _library_imports(_ROOT))
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch.") or name in bound:
            monkeypatch.delitem(sys.modules, name)
    # A None entry makes ``import torch`` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"pip install pyapprox\[torch\]"):
        importlib.import_module(module)


def test_torch_importers_are_found() -> None:
    """The scan that parametrizes the guard test finds the known importers,
    so an empty parametrization cannot pass silently."""
    if not (_ROOT / "packages").is_dir():
        pytest.skip("not a source checkout")
    importers = _torch_importers()
    assert "pyapprox.util.backends.torch" in importers
    assert "pyapprox.surrogates.tensorproduct.compute_torch" in importers
