"""Utilities for handling optional dependencies.

This module provides functions for checking and importing optional dependencies
with helpful error messages when they are not installed.
"""

import importlib
import importlib.util
import warnings
from typing import Any, Optional


_package_cache: dict[str, bool] = {}


def package_available(name: str) -> bool:
    """Check if a package is available for import.

    Results are cached so that the import check (and any warning) is
    performed only once per package per process, regardless of how many
    call sites invoke this function.

    If the package is installed but fails to import (e.g. due to an
    incompatible dependency version), a single warning is issued so the
    user knows that acceleration is disabled and why.

    Parameters
    ----------
    name : str
        Name of the package to check.

    Returns
    -------
    bool
        True if the package can be imported, False otherwise.
    """
    if name in _package_cache:
        return _package_cache[name]

    try:
        importlib.import_module(name)
        _package_cache[name] = True
        return True
    except ModuleNotFoundError:
        _package_cache[name] = False
        return False
    except ImportError as err:
        # Package is installed but can't be imported — warn once
        if importlib.util.find_spec(name) is not None:
            warnings.warn(
                f"'{name}' is installed but failed to import: {err}. "
                f"Falling back to vectorized implementations. "
                f"To fix, install a compatible version: "
                f"pip install pyapprox[{name}]",
                stacklevel=2,
            )
        _package_cache[name] = False
        return False


def import_optional_dependency(
    name: str,
    feature_name: Optional[str] = None,
    extra_name: Optional[str] = None,
) -> Any:
    """Import an optional dependency, raising a helpful error if not installed.

    Parameters
    ----------
    name : str
        Name of the package to import.

    feature_name : str, optional
        Name of the feature that requires this dependency, for error messages.
        If None, uses the package name.

    extra_name : str, optional
        Name of the pip extra to install for this dependency.
        If None, uses the package name.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        If the package is not installed, with installation instructions.

    Examples
    --------
    >>> cvxpy = import_optional_dependency(
    ...     "cvxpy",
    ...     feature_name="MLBLUESPDOptimizer",
    ...     extra_name="cvxpy"
    ... )
    """
    if extra_name is None:
        extra_name = name
    if feature_name is None:
        feature_name = name

    try:
        return importlib.import_module(name)
    except (ModuleNotFoundError, ImportError) as err:
        raise ImportError(
            f"{feature_name} requires the optional dependency '{name}'. "
            f"Install it with: pip install pyapprox[{extra_name}] "
            f"or pip install {name}"
        ) from err
