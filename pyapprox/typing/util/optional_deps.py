"""Utilities for handling optional dependencies.

This module provides functions for checking and importing optional dependencies
with helpful error messages when they are not installed.
"""

import importlib
from typing import Any, Optional


def package_available(name: str) -> bool:
    """Check if a package is available for import.

    Parameters
    ----------
    name : str
        Name of the package to check.

    Returns
    -------
    bool
        True if the package can be imported, False otherwise.
    """
    try:
        importlib.import_module(name)
        return True
    except (ModuleNotFoundError, ImportError):
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
