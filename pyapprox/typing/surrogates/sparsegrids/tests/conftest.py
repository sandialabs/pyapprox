"""Pytest configuration for sparse grids tests.

This file configures pytest to skip base test classes that have __test__ = False.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip test classes with __test__ = False."""
    skip_marker = pytest.mark.skip(reason="Base test class")
    for item in items:
        if hasattr(item, "cls") and item.cls is not None:
            # Check __dict__ directly to avoid inheritance issues
            if item.cls.__dict__.get("__test__", True) is False:
                item.add_marker(skip_marker)
