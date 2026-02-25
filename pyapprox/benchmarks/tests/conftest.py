"""pytest configuration for benchmark tests."""

from typing import Any, List


# Exclude base test classes from pytest discovery
collect_ignore_glob: List[str] = []


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Filter out test classes with __test__ = False."""
    pass
