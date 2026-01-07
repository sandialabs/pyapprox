"""Pytest configuration for local OED tests."""


def pytest_collection_modifyitems(items):
    """Skip base test classes that have __test__ = False."""
    for item in items:
        if hasattr(item.cls, "__test__") and not item.cls.__test__:
            item.add_marker("skip")
