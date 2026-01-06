"""pytest configuration for expdesign tests."""

# This ensures that base test classes with __test__ = False are not collected
collect_ignore_glob = []


def pytest_collection_modifyitems(items):
    """Skip base test classes that have __test__ = False."""
    pass
