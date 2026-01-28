"""pytest configuration for adaptive GP tests."""

collect_ignore_glob = []


def pytest_collection_modifyitems(items):
    """Skip base test classes that have __test__ = False."""
    pass
