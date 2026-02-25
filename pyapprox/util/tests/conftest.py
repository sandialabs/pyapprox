"""Configure pytest to skip base test classes."""

# This allows us to use Generic base classes that define tests
# without pytest trying to instantiate them directly
collect_ignore_glob = []


def pytest_pycollect_makeitem(collector, name, obj):
    """Skip classes with __test__ = False."""
    if hasattr(obj, "__test__") and not obj.__test__:
        return None
    return None
