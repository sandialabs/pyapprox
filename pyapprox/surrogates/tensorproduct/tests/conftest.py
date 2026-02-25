"""Configure pytest to skip base test classes."""


def pytest_pycollect_makeitem(collector, name, obj):
    """Skip classes with __test__ = False."""
    if hasattr(obj, "__test__") and not obj.__test__:
        return None
    return None
