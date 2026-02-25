"""Pytest configuration for surrogate tests."""

import unittest


def pytest_pycollect_makeitem(collector, name, obj):
    """
    Custom collection hook to handle __test__ = False inheritance.

    Base test classes set __test__ = False to exclude them from test collection.
    However, pytest's default behavior inherits __test__ = False to derived classes.

    This hook checks if __test__ = False is defined directly on the class (not
    inherited),
    allowing derived classes to be collected normally by setting __test__ = True on
    derived classes that don't explicitly define __test__.
    """
    if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
        # If __test__ = False is inherited (not in this class's __dict__),
        # override it to True so the derived class is collected
        if "__test__" not in obj.__dict__ and getattr(obj, "__test__", True) is False:
            obj.__test__ = True
    # Return None to let pytest handle collection normally
    return None
