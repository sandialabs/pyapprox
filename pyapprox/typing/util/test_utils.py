"""Shared test utilities for the typing module."""

import unittest


def load_tests(loader, tests, pattern):
    """
    Exclude base classes from unittest discovery.

    Base classes should set __test__ = False to be excluded.
    This function works with both pytest (which respects __test__ natively)
    and unittest (via this load_tests protocol).

    Note: We check __dict__ directly to avoid inheritance issues. If we used
    getattr(), derived classes would inherit __test__ = False from their base.

    Usage:
        # In test file, import and re-export:
        from pyapprox.typing.util.test_utils import load_tests

        class MyBaseTest(unittest.TestCase):
            __test__ = False
            ...

        class MyConcreteTest(MyBaseTest):
            pass  # No __test__ in __dict__, so defaults to True
    """
    suite = unittest.TestSuite()
    for group in tests:
        for test in group:
            # Check __dict__ directly to avoid inheritance issues
            if test.__class__.__dict__.get('__test__', True):
                suite.addTest(test)
    return suite
