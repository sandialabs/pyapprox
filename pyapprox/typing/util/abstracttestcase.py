from abc import ABC
from typing import Any, Optional, Union, Callable, ContextManager


class AbstractTestCase(ABC):
    """
    Abstract class that mimics unittest.TestCase with all assertion methods.
    This class does nothing and is intended for use as a base class to avoid
    running shared logic as a test case when testing mulitple Backends
    """

    def assertEqual(
        self, first: Any, second: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertNotEqual(
        self, first: Any, second: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertTrue(self, expr: Any, msg: Optional[Any] = None) -> None:
        pass

    def assertFalse(self, expr: Any, msg: Optional[Any] = None) -> None:
        pass

    def assertIs(
        self, first: Any, second: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertIsNot(
        self, first: Any, second: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertIsNone(self, expr: Any, msg: Optional[Any] = None) -> None:
        pass

    def assertIsNotNone(self, expr: Any, msg: Optional[Any] = None) -> None:
        pass

    def assertIn(
        self, member: Any, container: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertNotIn(
        self, member: Any, container: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertIsInstance(
        self, obj: Any, cls: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertNotIsInstance(
        self, obj: Any, cls: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertGreater(
        self, first: Any, second: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertGreaterEqual(
        self, first: Any, second: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertLess(
        self, first: Any, second: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertLessEqual(
        self, first: Any, second: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    def assertRaises(
        self,
        expected_exception: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass

    def assertWarns(
        self,
        expected_warning: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass

    def assertLogs(
        self, logger: Optional[Any] = None, level: Optional[Any] = None
    ) -> Any:
        pass

    def assertAlmostEqual(
        self,
        first: Any,
        second: Any,
        places: Optional[Any] = None,
        msg: Optional[Any] = None,
        delta: Optional[Any] = None,
    ) -> None:
        pass

    def assertNotAlmostEqual(
        self,
        first: Any,
        second: Any,
        places: Optional[Any] = None,
        msg: Optional[Any] = None,
        delta: Optional[Any] = None,
    ) -> None:
        pass

    def assertRegex(
        self,
        text: Any,
        expected_regex: Any,
        msg: Optional[Any] = None,
    ) -> None:
        pass

    def assertNotRegex(
        self,
        text: Any,
        unexpected_regex: Any,
        msg: Optional[Any] = None,
    ) -> None:
        pass

    def assertCountEqual(
        self, first: Any, second: Any, msg: Optional[Any] = None
    ) -> None:
        pass

    # def fail(self, msg: Optional[Any] = None) -> None:  # type: ignore
    #     pass
