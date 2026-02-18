"""Tests for parallel execution backends."""

import unittest

from pyapprox.typing.util.test_utils import load_tests

# Check for optional dependencies
try:
    import mpire
    HAS_MPIRE = True
except ImportError:
    HAS_MPIRE = False


def _square(x: int) -> int:
    """Simple function for testing."""
    return x * x


def _add(x: int, y: int) -> int:
    """Simple function for starmap testing."""
    return x + y


class TestJoblibBackend(unittest.TestCase):
    """Tests for JoblibBackend."""

    def test_map_basic(self):
        """Test basic map functionality."""
        from pyapprox.typing.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=2)
        items = [1, 2, 3, 4]
        result = backend.map(_square, items)
        self.assertEqual(list(result), [1, 4, 9, 16])

    def test_map_empty(self):
        """Test map with empty input."""
        from pyapprox.typing.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=2)
        result = backend.map(_square, [])
        self.assertEqual(list(result), [])

    def test_starmap_basic(self):
        """Test basic starmap functionality."""
        from pyapprox.typing.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=2)
        items = [(1, 2), (3, 4), (5, 6)]
        result = backend.starmap(_add, items)
        self.assertEqual(list(result), [3, 7, 11])

    def test_backend_name(self):
        """Test backend name string."""
        from pyapprox.typing.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=4, prefer="threads")
        name = backend.backend_name()
        self.assertIn("joblib", name)
        self.assertIn("4", name)
        self.assertIn("threads", name)

    def test_override_n_jobs(self):
        """Test overriding n_jobs in method call."""
        from pyapprox.typing.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=4)
        items = [1, 2, 3, 4]
        # Override with n_jobs=2
        result = backend.map(_square, items, n_jobs=2)
        self.assertEqual(list(result), [1, 4, 9, 16])


@unittest.skipUnless(HAS_MPIRE, "mpire not installed")
class TestMpireBackend(unittest.TestCase):
    """Tests for MpireBackend."""

    def test_map_basic(self):
        """Test basic map functionality."""
        from pyapprox.typing.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=2)
        items = [1, 2, 3, 4]
        result = backend.map(_square, items)
        self.assertEqual(list(result), [1, 4, 9, 16])

    def test_map_empty(self):
        """Test map with empty input."""
        from pyapprox.typing.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=2)
        result = backend.map(_square, [])
        self.assertEqual(list(result), [])

    def test_starmap_basic(self):
        """Test basic starmap functionality."""
        from pyapprox.typing.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=2)
        items = [(1, 2), (3, 4), (5, 6)]
        result = backend.starmap(_add, items)
        self.assertEqual(list(result), [3, 7, 11])

    def test_backend_name(self):
        """Test backend name string."""
        from pyapprox.typing.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=4, progress_bar=True)
        name = backend.backend_name()
        self.assertIn("mpire", name)
        self.assertIn("4", name)

    def test_override_n_jobs(self):
        """Test overriding n_jobs in method call."""
        from pyapprox.typing.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=4)
        items = [1, 2, 3, 4]
        # Override with n_jobs=2
        result = backend.map(_square, items, n_jobs=2)
        self.assertEqual(list(result), [1, 4, 9, 16])


class TestFuturesBackend(unittest.TestCase):
    """Tests for FuturesBackend."""

    def test_map_basic(self):
        """Test basic map functionality."""
        from pyapprox.typing.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=2)
        items = [1, 2, 3, 4]
        result = backend.map(_square, items)
        self.assertEqual(list(result), [1, 4, 9, 16])

    def test_map_empty(self):
        """Test map with empty input."""
        from pyapprox.typing.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=2)
        result = backend.map(_square, [])
        self.assertEqual(list(result), [])

    def test_starmap_basic(self):
        """Test basic starmap functionality."""
        from pyapprox.typing.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=2)
        items = [(1, 2), (3, 4), (5, 6)]
        result = backend.starmap(_add, items)
        self.assertEqual(list(result), [3, 7, 11])

    def test_backend_name(self):
        """Test backend name string."""
        from pyapprox.typing.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=4)
        name = backend.backend_name()
        self.assertIn("futures", name)
        self.assertIn("4", name)

    def test_override_n_jobs(self):
        """Test overriding n_jobs in method call."""
        from pyapprox.typing.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=4)
        items = [1, 2, 3, 4]
        # Override with n_jobs=2
        result = backend.map(_square, items, n_jobs=2)
        self.assertEqual(list(result), [1, 4, 9, 16])


class TestProtocolCompliance(unittest.TestCase):
    """Test that backends comply with ParallelBackendProtocol."""

    def test_joblib_protocol(self):
        """Test JoblibBackend implements protocol."""
        from pyapprox.typing.interface.parallel.protocols import (
            ParallelBackendProtocol,
        )
        from pyapprox.typing.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=2)
        self.assertTrue(isinstance(backend, ParallelBackendProtocol))

    @unittest.skipUnless(HAS_MPIRE, "mpire not installed")
    def test_mpire_protocol(self):
        """Test MpireBackend implements protocol."""
        from pyapprox.typing.interface.parallel.protocols import (
            ParallelBackendProtocol,
        )
        from pyapprox.typing.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=2)
        self.assertTrue(isinstance(backend, ParallelBackendProtocol))


    def test_futures_protocol(self):
        """Test FuturesBackend implements protocol."""
        from pyapprox.typing.interface.parallel.protocols import (
            ParallelBackendProtocol,
        )
        from pyapprox.typing.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=2)
        self.assertTrue(isinstance(backend, ParallelBackendProtocol))


if __name__ == "__main__":
    unittest.main()
