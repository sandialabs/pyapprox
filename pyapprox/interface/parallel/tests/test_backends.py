"""Tests for parallel execution backends."""

import pytest

# Check for optional dependencies
try:
    import mpire  # noqa: F401

    HAS_MPIRE = True
except ImportError:
    HAS_MPIRE = False


def _square(x: int) -> int:
    """Simple function for testing."""
    return x * x


def _add(x: int, y: int) -> int:
    """Simple function for starmap testing."""
    return x + y


class TestJoblibBackend:
    """Tests for JoblibBackend."""

    def test_map_basic(self):
        """Test basic map functionality."""
        from pyapprox.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=2)
        items = [1, 2, 3, 4]
        result = backend.map(_square, items)
        assert list(result) == [1, 4, 9, 16]

    def test_map_empty(self):
        """Test map with empty input."""
        from pyapprox.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=2)
        result = backend.map(_square, [])
        assert list(result) == []

    def test_starmap_basic(self):
        """Test basic starmap functionality."""
        from pyapprox.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=2)
        items = [(1, 2), (3, 4), (5, 6)]
        result = backend.starmap(_add, items)
        assert list(result) == [3, 7, 11]

    def test_backend_name(self):
        """Test backend name string."""
        from pyapprox.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=4, prefer="threads")
        name = backend.backend_name()
        assert "joblib" in name
        assert "4" in name
        assert "threads" in name

    def test_override_n_jobs(self):
        """Test overriding n_jobs in method call."""
        from pyapprox.interface.parallel.joblib_backend import (
            JoblibBackend,
        )

        backend = JoblibBackend(n_jobs=4)
        items = [1, 2, 3, 4]
        # Override with n_jobs=2
        result = backend.map(_square, items, n_jobs=2)
        assert list(result) == [1, 4, 9, 16]


@pytest.mark.skipif(not HAS_MPIRE, reason="mpire not installed")
class TestMpireBackend:
    """Tests for MpireBackend."""

    def test_map_basic(self):
        """Test basic map functionality."""
        from pyapprox.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=2)
        items = [1, 2, 3, 4]
        result = backend.map(_square, items)
        assert list(result) == [1, 4, 9, 16]

    def test_map_empty(self):
        """Test map with empty input."""
        from pyapprox.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=2)
        result = backend.map(_square, [])
        assert list(result) == []

    def test_starmap_basic(self):
        """Test basic starmap functionality."""
        from pyapprox.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=2)
        items = [(1, 2), (3, 4), (5, 6)]
        result = backend.starmap(_add, items)
        assert list(result) == [3, 7, 11]

    def test_backend_name(self):
        """Test backend name string."""
        from pyapprox.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=4, progress_bar=True)
        name = backend.backend_name()
        assert "mpire" in name
        assert "4" in name

    def test_override_n_jobs(self):
        """Test overriding n_jobs in method call."""
        from pyapprox.interface.parallel.mpire_backend import (
            MpireBackend,
        )

        backend = MpireBackend(n_jobs=4)
        items = [1, 2, 3, 4]
        # Override with n_jobs=2
        result = backend.map(_square, items, n_jobs=2)
        assert list(result) == [1, 4, 9, 16]


class TestFuturesBackend:
    """Tests for FuturesBackend."""

    def test_map_basic(self):
        """Test basic map functionality."""
        from pyapprox.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=2)
        items = [1, 2, 3, 4]
        result = backend.map(_square, items)
        assert list(result) == [1, 4, 9, 16]

    def test_map_empty(self):
        """Test map with empty input."""
        from pyapprox.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=2)
        result = backend.map(_square, [])
        assert list(result) == []

    def test_starmap_basic(self):
        """Test basic starmap functionality."""
        from pyapprox.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=2)
        items = [(1, 2), (3, 4), (5, 6)]
        result = backend.starmap(_add, items)
        assert list(result) == [3, 7, 11]

    def test_backend_name(self):
        """Test backend name string."""
        from pyapprox.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=4)
        name = backend.backend_name()
        assert "futures" in name
        assert "4" in name

    def test_override_n_jobs(self):
        """Test overriding n_jobs in method call."""
        from pyapprox.interface.parallel.futures_backend import (
            FuturesBackend,
        )

        backend = FuturesBackend(n_jobs=4)
        items = [1, 2, 3, 4]
        # Override with n_jobs=2
        result = backend.map(_square, items, n_jobs=2)
        assert list(result) == [1, 4, 9, 16]


class TestProtocolCompliance:
    """Test that backends comply with ParallelBackendProtocol."""

    def test_joblib_protocol(self):
        """Test JoblibBackend implements protocol."""
        from pyapprox.interface.parallel.joblib_backend import (
            JoblibBackend,
        )
        from pyapprox.interface.parallel.protocols import (
            ParallelBackendProtocol,
        )

        backend = JoblibBackend(n_jobs=2)
        assert isinstance(backend, ParallelBackendProtocol)

    @pytest.mark.skipif(not HAS_MPIRE, reason="mpire not installed")
    def test_mpire_protocol(self):
        """Test MpireBackend implements protocol."""
        from pyapprox.interface.parallel.mpire_backend import (
            MpireBackend,
        )
        from pyapprox.interface.parallel.protocols import (
            ParallelBackendProtocol,
        )

        backend = MpireBackend(n_jobs=2)
        assert isinstance(backend, ParallelBackendProtocol)

    def test_futures_protocol(self):
        """Test FuturesBackend implements protocol."""
        from pyapprox.interface.parallel.futures_backend import (
            FuturesBackend,
        )
        from pyapprox.interface.parallel.protocols import (
            ParallelBackendProtocol,
        )

        backend = FuturesBackend(n_jobs=2)
        assert isinstance(backend, ParallelBackendProtocol)
