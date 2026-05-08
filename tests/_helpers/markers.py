"""Slow-test marker aliases."""
import pytest

# Convenience aliases — apply the marker with "*" (all backends).
slow_test = pytest.mark.slow_on("*")
slower_test = pytest.mark.slower_on("*")
slowest_test = pytest.mark.slowest_on("*")
