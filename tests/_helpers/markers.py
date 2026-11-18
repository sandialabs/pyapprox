"""Slow-test marker aliases, and how a marker's targets match a test."""

from typing import Optional, Tuple

import pytest

# Convenience aliases — apply the marker with "*" (all backends).
slow_test = pytest.mark.slow_on("*")
slower_test = pytest.mark.slower_on("*")
slowest_test = pytest.mark.slowest_on("*")


def matching_target(
    parametrize_id: Optional[str], targets: Tuple[str, ...]
) -> Optional[str]:
    """The first of ``targets`` naming part of a test's parametrize id, or
    None (also when the test is not parametrized).

    A target matches a run of whole hyphen-separated segments of the
    combined id, in any position, so it names a backend (``NumpyBkd``,
    ``TorchBkd``) or a stack case (``csr``, ``dense-torch``, ``mock``)
    alike: ``dense-torch`` matches ``dense-torch-RecomputeAlways``, and
    ``dense`` matches both dense stacks; ``mock`` does not match
    ``mockery``.
    """
    if parametrize_id is None:
        return None
    padded = f"-{parametrize_id}-"
    for target in targets:
        if f"-{target}-" in padded:
            return target
    return None
