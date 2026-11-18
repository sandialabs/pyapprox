"""How a slow marker's targets match a test's parametrize id.

``slow_on("TorchBkd")`` names a backend fixture's id; ``slow_on("mock")``
or ``slow_on("dense-torch")`` names a stack case's. A target matches a
run of whole hyphen-separated segments of the combined id, in any
position, never part of a segment.
"""

from typing import Optional, Tuple

import pytest

from tests._helpers.markers import matching_target


@pytest.mark.parametrize(
    "parametrize_id, targets, expected",
    [
        ("TorchBkd", ("TorchBkd",), "TorchBkd"),
        ("NumpyBkd-3-adjoint", ("TorchBkd",), None),
        ("NumpyBkd-3-adjoint", ("adjoint",), "adjoint"),
        ("dense-torch", ("dense-torch",), "dense-torch"),
        ("dense-torch-RecomputeAlways", ("dense-torch",), "dense-torch"),
        ("RecomputeAlways-dense-torch", ("dense-torch",), "dense-torch"),
        # A whole segment of a case id matches: "dense" names both.
        ("dense-torch-RecomputeAlways", ("torch",), "torch"),
        ("dense-numpy", ("dense",), "dense"),
        ("csr-KeepAll", ("dense-torch", "mock"), None),
        ("mock-KeepAll", ("dense-torch", "mock"), "mock"),
        ("mockery", ("mock",), None),
        (None, ("mock",), None),
    ],
)
def test_matching_target(
    parametrize_id: Optional[str], targets: Tuple[str, ...], expected: Optional[str]
) -> None:
    assert matching_target(parametrize_id, targets) == expected
