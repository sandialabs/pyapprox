"""Test helpers for ``pyapprox.pde.coupling`` on the opaque mock stack."""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.linalg.operators import SpaceDescriptor
from tests._helpers.linalg.mock_stack import MockVector, peek, wrap


class MockPortRows:
    """Port rows of the mock stack; only stack code looks inside a handle.
    Port blocks are NumPy arrays."""

    def __init__(self, descriptor: SpaceDescriptor, dofs: Sequence[int]) -> None:
        self._descriptor = descriptor
        self._dofs = np.asarray(dofs, dtype=np.intp)

    def nrows(self) -> int:
        return int(self._dofs.shape[0])

    def take(self, x: MockVector) -> NDArray[np.float64]:
        return peek(x)[self._dofs, :]

    def add_to(self, y: MockVector, block: NDArray[np.float64]) -> MockVector:
        data = peek(y)
        np.add.at(data, self._dofs, block)
        return wrap(data, self._descriptor)


class MockPortAdapter:
    """The mock stack's adapter between exchange blocks (NumPy arrays)
    and its own port blocks."""

    def to_port(
        self, block: NDArray[np.float64], descriptor: SpaceDescriptor
    ) -> MockVector:
        return wrap(block, descriptor)

    def to_exchange(self, block: MockVector) -> NDArray[np.float64]:
        return peek(block)
