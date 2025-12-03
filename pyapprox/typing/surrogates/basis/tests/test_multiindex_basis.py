import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from pyapprox.typing.surrogates.basis.multiindex_basis_factory import (
    multiindex_basis_factory,
)
from pyapprox.typing.surrogates.basis.monomial1d import MonomialBasis1D


class TestMultiIndexBasis(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_multiindex_basis(self) -> None:
        """
        Test the MultiIndexBasis class.
        """
        bkd = self.bkd()

        # Create univariate bases
        basis1 = MonomialBasis1D(bkd)
        basis2 = MonomialBasis1D(bkd)

        # Create multi-index basis
        indices = bkd.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]).T
        multi_index_basis = multiindex_basis_factory([basis1, basis2], indices)

        # Define samples
        samples = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Compute basis matrix
        basis_matrix = multi_index_basis(samples)

        # Expected basis matrix
        expected_basis_matrix = bkd.asarray(
            [
                [1.0, 1.0, 4.0, 4.0],
                [1.0, 2.0, 5.0, 10.0],
                [1.0, 3.0, 6.0, 18.0],
            ]
        )

        # Assert that the basis matrix matches the expected values
        bkd.assert_allclose(basis_matrix, expected_basis_matrix)


class TestMultiIndexBasisNumpy(
    TestMultiIndexBasis[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestMultiIndexBasisTorch(
    TestMultiIndexBasis[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
