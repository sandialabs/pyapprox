import unittest
from typing import Generic, Any


from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from pyapprox.typing.surrogates.basis.basisexp import BasisExpansion
from pyapprox.typing.surrogates.basis.multiindex_basis import MultiIndexBasis
from pyapprox.typing.surrogates.basis.monomial import MonomialBasis1D


class TestBasisExpansion(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_basis_expansion(self) -> None:
        """
        Test the BasisExpansion class using MonomialBasis1D.
        """
        bkd = self.bkd()

        # Create univariate bases
        basis1 = MonomialBasis1D(bkd)
        basis2 = MonomialBasis1D(bkd)

        # Create multi-index basis
        indices = bkd.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]).T
        multi_index_basis = MultiIndexBasis([basis1, basis2], indices)

        # Create basis expansion
        expansion = BasisExpansion(nqoi=1, basis=multi_index_basis)

        # Set parameters
        params = bkd.asarray([[1.0], [2.0], [3.0], [4.0]])
        expansion.set_parameters(params)

        # Define samples
        samples = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Compute expansion
        values = expansion(samples)

        # Assert that the expansion values matche the expected values
        expected_basis_matrix = bkd.asarray(
            [
                [1.0, 1.0, 4.0, 4.0],
                [1.0, 2.0, 5.0, 10.0],
                [1.0, 3.0, 6.0, 18.0],
            ]
        )
        expected_values = expected_basis_matrix @ params
        bkd.assert_allclose(values, expected_values)


class TestBasisExpansionNumpy(
    TestBasisExpansion[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestBasisExpansionTorch(
    TestBasisExpansion[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
