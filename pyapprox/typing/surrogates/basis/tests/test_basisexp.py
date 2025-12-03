import unittest
from typing import Generic, Any


from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from pyapprox.typing.surrogates.basis.basisexp_factory import (
    basis_expansion_factory,
)
from pyapprox.typing.surrogates.basis.multiindex_basis_factory import (
    multiindex_basis_factory,
)
from pyapprox.typing.surrogates.basis.monomial1d import MonomialBasis1D
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


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
        multi_index_basis = multiindex_basis_factory([basis1, basis2], indices)

        # Create basis expansion
        expansion = basis_expansion_factory(nqoi=1, basis=multi_index_basis)

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

        # Derivative checks
        derivative_checker = DerivativeChecker(expansion)

        # Check derivatives at a single sample point
        sample = bkd.asarray([[1.0], [4.0]])
        errors = derivative_checker.check_derivatives(sample)

        # Assert that the derivative errors are below a tolerance
        self.assertTrue(
            derivative_checker.error_ratios_satisfied(errors[0], 1e-7)
        )  # Jacobian errors
        self.assertTrue(
            derivative_checker.error_ratios_satisfied(errors[0], 1e-7)
        )  # Hessian errors


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
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
