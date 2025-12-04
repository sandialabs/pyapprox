import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.basis.monomial1d import MonomialBasis1D


class TestMonomialBasis1D(Generic[Array], unittest.TestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_monomial_basis(self) -> None:
        """
        Test the MonomialBasis1D class.
        """
        bkd = self.bkd()

        # Initialize MonomialBasis1D
        basis = MonomialBasis1D(bkd)

        # Set number of terms
        basis.set_nterms(3)

        # Define samples
        samples = bkd.asarray([[1.0, 2.0, 3.0]])

        # Compute basis matrix
        basis_matrix = basis(samples)

        # Expected basis matrix
        expected_basis_matrix = bkd.asarray(
            [[1.0, 1.0, 1.0], [1.0, 2.0, 4.0], [1.0, 3.0, 9.0]]
        )

        # Assert that the basis matrix matches the expected values
        bkd.assert_allclose(basis_matrix, expected_basis_matrix)


# Derived test class for NumPy backend
class TestMonomialBasis1DNumpy(TestMonomialBasis1D[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestMonomialBasis1DTorch(TestMonomialBasis1D[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


# Custom test loader to exclude the base class
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base class
    ContinuousScipyRandomVariable1D.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestMonomialBasis1DNumpy,
        TestMonomialBasis1DTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
