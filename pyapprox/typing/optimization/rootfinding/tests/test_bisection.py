import unittest
from typing import Generic

import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.optimization.rootfinding.bisection import (
    BisectionSearch,
    BisectionResidualProtocol,
)


class TestBisectionSearch(Generic[Array], unittest.TestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_bisection_search(self) -> None:
        """
        Test the bisection search algorithm.
        """
        bkd = self.bkd()

        # Define the residual class
        class Residual(BisectionResidualProtocol[Array]):
            def __init__(self, backend: Backend[Array]) -> None:
                self._bkd = backend

            def bkd(self) -> Backend[Array]:
                return self._bkd

            def __call__(self, iterate: Array) -> Array:
                self._rhs = self._bkd.array([0.1, 0.3, 0.6])
                return iterate**2 - self._rhs

        # Initialize the bisection search and residual
        residual = Residual(backend=bkd)
        bisearch = BisectionSearch(residual)

        # Define bounds for the search
        bounds = bkd.array([[0.0, 0.5], [0.1, 1.0], [0.5, 1.0]])

        # Solve using bisection search
        roots = bisearch.solve(bounds, atol=1e-10)

        # Assert that the computed roots match the expected values
        expected_roots = bkd.sqrt(residual._rhs)
        bkd.assert_allclose(roots, expected_roots)


# Derived test class for NumPy backend
class TestBisectionSearchNumpy(TestBisectionSearch[Array]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestBisectionSearchTorch(TestBisectionSearch[torch.Tensor]):
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
        TestBisectionSearchNumpy,
        TestBisectionSearchTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
