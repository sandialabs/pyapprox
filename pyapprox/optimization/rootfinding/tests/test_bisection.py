from pyapprox.optimization.rootfinding.bisection import (
    BisectionResidualProtocol,
    BisectionSearch,
)
from pyapprox.util.backends.protocols import Backend


class TestBisectionSearch:

    def test_bisection_search(self, bkd) -> None:
        """
        Test the bisection search algorithm.
        """
        # Define the residual class
        class Residual(BisectionResidualProtocol):
            def __init__(self, backend: Backend) -> None:
                self._bkd = backend

            def bkd(self) -> Backend:
                return self._bkd

            def __call__(self, iterate):
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
