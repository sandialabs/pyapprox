from typing import Protocol, Generic, Tuple
from pyapprox.typing.util.backends.protocols import Array, Backend


class PiecewisePolynomialProtocol(Protocol, Generic[Array]):
    """
    Protocol for piecewise quadratic basis functions and quadrature rules.
    """

    def __call__(self, xx: Array) -> Array:
        """
        Evaluate the piecewise quadratic basis functions at given points.

        Parameters
        ----------
        xx : Array
            Points where the basis functions are evaluated.

        Returns
        -------
        Array
            Values of the basis functions at the given points.
        """
        ...

    def quadrature_rule(self) -> Tuple[Array, Array]:
        """
        Compute quadrature points and weights based on the nodes.

        Returns
        -------
        Tuple[Array, Array]
            Quadrature points and weights.
        """
        ...

    def nodes(self) -> Array:
        """
        Return the nodes where the basis functions are defined.

        Returns
        -------
        Array
            The nodes (abscissas) where the basis functions are defined.
        """
        ...

    def bkd(self) -> Backend:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend
            Backend used for computations (e.g., NumPy or PyTorch).
        """
        ...
