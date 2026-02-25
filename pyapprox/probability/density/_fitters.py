"""Density coefficient fitters: strategies for computing coefficients from M and b."""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class DensityFitterProtocol(Protocol, Generic[Array]):
    """Protocol for density coefficient fitting strategies.

    Given a mass matrix M and load vector b (computed from a basis and
    quadrature data), produce coefficients d.
    """

    def fit(self, mass_matrix: Array, load_vector: Array) -> Array:
        """Compute density coefficients from mass matrix and load vector.

        Parameters
        ----------
        mass_matrix : Array
            Shape (nbasis, nbasis).
        load_vector : Array
            Shape (nbasis,).

        Returns
        -------
        Array
            Coefficients d, shape (nbasis,).
        """
        ...


class LinearDensityFitter(Generic[Array]):
    """Fit density coefficients by solving M*d = b.

    This is the L2 projection approach: finds d that minimizes
    ||f_approx - f_true||_{L2}^2 in the span of the basis.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def fit(self, mass_matrix: Array, load_vector: Array) -> Array:
        """Solve M*d = b for density coefficients.

        Parameters
        ----------
        mass_matrix : Array
            Shape (nbasis, nbasis).
        load_vector : Array
            Shape (nbasis,).

        Returns
        -------
        Array
            Coefficients d, shape (nbasis,).
        """
        return self._bkd.solve(mass_matrix, load_vector)


class KDEFitter(Generic[Array]):
    """Fit density coefficients by using load vector directly.

    Returns the load vector b as coefficients, bypassing the mass matrix
    solve. This recovers standard kernel density estimation when the basis
    centers are placed at the data points and the weights encode the
    desired KDE coefficients.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def fit(self, mass_matrix: Array, load_vector: Array) -> Array:
        """Return load vector directly as coefficients.

        Parameters
        ----------
        mass_matrix : Array
            Ignored.
        load_vector : Array
            Shape (nbasis,). Used directly as coefficients.

        Returns
        -------
        Array
            Coefficients d = load_vector, shape (nbasis,).
        """
        return load_vector


__all__ = ["DensityFitterProtocol", "LinearDensityFitter", "KDEFitter"]
