"""Protocols for Karhunen-Loève Expansion implementations."""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class KLEProtocol(Protocol, Generic[Array]):
    """Protocol for Karhunen-Loève Expansion implementations.

    A KLE represents a random field as a truncated eigenfunction expansion:
        f(x) = mean(x) + sum_{i=1}^{nterms} sqrt(lambda_i) * phi_i(x) * z_i

    where lambda_i are eigenvalues, phi_i are eigenfunctions, and z_i are
    random coefficients.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nterms(self) -> int:
        """Return the number of KLE terms (truncation level)."""
        ...

    def __call__(self, coef: Array) -> Array:
        """Evaluate the KLE at given coefficients.

        Parameters
        ----------
        coef : Array, shape (nterms, nsamples)
            Random coefficients for each sample.

        Returns
        -------
        Array, shape (ncoords, nsamples)
            Field values at mesh coordinates for each sample.
        """
        ...

    def eigenvectors(self) -> Array:
        """Return the unweighted eigenvectors.

        Returns
        -------
        Array, shape (ncoords, nterms)
            Eigenvectors without eigenvalue scaling.
        """
        ...

    def weighted_eigenvectors(self) -> Array:
        """Return eigenvectors scaled by sqrt(eigenvalues) and sigma.

        Returns
        -------
        Array, shape (ncoords, nterms)
            Eigenvectors multiplied by sqrt(eigenvalues) * sigma.
        """
        ...

    def eigenvalues(self) -> Array:
        """Return the eigenvalues.

        Returns
        -------
        Array, shape (nterms,)
            Eigenvalues in descending order.
        """
        ...

    def mean_field(self) -> Array:
        """Return the mean field.

        Part of the contract because the expansion is defined about a
        mean: ``f(x) = mean(x) + sum_i ...``. A KLE without one is a
        KLE with a zero mean, not a KLE of a different kind, so every
        implementation can answer this.

        Implementations that evaluate at arbitrary points hold the mean
        as a scalar or callable rather than an array. They report it
        here at the same points as ``eigenvectors()``, so the two agree
        on where the field is being described.

        Returns
        -------
        Array, shape (ncoords,)
            Mean field at the points the basis is given at.
        """
        ...


@runtime_checkable
class ReducibleKLEProtocol(Protocol, Generic[Array]):
    """Protocol for KLEs that support state reduction and expansion."""

    def reduce_state(self, state: Array) -> Array:
        """Project a full-order state onto the reduced basis.

        Parameters
        ----------
        state : Array
            Full-order state to be reduced.

        Returns
        -------
        Array
            Reduced-order state.
        """
        ...

    def expand_reduced_state(self, reduced_state: Array) -> Array:
        """Expand a reduced-order state back to full-order.

        Parameters
        ----------
        reduced_state : Array
            Reduced-order state.

        Returns
        -------
        Array
            Full-order state.
        """
        ...
