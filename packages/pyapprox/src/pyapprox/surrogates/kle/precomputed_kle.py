"""A Karhunen-Loeve expansion reduced to what evaluation needs.

Computing a KLE is expensive; evaluating one is a matmul. The two
halves of :class:`MeshKLE` reflect that: the kernel, coordinates,
quadrature weights and eigensolver exist only to *build* the basis,
while the eigenvalues, eigenvectors, mean and scaling are all that
*evaluating* it touches.

:class:`PrecomputedKLE` is the second half alone. It is what a stored
basis becomes when reloaded, and it is deliberately unable to
reconstruct the first half: there is no kernel to re-solve with.

``KLEProtocol`` already describes exactly this smaller object -- it
requires ``bkd``, ``nterms``, ``__call__``, ``eigenvectors``,
``weighted_eigenvectors`` and ``eigenvalues``, and mentions neither a
kernel nor coordinates. So this class is not a reduced KLE so much as
the protocol's own shape, made concrete.
"""

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend


class PrecomputedKLE(Generic[Array]):
    r"""A KLE built from an already-computed basis.

    Evaluates

    .. math::
        f(x) = \bar{f}(x) + \sigma \sum_{i=1}^{k}
               \sqrt{\lambda_i}\, \phi_i(x)\, z_i

    which is the same expression :class:`MeshKLE` and
    :class:`NystromKLE` evaluate; only the provenance of
    :math:`\phi` differs.

    Where the rows of ``eigenvectors`` came from is not recorded,
    because it changes no computation performed here. They may be a
    :class:`MeshKLE` collocation basis, or a :class:`NystromKLE` basis
    extended to arbitrary points via ``eigenvectors_at``. Both are
    ``(npoints, nterms)`` arrays carrying the same meaning.

    That is also the limitation: this class cannot extend to points
    beyond the ones it was given, since extension needs the kernel.
    Callers who want a basis somewhere new must evaluate it there
    *before* constructing this object.

    Parameters
    ----------
    eigenvalues : Array, shape (nterms,)
        Eigenvalues in descending order, non-negative.
    eigenvectors : Array, shape (npoints, nterms)
        The unweighted basis -- without ``sqrt(eigenvalue)`` or
        ``sigma`` folded in, matching ``eigenvectors()`` elsewhere in
        this module rather than ``weighted_eigenvectors()``.
    mean_field : Array, shape (npoints,)
        Mean field at the same points as ``eigenvectors``. An array
        rather than a scalar or callable: a callable would have to be
        stored as a pickled closure, which does not survive being
        written to a file and read back by other software.
    sigma : float
        Standard deviation scaling applied to the basis.
    use_log : bool
        When True, return ``exp(mean + basis @ coef)``, giving a
        lognormal field.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        eigenvalues: Array,
        eigenvectors: Array,
        mean_field: Array,
        sigma: float = 1.0,
        use_log: bool = False,
        bkd: Optional[Backend[Array]] = None,
    ) -> None:
        if bkd is None:
            raise ValueError("bkd must be provided")
        self._bkd = bkd
        self._validate(eigenvalues, eigenvectors, mean_field)
        self._eig_vals = eigenvalues
        self._unweighted_eig_vecs = eigenvectors
        self._mean_field = mean_field
        self._sigma = sigma
        self._use_log = use_log
        self._nterms = int(eigenvalues.shape[0])
        self._sqrt_eig_vals = bkd.sqrt(eigenvalues)
        self._eig_vecs = eigenvectors * self._sqrt_eig_vals * sigma

    def _validate(
        self, eigenvalues: Array, eigenvectors: Array, mean_field: Array
    ) -> None:
        """Check the three arrays describe one consistent basis.

        Each mismatch here would otherwise surface as a broadcasting
        error inside a matmul, at evaluation time, naming shapes that
        have already been combined -- far from the array that was
        actually wrong.
        """
        if eigenvalues.ndim != 1:
            raise ValueError(
                f"eigenvalues must be 1D, got ndim={eigenvalues.ndim}"
            )
        if eigenvectors.ndim != 2:
            raise ValueError(
                f"eigenvectors must be 2D with shape (npoints, nterms), "
                f"got ndim={eigenvectors.ndim}"
            )
        if mean_field.ndim != 1:
            raise ValueError(
                f"mean_field must be 1D, got ndim={mean_field.ndim}"
            )
        if eigenvectors.shape[1] != eigenvalues.shape[0]:
            raise ValueError(
                f"eigenvectors has {eigenvectors.shape[1]} columns but "
                f"eigenvalues has {eigenvalues.shape[0]} entries; each "
                "column is scaled by one eigenvalue"
            )
        if mean_field.shape[0] != eigenvectors.shape[0]:
            raise ValueError(
                f"mean_field has {mean_field.shape[0]} entries but "
                f"eigenvectors has {eigenvectors.shape[0]} rows; both "
                "must be given at the same points"
            )
        smallest = float(self._bkd.to_numpy(eigenvalues).min())
        if smallest < 0.0:
            raise ValueError(
                f"eigenvalues must be non-negative, got {smallest:.3e}. A "
                "covariance operator is positive semi-definite, and the "
                "expansion takes sqrt of each eigenvalue"
            )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nterms(self) -> int:
        """Return the number of KLE terms."""
        return self._nterms

    def nvars(self) -> int:
        """Return the number of KLE terms (alias for nterms)."""
        return self._nterms

    def npoints(self) -> int:
        """Return the number of points the basis is given at."""
        return int(self._unweighted_eig_vecs.shape[0])

    def eigenvalues(self) -> Array:
        """Return the eigenvalues, shape ``(nterms,)``."""
        return self._eig_vals

    def singular_values(self) -> Array:
        r"""Return ``sqrt(eigenvalues)``, shape ``(nterms,)``.

        The same spectrum on the scale a POD energy fraction is taken
        on. Offered beside :meth:`eigenvalues` so which convention a
        caller gets does not depend on which class they reached for --
        the eigenvalues are variances and weigh a KLE truncation, the
        singular values are amplitudes.

        Note this class stores no sample count, so unlike
        ``DataDrivenKLE`` there is no ``1/(n-1)`` between the two: it is
        handed a basis and a spectrum, not the data they came from.
        """
        return self._bkd.sqrt(self._eig_vals)

    def eigenvectors(self) -> Array:
        """Unweighted eigenvectors, shape ``(npoints, nterms)``."""
        return self._unweighted_eig_vecs

    def weighted_eigenvectors(self) -> Array:
        """Eigenvectors scaled by ``sqrt(eigenvalue)`` and sigma."""
        return self._eig_vecs

    def mean_field(self) -> Array:
        """Return the mean field, shape ``(npoints,)``."""
        return self._mean_field

    def sigma(self) -> float:
        """Return the standard deviation scaling."""
        return self._sigma

    def use_log(self) -> bool:
        """Whether realizations are exponentiated."""
        return self._use_log

    def __call__(self, coef: Array) -> Array:
        """Evaluate the KLE at given coefficients.

        Parameters
        ----------
        coef : Array, shape (nterms, nsamples)
            Random coefficients for each sample.

        Returns
        -------
        Array, shape (npoints, nsamples)
            Field values at the basis points for each sample.
        """
        if coef.ndim != 2:
            raise ValueError(f"coef.ndim={coef.ndim} but should be 2")
        if coef.shape[0] != self._nterms:
            raise ValueError(
                f"coef.shape[0]={coef.shape[0]} != nterms={self._nterms}"
            )
        field = self._mean_field[:, None] + self._eig_vecs @ coef
        if self._use_log:
            return self._bkd.exp(field)
        return field

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nterms={self._nterms}, "
            f"npoints={self.npoints()}, sigma={self._sigma})"
        )
