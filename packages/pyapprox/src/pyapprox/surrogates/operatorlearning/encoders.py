"""Field encoders for operator learning.

An encoder maps field values on a grid to coefficients in a finite
basis. For an *output* encoder the map must be an isometry, since the
least-squares fit minimizes the Euclidean norm of coefficient residuals
and that equals the Bochner error only when norms are preserved.

TODO(layering): the Y-inner-product is taken as a MassMatrixProtocol
imported from pyapprox.ode. A mass matrix is pure linear algebra, not
an ODE concept, and util.linalg already owns the sparse dispatch that
ode.mass_matrix itself calls, so util.linalg is the natural home. The
move touches ode/, pde/ and surrogates/dynamical_systems/, so it is
deferred to its own change. The import direction is at least sound:
surrogates already depends on ode elsewhere and ode never imports
surrogates.
"""

from __future__ import annotations

from typing import Generic, List, Optional, Sequence

from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.surrogates.operatorlearning.protocols import (
    FieldEncoderProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class IdentityFieldEncoder(Generic[Array]):
    """Encoder whose coefficients are the grid values themselves.

    An isometry only when the grid inner product is the Euclidean one,
    which for a quadrature-weighted space means uniform unit weights.
    Useful for tests and for fields already expressed in coefficients.

    Parameters
    ----------
    ngrid : int
        Number of grid points, equal to the number of coefficients.
    bkd : Backend[Array]
        Computational backend.
    is_isometry : bool
        Whether the identity map preserves the norm of interest.
        Defaults to True, the Euclidean case.
    """

    def __init__(
        self, ngrid: int, bkd: Backend[Array], is_isometry: bool = True
    ) -> None:
        self._ngrid = ngrid
        self._bkd = bkd
        self._is_isometry = is_isometry

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ncodes(self) -> int:
        """Return the number of coefficients."""
        return self._ngrid

    def ngrid(self) -> int:
        """Return the number of grid points."""
        return self._ngrid

    def encode(self, f_grid: Array) -> Array:
        """Encode grid values to coefficients. (ngrid, N) -> (ncodes, N)."""
        return f_grid

    def decode(self, codes: Array) -> Array:
        """Decode coefficients to grid values. (ncodes, N) -> (ngrid, N)."""
        return codes

    def is_isometry(self) -> bool:
        """Return whether encoding preserves the Y-norm."""
        return self._is_isometry


class GramProjectionEncoder(Generic[Array]):
    r"""Encoder projecting onto a basis under an inner product operator.

    Coefficients are the projections

    .. math:: c_j = \langle \psi_j, f \rangle_Y = \psi_j^T M f

    and decoding evaluates :math:`\sum_j c_j \psi_j`. When the basis is
    orthonormal under that inner product, :math:`\Psi^T M \Psi = I`,
    the map is an isometry and the Euclidean coefficient norm equals
    the Y-norm of the field.

    The inner product is supplied as a mass matrix rather than a vector
    of quadrature weights, so a non-diagonal :math:`M` — the usual case
    for a finite element space — is handled exactly instead of being
    silently lumped to its diagonal. Lumping would corrupt both the
    projection and the orthonormality check, leaving ``is_isometry()``
    true for a basis that is not one under the real inner product. Pass
    :class:`DiagonalMassMatrix` for a quadrature rule,
    :class:`ConstantSparseMassMatrix` for an assembled FE mass matrix,
    or :class:`IdentityMassMatrix` for the Euclidean case.

    Orthonormality is checked at construction rather than assumed, so a
    basis that is merely linearly independent is reported as
    non-isometric instead of silently producing a wrong error norm. Use
    :func:`orthonormalize_basis` to fix such a basis.

    Parameters
    ----------
    basis_values : Array
        Basis functions at the grid points, :math:`\psi_j(x_k)`.
        Shape: (ngrid, ncodes)
    mass_matrix : MassMatrixProtocol[Array]
        The Y-inner-product operator :math:`M`, acting on (ngrid, N).
    bkd : Backend[Array]
        Computational backend.
    orthonormality_tol : float
        Tolerance on :math:`\|\Psi^T M \Psi - I\|_\infty` below which
        the encoder reports itself an isometry.
    """

    def __init__(
        self,
        basis_values: Array,
        mass_matrix: MassMatrixProtocol[Array],
        bkd: Backend[Array],
        orthonormality_tol: float = 1e-10,
    ) -> None:
        if basis_values.ndim != 2:
            raise ValueError(
                f"basis_values must be 2D, got shape {basis_values.shape}"
            )
        if not isinstance(mass_matrix, MassMatrixProtocol):
            raise TypeError(
                f"mass_matrix must satisfy MassMatrixProtocol, got "
                f"{type(mass_matrix).__name__}"
            )
        self._basis_values = basis_values
        self._mass = mass_matrix
        self._bkd = bkd
        self._gram = self._compute_gram()
        self._is_isometry = bool(
            bkd.max(bkd.abs(self._gram - bkd.eye(self.ncodes())))
            < orthonormality_tol
        )

    def _compute_gram(self) -> Array:
        """Return the Gram matrix Psi^T M Psi. Shape: (ncodes, ncodes)."""
        return self._bkd.dot(
            self._basis_values.T, self._mass.apply(self._basis_values)
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ncodes(self) -> int:
        """Return the number of coefficients."""
        return int(self._basis_values.shape[1])

    def ngrid(self) -> int:
        """Return the number of grid points."""
        return int(self._basis_values.shape[0])

    def basis_values(self) -> Array:
        """Return the basis at the grid points. (ngrid, ncodes)."""
        return self._basis_values

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return the Y-inner-product operator."""
        return self._mass

    def gram(self) -> Array:
        """Return the Gram matrix Psi^T M Psi. (ncodes, ncodes)."""
        return self._gram

    def encode(self, f_grid: Array) -> Array:
        """Encode grid values to coefficients. (ngrid, N) -> (ncodes, N)."""
        if f_grid.shape[0] != self.ngrid():
            raise ValueError(
                f"f_grid has wrong leading dimension {f_grid.shape[0]}, "
                f"expected {self.ngrid()}"
            )
        return self._bkd.dot(
            self._basis_values.T, self._mass.apply(f_grid)
        )

    def decode(self, codes: Array) -> Array:
        """Decode coefficients to grid values. (ncodes, N) -> (ngrid, N)."""
        if codes.shape[0] != self.ncodes():
            raise ValueError(
                f"codes has wrong leading dimension {codes.shape[0]}, "
                f"expected {self.ncodes()}"
            )
        return self._bkd.dot(self._basis_values, codes)

    def is_isometry(self) -> bool:
        """Return whether encoding preserves the Y-norm."""
        return self._is_isometry


def orthonormalize_basis(
    basis_values: Array,
    mass_matrix: MassMatrixProtocol[Array],
    bkd: Backend[Array],
) -> Array:
    r"""Orthonormalize a basis under an inner product operator.

    Returns basis values spanning the same space but satisfying
    :math:`\Psi^T M \Psi = I`, so a :class:`GramProjectionEncoder`
    built on them is an isometry.

    Uses the Cholesky factor of the Gram matrix,
    :math:`\Psi \leftarrow \Psi L^{-T}`, which is stable for a
    well-conditioned basis and fails loudly for a rank-deficient one
    rather than returning a basis that is only approximately
    orthonormal.

    The factorized Gram is only (ncodes, ncodes), so this stays cheap
    however fine the grid; ``mass_matrix`` may be sparse. The basis
    itself is a dense (ngrid, ncodes) array, which bounds ngrid by
    what fits in memory.

    Parameters
    ----------
    basis_values : Array
        Basis at the grid points. Shape: (ngrid, ncodes)
    mass_matrix : MassMatrixProtocol[Array]
        The Y-inner-product operator :math:`M`.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Orthonormalized basis values. Shape: (ngrid, ncodes)
    """
    gram = bkd.dot(basis_values.T, mass_matrix.apply(basis_values))
    factor = bkd.cholesky(gram)
    return bkd.solve_triangular(factor, basis_values.T, lower=True).T


class ProductFieldEncoder(Generic[Array]):
    r"""Encoder over several fields, concatenating their coefficients.

    Encodes each field with its own sub-encoder and stacks the results,
    so a multi-field problem presents the same interface as a
    single-field one and consumers never handle offsets.

    The product of isometries is an isometry: if each part satisfies
    :math:`\|f_j\|_{Y_j} = \|c_j\|_2`, then

    .. math:: \|c\|_2^2 = \sum_j \|c_j\|_2^2 = \sum_j \|f_j\|_{Y_j}^2

    which is the product-space norm. Per-field ``scalings`` rescale that
    norm — a modeling choice, for making fields of different physical
    units commensurate — and are applied so that decoding inverts
    encoding exactly.

    Parameters
    ----------
    encoders : Sequence[FieldEncoderProtocol[Array]]
        One encoder per field, at least one.
    bkd : Backend[Array]
        Computational backend.
    scalings : Array, optional
        Per-field weights :math:`\alpha_j` applied as
        :math:`\sqrt{\alpha_j}` to each block of coefficients, making
        the induced norm :math:`\sum_j \alpha_j \|f_j\|_{Y_j}^2`.
        Shape: (nfields,). Defaults to unit weights, the plain product
        norm.
    """

    def __init__(
        self,
        encoders: Sequence[FieldEncoderProtocol[Array]],
        bkd: Backend[Array],
        scalings: Optional[Array] = None,
    ) -> None:
        if len(encoders) == 0:
            raise ValueError("encoders must not be empty")
        for encoder in encoders:
            if not isinstance(encoder, FieldEncoderProtocol):
                raise TypeError(
                    f"encoders must satisfy FieldEncoderProtocol, got "
                    f"{type(encoder).__name__}"
                )
        if scalings is not None:
            if scalings.shape != (len(encoders),):
                raise ValueError(
                    f"scalings has wrong shape {scalings.shape}, "
                    f"expected ({len(encoders)},)"
                )
            if bool(bkd.min(scalings) <= 0):
                raise ValueError("scalings must be strictly positive")
        self._encoders = list(encoders)
        self._bkd = bkd
        self._scalings = scalings
        self._sqrt_scalings = (
            None if scalings is None else bkd.sqrt(scalings)
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nfields(self) -> int:
        """Return the number of composed fields."""
        return len(self._encoders)

    def encoders(self) -> List[FieldEncoderProtocol[Array]]:
        """Return the per-field encoders."""
        return list(self._encoders)

    def ncodes(self) -> int:
        """Return the total number of coefficients across fields."""
        return sum(encoder.ncodes() for encoder in self._encoders)

    def ngrid(self) -> int:
        """Return the total number of grid points across fields."""
        return sum(encoder.ngrid() for encoder in self._encoders)

    def _scale(self, index: int) -> Optional[Array]:
        """Return the amplitude applied to field ``index``, if any."""
        if self._sqrt_scalings is None:
            return None
        return self._sqrt_scalings[index]

    def encode(self, f_grid: Array) -> Array:
        """Encode stacked grid values. (ngrid, N) -> (ncodes, N).

        Fields are stacked in encoder order along the first axis.
        """
        if f_grid.shape[0] != self.ngrid():
            raise ValueError(
                f"f_grid has wrong leading dimension {f_grid.shape[0]}, "
                f"expected {self.ngrid()}"
            )
        blocks = []
        offset = 0
        for index, encoder in enumerate(self._encoders):
            size = encoder.ngrid()
            codes = encoder.encode(f_grid[offset : offset + size])
            scale = self._scale(index)
            blocks.append(codes if scale is None else scale * codes)
            offset += size
        return self._bkd.concatenate(blocks, axis=0)

    def decode(self, codes: Array) -> Array:
        """Decode stacked coefficients. (ncodes, N) -> (ngrid, N)."""
        if codes.shape[0] != self.ncodes():
            raise ValueError(
                f"codes has wrong leading dimension {codes.shape[0]}, "
                f"expected {self.ncodes()}"
            )
        blocks = []
        offset = 0
        for index, encoder in enumerate(self._encoders):
            size = encoder.ncodes()
            block = codes[offset : offset + size]
            scale = self._scale(index)
            blocks.append(
                encoder.decode(block if scale is None else block / scale)
            )
            offset += size
        return self._bkd.concatenate(blocks, axis=0)

    def is_isometry(self) -> bool:
        """Return whether encoding preserves the product Y-norm.

        True when every sub-encoder is an isometry. Scalings change
        which norm is preserved, not whether one is, so they do not
        affect this.
        """
        return all(encoder.is_isometry() for encoder in self._encoders)
