"""MeshKLEFieldMap: field map wrapping MeshKLE weighted eigenvectors."""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class MeshKLEFieldMap(Generic[Array]):
    """Field map wrapping MeshKLE weighted eigenvectors + array mean.

    Maps parameter vector to spatial field:
        field(x) = mean_field(x) + W @ params
    where W = MeshKLE.weighted_eigenvectors() (pre-scaled by sqrt(lambda)*sigma).

    Satisfies FieldMapProtocol with constant Jacobian = W.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mean_field : Array
        Mean field at mesh nodes. Shape: (npts,).
    weighted_eigenvectors : Array
        Pre-scaled eigenvectors from MeshKLE.weighted_eigenvectors().
        Shape: (npts, nterms).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        mean_field: Array,
        weighted_eigenvectors: Array,
    ) -> None:
        self._bkd = bkd
        self._mean_field = mean_field
        self._W = weighted_eigenvectors

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._W.shape[1]

    def __call__(self, params_1d: Array) -> Array:
        """Evaluate field: mean_field + W @ params. Preserves autograd."""
        return self._mean_field + self._bkd.dot(self._W, params_1d)

    def jacobian(self, params_1d: Array) -> Array:
        """Return Jacobian d(field)/d(params). Shape: (npts, nvars).

        Constant for linear map -- returns cached W matrix.
        """
        return self._W
