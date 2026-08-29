"""PCA-based function encoder."""

from __future__ import annotations

from typing import Generic, Optional

from pyapprox.surrogates.kle.snapshot_eigensolvers import (
    default_snapshot_eigensolver,
)
from pyapprox.surrogates.kle.truncation import resolve_nterms
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import InnerProductProtocol


class PCAFunctionEncoder(Generic[Array]):
    r"""PCA encoder: an orthonormal basis about a mean.

    .. math::
        z = V^T M (f - \bar{f}), \qquad f = V z + \bar{f}

    The metric :math:`M` is what makes ``encode`` a projection rather
    than merely a matrix product. A basis orthonormal under
    :math:`\mathrm{diag}(w)` paired with a Euclidean :math:`V^T` is not
    idempotent and does not return the nearest point in the subspace, so
    the metric travels with the basis rather than being supplied
    separately at each call.

    Parameters
    ----------
    basis : Array, shape (full_dim, latent_dim)
        Orthonormal basis columns, in the metric below.
    mean : Array, shape (full_dim, 1)
        Mean of the training data.
    bkd : Backend[Array]
        Computational backend.
    metric : InnerProductProtocol, optional
        The inner product the basis is orthonormal in. None means
        Euclidean, for which ``encode`` reduces to ``V^T (f - mean)``.
    """

    def __init__(
        self,
        basis: Array,
        mean: Array,
        bkd: Backend[Array],
        metric: Optional[InnerProductProtocol[Array]] = None,
    ) -> None:
        self._basis = basis
        self._mean = mean
        self._bkd = bkd
        self._metric = metric

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def latent_dim(self) -> int:
        return int(self._basis.shape[1])

    def full_dim(self) -> int:
        return int(self._basis.shape[0])

    def basis(self) -> Array:
        """Return the ``(full_dim, latent_dim)`` orthonormal basis."""
        return self._basis

    def mean(self) -> Array:
        """Return the ``(full_dim, 1)`` training-data mean."""
        return self._mean

    def metric(self) -> Optional[InnerProductProtocol[Array]]:
        """Return the inner product the basis is orthonormal in."""
        return self._metric

    def encode(self, samples: Array) -> Array:
        """Full to latent. ``(full_dim, N) -> (latent_dim, N)``."""
        centered = samples - self._mean
        if self._metric is not None:
            centered = self._metric.apply(centered)
        return self._bkd.dot(self._basis.T, centered)

    def decode(self, latents: Array) -> Array:
        """Latent to full. ``(latent_dim, N) -> (full_dim, N)``."""
        return self._bkd.dot(self._basis, latents) + self._mean

    def decode_std(self, std_latents: Array) -> Array:
        r"""Propagate latent std to full space, without the mean shift.

        :math:`\sigma_f = \sqrt{V^2 \sigma_z^2}`, which assumes the
        latent coordinates are uncorrelated. Exact when the latent
        posterior covariance is diagonal; an under-estimate for
        coregionalization kernels, whose cross-coordinate covariance
        this discards.

        The assumption is stated rather than hidden because it is not
        recoverable from the output: the result has the shape and units
        of a standard deviation whether or not the assumption holds.
        Removing it needs a covariance rather than a std from the latent
        regressor, and a matching ``decode_cov`` here.
        """
        var_full = self._bkd.dot(self._basis**2, std_latents**2)
        return self._bkd.sqrt(var_full)

    @classmethod
    def fit_from_data(
        cls,
        f_grid_data: Array,
        bkd: Backend[Array],
        latent_dim: Optional[int] = None,
        variance_fraction: Optional[float] = None,
        center: bool = True,
        metric: Optional[InnerProductProtocol[Array]] = None,
    ) -> PCAFunctionEncoder[Array]:
        """Fit an encoder to training data.

        Parameters
        ----------
        f_grid_data : Array, shape (full_dim, N)
            Training samples, one per column.
        bkd : Backend[Array]
            Computational backend.
        latent_dim : int, optional
            Number of basis vectors to keep.
        variance_fraction : float, optional
            Fraction of variance to retain, which selects ``latent_dim``.
            Exactly one of the two must be given.
        center : bool, optional
            If True (default), subtract the sample mean first. If False
            the stored mean is zero, so encode and decode reduce to
            ``V^T x`` and ``V z`` (uncentered POD).
        metric : InnerProductProtocol, optional
            The inner product to make the basis orthonormal in. None
            means Euclidean, which biases the basis toward wherever a
            non-uniform mesh is refined; pass the quadrature weights or
            mass matrix to avoid that.

        Returns
        -------
        PCAFunctionEncoder
            The fitted encoder, carrying its metric.
        """
        if center:
            mean = bkd.mean(f_grid_data, axis=1)
            mean = bkd.reshape(mean, (mean.shape[0], 1))
        else:
            full_dim = f_grid_data.shape[0]
            mean = bkd.zeros((full_dim, 1))
        centered = f_grid_data - mean

        # One decomposition. Omitting nterms keeps every mode carrying
        # variance, and truncation is then a slice of the result -- the
        # solver computes the whole spectrum regardless, so asking for a
        # count up front would mean deciding it before the spectrum that
        # decides it is known, and paying for a second pass to find out.
        eig_vals, eig_vecs = default_snapshot_eigensolver(
            bkd, metric
        ).solve(centered, metric=metric)
        n = resolve_nterms(
            eig_vals,
            bkd,
            nterms=latent_dim,
            variance_fraction=variance_fraction,
        )
        # The solver canonicalizes signs, so a basis fitted here agrees
        # with one DataDrivenKLE builds from the same snapshots rather
        # than differing by a per-mode sign.
        return cls(eig_vecs[:, :n], mean, bkd, metric=metric)
