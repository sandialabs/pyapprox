r"""Reading a KLE basis as a reduction rather than as an expansion.

A KLE and a PCA encoder hold the same basis, mean and spectrum, and
traverse them in opposite directions.

The expansion runs coefficients to field,

.. math:: f = \bar{f} + \sum_i \sqrt{\lambda_i}\, \phi_i z_i,

where :math:`z` is standardized -- mean zero, unit variance -- so the
:math:`\sqrt{\lambda}` scaling is what gives a realization its
covariance. The encoder runs field to coordinates and back,

.. math:: z = V^T M (f - \bar{f}), \qquad f = V z + \bar{f},

where :math:`z` is a coordinate rather than a standardized variable, so
the basis is used unweighted and no :math:`\sqrt{\lambda}` appears.

:class:`KLEEncoder` supplies the second over any object satisfying
``KLEProtocol``, so a basis solved once can be used either way without
being rebuilt, and an encoder can be persisted by ``save_kle`` because
what is stored is still a KLE.

**Composition rather than more methods on the KLE.** The two readings do
not always coincide: a KLE may exponentiate its expansion, and then its
decode is not linear and no linear projection inverts it. Adding
``encode`` to that class would mean shipping a method whose only job in
the lognormal case is to refuse, and an ``isinstance`` check would pass
for an object that raises on use. Keeping the encoder separate makes the
capability's absence a fact about which objects exist: a lognormal KLE
simply has no encoder.

The basis is shared rather than copied, so the two views cannot drift.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.surrogates.kle.data_driven_kle import DataDrivenKLE
from pyapprox.surrogates.kle.protocols import KLEProtocol
from pyapprox.surrogates.kle.snapshot_eigensolvers import (
    SnapshotEigenSolverProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import InnerProductProtocol


@runtime_checkable
class _ExponentiatingKLE(Protocol):
    """A KLE that can report whether it exponentiates its expansion.

    Not every KLE has the capability to report, and one that cannot has
    nothing to exponentiate -- the flag only exists on the classes that
    implement it. Declared as a protocol so the question is asked by
    ``isinstance`` rather than by sniffing for an attribute.
    """

    def use_log(self) -> bool:
        ...


class KLEEncoder(Generic[Array]):
    r"""The reductive reading of a KLE basis.

    .. math::
        z = V^T M (f - \bar{f}), \qquad f = V z + \bar{f}

    Uses the *unweighted* basis, so a latent coordinate is the projection
    itself. The generative side scales by :math:`\sqrt{\lambda}` instead,
    because its argument is a standardized random coefficient rather than
    a coordinate; the two readings therefore have different units and
    ``decode`` is deliberately not ``kle(z)``.

    Parameters
    ----------
    kle : KLEProtocol[Array]
        The basis to read reductively. Held rather than copied, so this
        view cannot drift from the expansion it came from.
    metric : InnerProductProtocol, optional
        The inner product the basis is orthonormal in. None means
        Euclidean. Passing the wrong one makes ``encode`` something
        other than a projection: it stops being idempotent and stops
        returning the nearest point in the subspace, with no error, so
        it must be the metric the basis was *built* in.

    Raises
    ------
    TypeError
        If ``kle`` does not satisfy ``KLEProtocol``.
    ValueError
        If ``kle`` exponentiates its expansion. A lognormal decode is
        not linear, so no linear projection inverts it. Build the
        encoder over the Gaussian basis being exponentiated and take
        logs before encoding.
    """

    def __init__(
        self,
        kle: KLEProtocol[Array],
        metric: Optional[InnerProductProtocol[Array]] = None,
    ) -> None:
        if not isinstance(kle, KLEProtocol):
            raise TypeError(
                f"kle must satisfy KLEProtocol, got {type(kle).__name__}"
            )
        if isinstance(kle, _ExponentiatingKLE) and kle.use_log():
            raise ValueError(
                f"{type(kle).__name__} exponentiates its expansion, so "
                "its decode is not linear and no linear projection "
                "inverts it. Encode against the Gaussian basis it "
                "exponentiates, taking logs of the samples first."
            )
        basis = kle.eigenvectors()
        if metric is not None and metric.nstates() != int(basis.shape[0]):
            raise ValueError(
                f"metric is defined on {metric.nstates()} states but the "
                f"basis has {int(basis.shape[0])} rows"
            )
        self._kle = kle
        self._metric = metric
        self._bkd = kle.bkd()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def kle(self) -> KLEProtocol[Array]:
        """Return the wrapped expansion."""
        return self._kle

    def metric(self) -> Optional[InnerProductProtocol[Array]]:
        """Return the inner product the basis is orthonormal in."""
        return self._metric

    def full_dim(self) -> int:
        """Dimension of the space the basis lives in."""
        return int(self._kle.eigenvectors().shape[0])

    def latent_dim(self) -> int:
        """Dimension of the reduced space."""
        return self._kle.nterms()

    def basis(self) -> Array:
        """The unweighted basis, ``(full_dim, latent_dim)``."""
        return self._kle.eigenvectors()

    def mean(self) -> Array:
        """The mean, shape ``(full_dim, 1)``.

        Column-shaped for the samples-are-columns convention, where the
        KLE's own ``mean_field`` is 1-D to match ``eigenvectors``.
        """
        return self._kle.mean_field()[:, None]

    def encode(self, samples: Array) -> Array:
        r"""Full to latent, :math:`z = V^T M (f - \bar{f})`.

        Parameters
        ----------
        samples : Array
            Shape ``(full_dim, nsamples)``.

        Returns
        -------
        Array
            Shape ``(latent_dim, nsamples)``.
        """
        if samples.ndim != 2:
            raise ValueError(
                f"samples must be 2D (full_dim, nsamples), got "
                f"ndim={samples.ndim}"
            )
        centered = samples - self.mean()
        if self._metric is not None:
            centered = self._metric.apply(centered)
        return self._bkd.dot(self.basis().T, centered)

    def decode(self, latents: Array) -> Array:
        r"""Latent to full, :math:`f = V z + \bar{f}`.

        Parameters
        ----------
        latents : Array
            Shape ``(latent_dim, nsamples)``.

        Returns
        -------
        Array
            Shape ``(full_dim, nsamples)``.
        """
        if latents.ndim != 2:
            raise ValueError(
                f"latents must be 2D (latent_dim, nsamples), got "
                f"ndim={latents.ndim}"
            )
        return self._bkd.dot(self.basis(), latents) + self.mean()

    def decode_std(self, std_latents: Array) -> Array:
        r"""Propagate latent std to full space, without the mean shift.

        :math:`\sigma_f = \sqrt{V^2 \sigma_z^2}`, which assumes the
        latent coordinates are uncorrelated. Exact when their covariance
        is diagonal, an under-estimate otherwise, and stated here rather
        than hidden because the result has the shape and units of a
        standard deviation either way.
        """
        return self._bkd.sqrt(
            self._bkd.dot(self.basis() ** 2, std_latents**2)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(full_dim={self.full_dim()}, "
            f"latent_dim={self.latent_dim()})"
        )


def fit_kle_encoder(
    samples: Array,
    bkd: Backend[Array],
    latent_dim: Optional[int] = None,
    variance_fraction: Optional[float] = None,
    center: bool = True,
    metric: Optional[InnerProductProtocol[Array]] = None,
    eigensolver: Optional[SnapshotEigenSolverProtocol[Array]] = None,
) -> KLEEncoder[Array]:
    """Build an encoder from snapshots in one call.

    Convenience over ``KLEEncoder(DataDrivenKLE(...))``, which is the
    common case and is otherwise two steps. Deliberately a function
    rather than a fitter class: the three things that genuinely vary
    here -- the metric, the eigensolver, and the truncation policy --
    are already injectable abstractions, so a fitter would have nothing
    left to choose between. Given those, the basis is determined rather
    than estimated, which is what separates this from the coefficient
    fitters elsewhere in ``surrogates``.

    Centering defaults to True here, unlike ``DataDrivenKLE``: a
    reduction is almost always taken about the data's mean, while an
    expansion may be taken about anything.

    Parameters
    ----------
    samples : Array
        Shape ``(full_dim, nsamples)``, one snapshot per column.
    bkd : Backend[Array]
        Computational backend.
    latent_dim : int, optional
        Number of modes to keep. Exclusive with ``variance_fraction``;
        with neither, every mode carrying variance is kept.
    variance_fraction : float, optional
        Keep the fewest modes carrying this fraction of the variance.
    center : bool
        Subtract the sample mean before decomposing.
    metric : InnerProductProtocol, optional
        The inner product the basis is orthonormal in, and the one
        ``encode`` projects with. None means Euclidean, which biases the
        basis toward wherever a non-uniform mesh is refined.
    eigensolver : SnapshotEigenSolverProtocol, optional
        How the basis is extracted. Defaults to the solver matching the
        metric.

    Returns
    -------
    KLEEncoder[Array]
        Wrapping the fitted expansion, reachable through
        :meth:`KLEEncoder.kle` and storable with ``save_kle``.
    """
    return KLEEncoder(
        DataDrivenKLE(
            samples,
            nterms=latent_dim,
            variance_fraction=variance_fraction,
            center=center,
            metric=metric,
            eigensolver=eigensolver,
            bkd=bkd,
        ),
        metric,
    )
