r"""The space a reduced model's state lives in, and how it is fitted.

A reduced state :math:`z` stands for the full state

.. math:: u = g(t) + \bar{u} + V z,

where :math:`g` is the lift carrying the boundary data
(:mod:`pyapprox.pde.rom.lift`), :math:`\bar{u}` a fixed reference (the
snapshot mean) and :math:`V` the basis. The trial space owns
:math:`\bar{u}` and :math:`V`; the lift stays separate because it
depends on time and the basis must not.

:math:`V` is used only through its action and transpose action, so a
basis held out of core (``StreamingBasis``) serves as well as a resident
one. The metric :math:`M` the basis was built in travels with it,
because the best approximation of a state in the span of :math:`V` is
the :math:`M`-orthogonal projection,

.. math:: z = (V^T M V)^{-1} V^T M (u - g - \bar{u}),

and pairing a basis with the wrong metric returns something that is not
the nearest point, with no error. The projection solves with the
:math:`r \times r` Gram rather than assuming :math:`V^T M V = I`, so a
basis that is not exactly orthonormal (from a greedy method, or a POD
basis evaluated in a different metric) still projects correctly.

POD fitting is :func:`~pyapprox.surrogates.kle.fit_kle_encoder`, whose
metric, eigensolver and truncation policy are already injectable; this
module adds the reduced-model view of its result and the diagnostics
used to choose the truncation.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.pde.rom.snapshots import SnapshotSet
from pyapprox.surrogates.kle import (
    BasisOperatorProtocol,
    DataDrivenKLE,
    SnapshotEigenSolverProtocol,
    as_basis_operator,
    fit_kle_encoder,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import (
    EuclideanInnerProduct,
    InnerProductProtocol,
)


@runtime_checkable
class TrialSpaceProtocol(Protocol, Generic[Array]):
    r"""The affine space :math:`\bar{u} + \mathrm{span}(V)` of a reduced
    model, excluding the time-dependent lift."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return the full-order dimension."""
        ...

    def ncodes(self) -> int:
        """Return the reduced dimension."""
        ...

    def reference(self) -> Array:
        r"""Return :math:`\bar{u}`. Shape: (nstates,)."""
        ...

    def apply(self, codes: Array) -> Array:
        """Return ``V codes``. Shape: (ncodes, k) -> (nstates, k)."""
        ...

    def apply_transpose(self, fields: Array) -> Array:
        """Return ``V^T fields``. Shape: (nstates, k) -> (ncodes, k)."""
        ...


class LinearTrialSpace(Generic[Array]):
    r"""A fixed reference plus the span of a basis, in a metric.

    Parameters
    ----------
    basis : BasisOperatorProtocol
        :math:`V`, shape (nstates, ncodes), full column rank.
    reference : Array
        :math:`\bar{u}`. Shape: (nstates,).
    metric : InnerProductProtocol, optional
        The inner product projections are taken in; Euclidean when
        omitted.

    Raises
    ------
    TypeError
        If ``basis`` or ``metric`` does not satisfy its protocol.
    ValueError
        If the shapes disagree.
    """

    def __init__(
        self,
        basis: BasisOperatorProtocol[Array],
        reference: Array,
        metric: Optional[InnerProductProtocol[Array]] = None,
    ) -> None:
        if not isinstance(basis, BasisOperatorProtocol):
            raise TypeError(
                f"basis must satisfy BasisOperatorProtocol, got {type(basis).__name__}"
            )
        bkd = basis.bkd()
        nstates = basis.nstates()
        if tuple(reference.shape) != (nstates,):
            raise ValueError(
                f"reference must have shape ({nstates},), got {tuple(reference.shape)}"
            )
        if metric is None:
            metric = EuclideanInnerProduct(nstates, bkd)
        if not isinstance(metric, InnerProductProtocol):
            raise TypeError(
                f"metric must satisfy InnerProductProtocol, got {type(metric).__name__}"
            )
        if metric.nstates() != nstates:
            raise ValueError(
                f"metric acts on {metric.nstates()} states but the basis "
                f"has {nstates} rows"
            )
        self._basis = basis
        self._reference = reference
        self._metric = metric
        self._bkd = bkd
        # r x r and fixed with the basis, so formed once.
        self._gram = reduced_gram(basis, metric)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return the full-order dimension."""
        return self._basis.nstates()

    def ncodes(self) -> int:
        """Return the reduced dimension."""
        return self._basis.nterms()

    def basis(self) -> BasisOperatorProtocol[Array]:
        """Return the basis."""
        return self._basis

    def metric(self) -> InnerProductProtocol[Array]:
        """Return the metric projections are taken in."""
        return self._metric

    def gram(self) -> Array:
        """Return ``V^T M V``. Shape: (ncodes, ncodes)."""
        return self._gram

    def reference(self) -> Array:
        r"""Return :math:`\bar{u}`. Shape: (nstates,)."""
        return self._reference

    def apply(self, codes: Array) -> Array:
        """Return ``V codes``. Shape: (ncodes, k) -> (nstates, k)."""
        return self._basis.apply(codes)

    def apply_transpose(self, fields: Array) -> Array:
        """Return ``V^T fields``. Shape: (nstates, k) -> (ncodes, k)."""
        return self._basis.apply_transpose(fields)

    def project(self, fields: Array) -> Array:
        r"""Return the coordinates of the best approximation in the metric.

        :math:`z = (V^T M V)^{-1} V^T M (f - \bar{u})`; ``fields`` must
        already have the lift removed. Shape: (nstates, k) -> (ncodes, k).
        """
        centered = fields - self._reference[:, None]
        return self._bkd.solve(
            self._gram, self.apply_transpose(self._metric.apply(centered))
        )

    def reconstruct(self, codes: Array) -> Array:
        r"""Return :math:`\bar{u} + V z`. Shape: (ncodes, k) -> (nstates, k)."""
        return self._reference[:, None] + self.apply(codes)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nstates={self.nstates()}, "
            f"ncodes={self.ncodes()}, metric={self._metric!r})"
        )


def reduced_gram(
    basis: BasisOperatorProtocol[Array], metric: InnerProductProtocol[Array]
) -> Array:
    r"""Return :math:`V^T M V`. Shape: (nterms, nterms).

    With the mass matrix as the metric this is the reduced mass; it is
    the identity exactly when the basis is orthonormal in the metric.
    Forms :math:`V` column by column through ``apply``, so the cost is
    one ambient-sized ``(nstates, nterms)`` array, never
    ``(nstates, nstates)``.
    """
    bkd = basis.bkd()
    columns = basis.apply(bkd.eye(basis.nterms()))
    return basis.apply_transpose(metric.apply(columns))


def fit_pod_trial_space(
    snapshots: SnapshotSet[Array],
    bkd: Backend[Array],
    latent_dim: Optional[int] = None,
    variance_fraction: Optional[float] = None,
    metric: Optional[InnerProductProtocol[Array]] = None,
    eigensolver: Optional[SnapshotEigenSolverProtocol[Array]] = None,
) -> LinearTrialSpace[Array]:
    """Fit a POD trial space centered at the snapshot mean.

    The snapshots should already be homogenized
    (:func:`~pyapprox.pde.rom.homogenize`), so the basis and the mean
    vanish on the constrained DOFs.

    Parameters
    ----------
    snapshots : SnapshotSet
        Homogenized snapshots.
    bkd : Backend
        Computational backend.
    latent_dim : int, optional
        Number of modes to keep. Exclusive with ``variance_fraction``;
        with neither, every mode above the numerical rank floor is kept.
    variance_fraction : float, optional
        Keep the fewest modes carrying this fraction of the snapshot
        energy in the metric.
    metric : InnerProductProtocol, optional
        The inner product the basis is orthonormal in, typically the
        mass matrix (``MassInnerProduct``); Euclidean when omitted.
    eigensolver : SnapshotEigenSolverProtocol, optional
        How the basis is extracted; defaults to the solver matching the
        metric.
    """
    encoder = fit_kle_encoder(
        snapshots.states,
        bkd,
        latent_dim=latent_dim,
        variance_fraction=variance_fraction,
        center=True,
        metric=metric,
        eigensolver=eigensolver,
    )
    return LinearTrialSpace(
        as_basis_operator(encoder.basis(), bkd), encoder.mean()[:, 0], metric
    )


def snapshot_spectrum(
    snapshots: SnapshotSet[Array],
    bkd: Backend[Array],
    metric: Optional[InnerProductProtocol[Array]] = None,
    eigensolver: Optional[SnapshotEigenSolverProtocol[Array]] = None,
) -> Array:
    r"""Return the singular values of the centered snapshots in the metric.

    Descending, every mode above the numerical rank floor. For a POD
    basis of :math:`r` modes fitted to the same snapshots and metric,

    .. math:: \sum_j \|s_j - \Pi_r s_j\|_M^2 = \sum_{i > r} \sigma_i^2,

    so the tail of this spectrum is the training projection error each
    truncation would leave; inspect it to choose ``latent_dim``.
    Computed by the same eigensolver as :func:`fit_pod_trial_space`, so
    the decay plotted is exactly what that function truncates.
    """
    kle = DataDrivenKLE(
        snapshots.states,
        center=True,
        metric=metric,
        eigensolver=eigensolver,
        bkd=bkd,
    )
    return kle.singular_values()


def projection_errors(trial_space: LinearTrialSpace[Array], fields: Array) -> Array:
    r"""Return :math:`\|f_j - \Pi f_j\|_M` for each column of ``fields``.

    ``fields`` must have the lift removed, like the snapshots the trial
    space was fitted to. Shape: (nstates, k) -> (k,).
    """
    residual = fields - trial_space.reconstruct(trial_space.project(fields))
    return trial_space.metric().norm(residual)
