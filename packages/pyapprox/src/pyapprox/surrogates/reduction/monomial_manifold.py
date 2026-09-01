"""Polynomial manifold dimensionality reduction.

Implements the greedy construction of Schwerdtner & Peherstorfer (2024),
"Greedy construction of quadratic manifolds for nonlinear dimensionality
reduction and nonlinear model reduction" (arXiv:2403.06732).

A polynomial manifold augments the linear PCA decoder with a nonlinear
correction

.. math::

    \\mathrm{decode}(z) = \\mu + V z + W h(z)

where :math:`h` is a fixed monomial feature map and :math:`W` is fitted
by regularized least squares. The degree is the feature map's to choose:
``degrees=(0, 1, 2)`` gives a quadratic manifold, adding 3 makes it
cubic, and an arbitrary multi-index set gives an anisotropic one.
Nothing here is specific to a degree, hence the name.

The default includes the constant and linear monomials. Eq. (5) states
the correction over quadratic terms alone, on the grounds that the
decoder's own :math:`\\mu` and :math:`V z` cover the lower degrees; but
:math:`W` is fitted to a residual living in the orthogonal complement of
the basis, which neither of those reaches, so the low-degree terms are
not redundant there. See
:class:`~pyapprox.surrogates.reduction.feature_maps.MonomialFeatureMap`
for the measurements.

The encoder stays *linear*,

.. math::

    \\mathrm{encode}(s) = V^T (s - \\mu)

inverting only the linear part and ignoring :math:`W`. The asymmetry is
deliberate. The encoder matching the decoder would be the closest-point
projection, a nonlinear least-squares solve per sample whose tangent is
state dependent, so a model reduced in those coordinates would carry
that Jacobian through every time derivative. With the linear encoder
:math:`\\dot z = V^T \\dot s` holds exactly and the latent dynamics keep
the form they have under a linear basis.

**Why the correction improves accuracy at a fixed latent dimension.**
The linear part reconstructs :math:`P_V s` -- the component inside
:math:`\\mathrm{span}(V)` -- leaving a residual :math:`s - P_V s` that
lies entirely in the orthogonal complement. :math:`W` is fitted to
predict exactly that residual, so its columns point in directions the
basis cannot represent at all. The correction is not adjusting the
linear answer, it is reaching outside the subspace, which is why
:math:`V^T` annihilates it and
:math:`\\mathrm{encode}(\\mathrm{decode}(z)) = z` still holds exactly.

The coefficients on those extra directions are not free parameters:
they are :math:`h(z)`, fixed functions of the coordinates already
retained. Adding :math:`p` linear modes would cost :math:`p` more latent
dimensions, whereas here the same :math:`p` directions cost none,
because their amplitudes are asserted to be polynomial in :math:`z`.
That assertion holds when the data lies near a curved
:math:`r`-dimensional manifold rather than a flat subspace -- discarded
energy that is *curvature* is predictable from the retained
coordinates, while genuinely high-rank data is not. A parabola in the
plane has full rank 2, so a linear basis of one vector has irreducible
error, yet one coordinate plus a quadratic term reproduces it exactly.

The greedy sweep is the other half of this: it selects the directions
whose complement is most predictable from them, not the highest-energy
ones, so a weaker direction leaving a cleanly quadratic residual is
preferred over a stronger one that does not.
:math:`\\mathrm{decode}(\\mathrm{encode}(s)) \\neq s` in general -- the
codes are not error-minimizing -- but :math:`W` is fitted against the
residual *this* projection leaves, so the correction repairs the error
this encoder actually makes.

Basis construction is greedy forward selection. The candidate pool is the
first :math:`m` left-singular vectors of the centered snapshot matrix. At
each iteration the algorithm sweeps every not-yet-selected candidate and
adds the one whose inclusion -- after fitting the optimal correction for
the enlarged subspace -- minimizes the reconstruction error. The leading
singular vectors are therefore not necessarily selected: the method picks
the directions the feature map can correct most efficiently. The repeated
inner least-squares solves are made cheap by reusing a single
decomposition of the snapshot matrix, so each candidate evaluation
scales with the number of snapshots rather than the ambient dimension
(Section 3.2).
"""

from __future__ import annotations

from typing import Generic, List, Optional, Sequence, Tuple

from pyapprox.surrogates.kle.snapshot_eigensolvers import (
    SnapshotDecomposition,
    SnapshotEigenSolverProtocol,
)
from pyapprox.surrogates.reduction.feature_maps import (
    DifferentiableFeatureMap,
    MonomialFeatureMap,
)
from pyapprox.surrogates.reduction.manifold_scoring import (
    ManifoldScorer,
    center_and_decompose,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import InnerProductProtocol


class MonomialManifoldEncoder(Generic[Array]):
    """Linear encoder with a monomial-correction decoder.

    Constructed via :meth:`fit_from_data`. Stores the selected basis
    ``V``, the weight matrix ``W``, the feature map ``h``, and the
    snapshot mean.
    """

    def __init__(
        self,
        basis: Array,
        weights: Array,
        feature_map: DifferentiableFeatureMap[Array],
        mean: Array,
        bkd: Backend[Array],
        selected_indices: Optional[List[int]] = None,
        fit_gamma: Optional[float] = None,
        gram_cond: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        basis : Array
            Shape: (full_dim, latent_dim). The selected singular vectors.
        weights : Array
            Shape: (full_dim, nterms). The correction weight matrix W.
        feature_map : DifferentiableFeatureMap
            The fixed nonlinear feature map h. Its Jacobian is what lets
            a consumer differentiate the decoder in closed form.
        mean : Array
            Shape: (full_dim, 1). The snapshot mean, zero if uncentered.
        bkd : Backend
            Computational backend.
        selected_indices : list of int, optional
            Indices into the candidate pool selected by the greedy
            procedure, in selection order. Retained for inspection.
        fit_gamma : float, optional
            The regularization actually used for the final W fit, after
            any validation selection. Retained for diagnostics.
        gram_cond : float, optional
            Condition number of the regularized correction Gram
            ``h h^T + fit_gamma I`` inverted in the fit, a direct measure
            of the least-squares conditioning. Retained for diagnostics.
        """
        self._basis = basis
        self._weights = weights
        self._feature_map = feature_map
        self._mean = mean
        self._bkd = bkd
        self._selected_indices = selected_indices
        self._fit_gamma = fit_gamma
        self._gram_cond = gram_cond

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def full_dim(self) -> int:
        """Dimension of the ambient state."""
        return int(self._basis.shape[0])

    def latent_dim(self) -> int:
        """Dimension of the latent space."""
        return int(self._basis.shape[1])

    def encode(self, samples: Array) -> Array:
        """Full to latent, ``z = V^T (s - mu)``.

        ``(full_dim, N) -> (latent_dim, N)``. Linear even though the
        decoder is not, so the time derivative maps exactly.
        """
        return self._bkd.dot(self._basis.T, samples - self._mean)

    def decode(self, latents: Array) -> Array:
        """Latent to full, ``s = mu + V z + W h(z)``.

        ``(latent_dim, N) -> (full_dim, N)``.
        """
        bkd = self._bkd
        linear = bkd.dot(self._basis, latents) + self._mean
        correction = bkd.dot(self._weights, self._feature_map(latents))
        return linear + correction

    def decode_jacobian(self, latents: Array) -> Array:
        """``d decode / d z`` at a single latent coordinate.

        ``V + W dh/dz``, in closed form from the feature map's own
        Jacobian rather than by differencing :meth:`decode`.

        Parameters
        ----------
        latents : Array
            Shape: (latent_dim, 1). A single latent coordinate.

        Returns
        -------
        Array
            Shape: (full_dim, latent_dim).
        """
        if latents.shape[1] != 1:
            raise ValueError(
                "decode_jacobian takes a single latent coordinate of "
                f"shape (latent_dim, 1), got {tuple(latents.shape)}."
            )
        feature_jac = self._feature_map.jacobian(latents)[:, :, 0]
        return self._basis + self._bkd.dot(self._weights, feature_jac)

    def is_isometry(self) -> bool:
        """Whether encoding preserves the norm. Never, for a manifold.

        ``||decode(z1) - decode(z2)||`` picks up ``W(h(z1) - h(z2))``,
        whose size depends on where in the latent space the points lie,
        so a coefficient distance is not a field distance. A consumer
        that minimizes a coefficient residual as a proxy for a field
        error -- least-squares operator learning, on its output side --
        must refuse this encoder rather than treat the discrepancy as
        small.
        """
        return False

    def basis(self) -> Array:
        """The basis matrix V, shape (full_dim, latent_dim)."""
        return self._basis

    def weights(self) -> Array:
        """The correction weight matrix W, shape (full_dim, nterms)."""
        return self._weights

    def feature_map(self) -> DifferentiableFeatureMap[Array]:
        """The nonlinear feature map h."""
        return self._feature_map

    def mean(self) -> Array:
        """The snapshot mean, shape (full_dim, 1)."""
        return self._mean

    def selected_indices(self) -> Optional[List[int]]:
        """Candidate indices chosen by the greedy sweep, in order."""
        return self._selected_indices

    def fit_gamma(self) -> Optional[float]:
        """Regularization used for the final W fit."""
        return self._fit_gamma

    def gram_cond(self) -> Optional[float]:
        """Condition number of the regularized Gram at the fit."""
        return self._gram_cond

    @classmethod
    def fit_from_data(
        cls,
        snapshots: Array,
        bkd: Backend[Array],
        latent_dim: int,
        feature_map: Optional[DifferentiableFeatureMap[Array]] = None,
        degrees: Sequence[int] = (0, 1, 2),
        gamma: float = 1e-6,
        ncandidates: Optional[int] = None,
        candidate_factor: int = 10,
        center: bool = True,
        precomputed: Optional[
            Tuple[Array, SnapshotDecomposition[Array]]
        ] = None,
        metric: Optional[InnerProductProtocol[Array]] = None,
        eigensolver: Optional[SnapshotEigenSolverProtocol[Array]] = None,
        validation_data: Optional[Array] = None,
        gamma_grid: Optional[Sequence[float]] = None,
        mean: Optional[Array] = None,
    ) -> "MonomialManifoldEncoder[Array]":
        """Greedily construct a polynomial manifold from snapshot data.

        Parameters
        ----------
        snapshots : Array
            Shape: (full_dim, N). Training snapshots, columns are points.
        bkd : Backend
            Computational backend.
        latent_dim : int
            Reduced dimension, the number of basis vectors to select.
        feature_map : DifferentiableFeatureMap, optional
            The nonlinear feature map h. If None, a
            :class:`MonomialFeatureMap` of the given ``degrees`` is
            built.
        degrees : sequence of int
            Degrees for the default monomial feature map: ``(2,)``
            quadratic, ``(2, 3)`` cubic. Ignored if ``feature_map`` is
            given.
        gamma : float
            Tikhonov regularization, used for two distinct purposes.
            (1) Greedy basis selection: the candidate-scoring objective
            always uses this fixed value to rank singular vectors,
            regardless of ``gamma_grid``. (2) The final W fit (Eq. 6),
            used only when ``gamma_grid`` is not given; otherwise the fit
            gamma is validation-selected.

            The *selection* gamma is a fixed default and is not
            validated, so the chosen basis can in principle depend on it.
            Only the *fit* gamma is tied to held-out reconstruction
            error.
        ncandidates : int, optional
            Size of the candidate singular-vector pool. Defaults to
            ``candidate_factor * latent_dim``, capped at the rank.
        candidate_factor : int
            Multiplier for the default pool size. The paper uses 10.
        center : bool
            Subtract the snapshot mean before decomposing.
        precomputed : tuple, optional
            ``(mean, decomposition)`` to reuse instead of recomputing
            the decomposition, which is the dominant cost and is shared
            across methods and dimensions at a fixed training set. The
            caller must have centered consistently with ``center``.
        metric : InnerProductProtocol, optional
            The inner product the basis is orthonormal in. None is
            Euclidean. A mesh-weighted metric makes the reduction
            measure error in the field norm rather than in a norm that
            happens to weight every node equally.
        eigensolver : SnapshotEigenSolverProtocol, optional
            How the basis is extracted. Defaults to the solver matching
            the metric.
        validation_data : Array, optional
            Shape: (full_dim, M). Held-out snapshots used to select
            ``gamma`` from ``gamma_grid``. Required if ``gamma_grid`` is
            given.
        gamma_grid : sequence of float, optional
            Candidate regularization values. If given, the value
            minimizing reconstruction error on ``validation_data`` is
            used (Section 5.1).
        mean : Array, optional
            Mean used for centering, shape (full_dim,) or (full_dim, 1).
            Computed from ``snapshots`` if None. Pass the mean of the
            full data when ``snapshots`` is a deliberately extremal
            subset, whose own mean is a biased centroid. Ignored when
            ``precomputed`` carries its own mean.

        Returns
        -------
        MonomialManifoldEncoder
        """
        if feature_map is None:
            feature_map = MonomialFeatureMap(
                latent_dim, bkd, degrees=degrees
            )
        elif feature_map.nreduced() != latent_dim:
            raise ValueError(
                f"feature_map.nreduced() ({feature_map.nreduced()}) must "
                f"equal latent_dim ({latent_dim})"
            )

        # One SVD of the snapshot matrix, reused throughout (Section 3.2),
        # or injected by the caller to share it across methods.
        centered, mean, decomposition = center_and_decompose(
            snapshots,
            bkd,
            center=center,
            precomputed=precomputed,
            mean=mean,
            metric=metric,
            eigensolver=eigensolver,
        )
        rank = decomposition.nterms()

        if ncandidates is None:
            ncandidates = candidate_factor * latent_dim
        ncandidates = min(ncandidates, rank)
        if latent_dim > ncandidates:
            raise ValueError(
                f"latent_dim ({latent_dim}) cannot exceed the candidate "
                f"pool size ({ncandidates})"
            )

        # Every snapshot's coordinates in the full basis, reused for
        # every candidate evaluation (Section 3.2). Carried by the
        # decomposition rather than recomputed, so the basis and the
        # coordinates cannot disagree about a column's sign.
        scorer = ManifoldScorer(decomposition.coordinates, gamma, bkd)

        selected = cls._greedy_select(
            scorer, feature_map, latent_dim, ncandidates
        )
        basis = decomposition.eigenvectors[:, selected]

        if gamma_grid is not None:
            if validation_data is None:
                raise ValueError(
                    "gamma_grid requires validation_data to select gamma"
                )
            selected_gamma = scorer.select_gamma(
                centered,
                basis,
                feature_map,
                gamma_grid,
                validation_data - mean,
            )
            assert isinstance(selected_gamma, float)
            gamma = selected_gamma

        weights = scorer.fit_weights(centered, basis, feature_map, gamma)
        gram_cond = scorer.gram_condition(
            centered, basis, feature_map, gamma
        )
        return cls(
            basis,
            weights,
            feature_map,
            mean,
            bkd,
            selected_indices=selected,
            fit_gamma=gamma,
            gram_cond=gram_cond,
        )

    @staticmethod
    def _greedy_select(
        scorer: ManifoldScorer[Array],
        feature_map: DifferentiableFeatureMap[Array],
        latent_dim: int,
        ncandidates: int,
    ) -> List[int]:
        """Forward-greedy selection of singular-vector indices.

        At each iteration, sweep every not-yet-selected candidate, score
        it by the reconstruction error achievable with the optimal
        correction for the enlarged subspace (objective J', Eq. 12), and
        keep the minimizer. Scoring is delegated to the scorer, which
        works in the precomputed SVD coordinates.

        The scorer is given the feature map's index set, and restricts it
        to each trial dimension itself, so the terms a trial is scored
        with are the ones that trial supports.
        """
        selected: List[int] = []
        candidates = list(range(ncandidates))
        indices = feature_map.indices()

        for _ in range(latent_dim):
            scores = scorer.greedy_scores(selected, candidates, indices)
            best = candidates[
                min(range(len(scores)), key=lambda i: scores[i])
            ]
            selected.append(best)
            candidates.remove(best)
        return selected


def build_monomial_manifold_encoder(
    trajectories: Sequence[Array],
    bkd: Backend[Array],
    latent_dim: int,
    degrees: Sequence[int] = (2,),
    feature_map: Optional[DifferentiableFeatureMap[Array]] = None,
    gamma: float = 1e-6,
    ncandidates: Optional[int] = None,
    center: bool = True,
    precomputed: Optional[
        Tuple[Array, SnapshotDecomposition[Array]]
    ] = None,
    metric: Optional[InnerProductProtocol[Array]] = None,
    eigensolver: Optional[SnapshotEigenSolverProtocol[Array]] = None,
    validation_data: Optional[Array] = None,
    gamma_grid: Optional[Sequence[float]] = None,
    mean: Optional[Array] = None,
) -> MonomialManifoldEncoder[Array]:
    """Build a manifold encoder from a list of trajectory matrices.

    Parameters
    ----------
    trajectories : sequence of Array
        Each of shape (full_dim, ntimes_i); stacked column-wise.
    latent_dim : int
        Reduced dimension.
    degrees : sequence of int
        Monomial degrees for the default feature map. Ignored if
        ``feature_map`` is given.

    The remaining arguments are passed through to
    :meth:`MonomialManifoldEncoder.fit_from_data`.

    Returns
    -------
    MonomialManifoldEncoder
    """
    return MonomialManifoldEncoder.fit_from_data(
        bkd.hstack(list(trajectories)),
        bkd,
        latent_dim=latent_dim,
        feature_map=feature_map,
        degrees=degrees,
        gamma=gamma,
        ncandidates=ncandidates,
        center=center,
        precomputed=precomputed,
        metric=metric,
        eigensolver=eigensolver,
        validation_data=validation_data,
        gamma_grid=gamma_grid,
        mean=mean,
    )
