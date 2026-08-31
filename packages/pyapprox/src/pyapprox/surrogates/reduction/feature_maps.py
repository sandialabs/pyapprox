"""Polynomial feature maps for manifold corrections.

A feature map :math:`h : \\mathbb{R}^r \\to \\mathbb{R}^p` lifts a reduced
coordinate to a vector of monomials.  It is the only component that
distinguishes a quadratic manifold from a cubic (or higher-order) one: the
greedy basis selection, weight-matrix fit, and decoder are all agnostic to
which monomials :math:`h` produces.

Feature maps here include the constant and linear monomials by default.
It is tempting to exclude them on the grounds that the decoder
:math:`g(z) = \\mu + V z + W h(z)` already has a constant and a linear
term, but :math:`W` is fitted to the residual :math:`s - P_V s`, which
lies in the orthogonal complement of the basis, where neither
:math:`\\mu` nor :math:`V z` reaches. See :class:`MonomialFeatureMap`
for the measurements.

Evaluating features and differentiating them are separate contracts.
:class:`FeatureMap` requires only evaluation, which is all a linear encoder
and the weight-matrix fit consume; :class:`DifferentiableFeatureMap` adds
the Jacobian that a nonlinear closest-point encoder needs. Consumers depend
on whichever is the weaker protocol that carries them.
"""

from __future__ import annotations

from typing import Generic, List, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import numpy.typing as npt

from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_level_indices,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class FeatureMap(Protocol[Array]):
    """Maps reduced coordinates to a vector of nonlinear features.

    All implementations operate column-wise: an input of shape ``(r, k)``
    (``k`` reduced coordinates) produces an output of shape ``(p, k)``.
    """

    def nreduced(self) -> int:
        """Reduced dimension ``r`` the map expects as input."""
        ...

    def nterms(self) -> int:
        """Number of features ``p`` produced."""
        ...

    def indices(self) -> Array:
        """The multi-index set defining the features, shape (r, p).

        Required because a construction working at a sequence of
        increasing dimensions needs the terms a smaller subspace
        supports, which is the index set restricted to those variables
        (:func:`~pyapprox.surrogates.affine.indices.restrict_indices_to_leading_vars`).
        Deriving them from the parameters that generated the set instead
        would only work for maps built that way.
        """
        ...

    def __call__(self, codes: Array) -> Array:
        """Evaluate features.

        Parameters
        ----------
        codes : Array
            Shape: (r, k). Reduced coordinates.

        Returns
        -------
        Array
            Shape: (p, k). Feature vectors.
        """
        ...


@runtime_checkable
class DifferentiableFeatureMap(FeatureMap[Array], Protocol):
    """A feature map that can also supply its Jacobian.

    Required by a closest-point (Gauss-Newton) encoder, which projects onto
    the manifold rather than applying a linear map, and so must
    differentiate the features. Every implementation in this module
    satisfies it; the split exists so that consumers needing only
    evaluation cannot silently acquire a dependence on the derivative.
    """

    def jacobian(self, codes: Array) -> Array:
        """Jacobian of the feature map at each coordinate.

        Parameters
        ----------
        codes : Array
            Shape: (r, k). Reduced coordinates.

        Returns
        -------
        Array
            Shape: (p, r, k). ``J[a, b, i] = d h_a / d z_b`` at column ``i``.
        """
        ...


class _MultiIndexFeatureMap(Generic[Array]):
    """Base for feature maps defined by an explicit multi-index set.

    Evaluation and the analytic Jacobian depend only on the multi-indices,
    not on how they were constructed, so both the degree-band
    :class:`MonomialFeatureMap` and the arbitrary-index-set
    :class:`SparseMonomialFeatureMap` share this implementation.

    The index matrix has shape ``(nreduced, p)``, the same convention
    ``compute_hyperbolic_level_indices`` returns; column ``a`` defines the
    monomial :math:`h_a(z) = \\prod_b z_b^{i_{ba}}`.
    """

    def __init__(
        self, nreduced: int, indices: Array, bkd: Backend[Array]
    ) -> None:
        self._nreduced = nreduced
        self._bkd = bkd
        self._indices = indices
        # The indices are integer structure, not data: they select which
        # powers to multiply, and are read once per element in the evaluation
        # loops below. Converting the whole matrix here keeps those loops free
        # of per-element device-to-host transfers, which under torch would
        # otherwise force a synchronization for every index read.
        self._indices_int: npt.NDArray[np.int64] = np.asarray(
            bkd.to_numpy(indices)
        ).astype(np.int64)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nreduced(self) -> int:
        """Reduced dimension ``r`` the map expects as input."""
        return self._nreduced

    def indices(self) -> Array:
        """Multi-index matrix, shape (r, p)."""
        return self._indices

    def nterms(self) -> int:
        return int(self._indices.shape[1])

    def __call__(self, codes: Array) -> Array:
        bkd = self._bkd
        r, k = codes.shape
        indices = self._indices_int
        if self.nterms() == 0:
            # An empty index set is a legitimate map with no correction
            # terms, not an error: stacking nothing has no shape, so the
            # empty result is built directly.
            return bkd.zeros((0, k))
        terms: List[Array] = []
        for a in range(self.nterms()):
            term = bkd.full((k,), 1.0)
            for b in range(r):
                power = int(indices[b, a])
                if power == 0:
                    continue
                term = term * (codes[b, :] ** power)
            terms.append(term)
        return bkd.stack(terms, axis=0)

    def jacobian(self, codes: Array) -> Array:
        bkd = self._bkd
        r, k = codes.shape
        indices = self._indices_int
        if self.nterms() == 0:
            return bkd.zeros((0, r, k))
        zero = bkd.full((k,), 0.0)
        rows: List[Array] = []
        for a in range(self.nterms()):
            cols: List[Array] = []
            for d in range(r):
                index_d = int(indices[d, a])
                if index_d == 0:
                    # h_a does not involve z_d, so the derivative vanishes.
                    cols.append(zero)
                    continue
                deriv = bkd.full((k,), float(index_d))
                for b in range(r):
                    index_b = int(indices[b, a])
                    power = index_b - 1 if b == d else index_b
                    if power == 0:
                        continue
                    deriv = deriv * (codes[b, :] ** power)
                cols.append(deriv)
            rows.append(bkd.stack(cols, axis=0))
        return bkd.stack(rows, axis=0)


class MonomialFeatureMap(_MultiIndexFeatureMap[Array], Generic[Array]):
    """Monomials of the requested total degrees.

    Uses ``compute_hyperbolic_level_indices`` (with pnorm=1.0, the level
    equals the total degree) to enumerate exactly the monomials of each
    requested degree. ``degrees=(0, 1, 2)`` -- the default -- gives every
    monomial up to quadratic; adding 3 extends it to cubic.

    **Include degrees 0 and 1 unless there is a specific reason not to.**
    The tempting argument for a bare ``degrees=(2,)`` band is that a
    manifold decoder ``mu + V z + W h(z)`` already carries a constant in
    ``mu`` and a linear term in ``V z``. That argument does not survive
    contact with what ``W`` is fitted to: the residual ``s - P_V s``,
    which lies in the orthogonal complement of the basis. ``mu`` was
    subtracted before that projection and ``V z`` spans the complement's
    orthogonal, so neither reaches the constant and linear parts of the
    residual, and a degree-2 band leaves them unrepresented. Centering
    guarantees a constant part is present, since the sample mean removes
    ``E[h(z)]`` from the data but not from the features. Measured on
    snapshots built to the decoder's own form, latent dimension 2:
    a degree-2 band recovers 2.1x the linear reconstruction error, adding
    degree 0 recovers 7.5x, and the full downward-closed set 167x.

    A pure degree band is also not downward closed for more than one
    variable -- it holds ``(1, 0, 1)`` but not ``(0, 0, 1)`` -- which puts
    it outside the class of index sets the rest of the library is built
    around. It remains available because the band is what the source
    paper's Eq. (5) states, and because restricting it to fewer variables
    is still well defined.
    """

    def __init__(
        self,
        nreduced: int,
        bkd: Backend[Array],
        degrees: Sequence[int] = (0, 1, 2),
    ) -> None:
        """
        Parameters
        ----------
        nreduced : int
            Reduced dimension ``r``.
        bkd : Backend
            Computational backend.
        degrees : sequence of int
            Total degrees of monomials to include. Defaults to every
            degree up to quadratic, which is downward closed. Omitting
            the low degrees is allowed but costs accuracy; see the class
            docstring.
        """
        checked_degrees = tuple(int(d) for d in degrees)
        if any(d < 0 for d in checked_degrees):
            raise ValueError(
                f"feature-map degrees must be non-negative, got "
                f"{checked_degrees}."
            )
        if len(set(checked_degrees)) != len(checked_degrees):
            raise ValueError(
                f"feature-map degrees must be distinct, got "
                f"{checked_degrees}; a repeated degree would enumerate "
                "the same monomials twice."
            )
        self._degrees = checked_degrees
        indices = self._build_indices(nreduced, checked_degrees, bkd)
        super().__init__(nreduced, indices, bkd)

    @staticmethod
    def _build_indices(
        nreduced: int, degrees: Tuple[int, ...], bkd: Backend[Array]
    ) -> Array:
        """Multi-indices for the selected degree band."""
        per_degree = [
            compute_hyperbolic_level_indices(nreduced, d, 1.0, bkd)
            for d in degrees
        ]
        return bkd.hstack(per_degree)

    def degrees(self) -> Tuple[int, ...]:
        """Total degrees of monomials included in this map."""
        return self._degrees


class SparseMonomialFeatureMap(_MultiIndexFeatureMap[Array], Generic[Array]):
    """Monomial features over an arbitrary, possibly anisotropic index set.

    Built directly from a multi-index set rather than a degree band, so the
    adaptive manifold can hold whatever downward-closed index set its
    refinement has grown -- e.g. cubic in one direction, linear in another, a
    single cross-term.  Only degree-``>=2`` columns belong here; the constant
    and linear (degree-0/1) terms are represented by the pinned linear part
    of the decoder.
    """

    def __init__(self, indices: Array, bkd: Backend[Array]) -> None:
        """
        Parameters
        ----------
        indices : Array
            Shape: (nreduced, p). Integer multi-index columns, each of total
            degree >= 2.
        bkd : Backend
            Computational backend.
        """
        nreduced = int(indices.shape[0])
        super().__init__(nreduced, indices, bkd)

    @classmethod
    def from_index_set(
        cls, indices: Array, bkd: Backend[Array]
    ) -> Tuple[List[int], "SparseMonomialFeatureMap[Array]"]:
        """Split a full index set into linear support and a correction map.

        Parameters
        ----------
        indices : Array
            Shape: (nreduced, nindices). A downward-closed multi-index set
            (e.g. the selected indices of an ``IterativeIndexGenerator``),
            including the constant (degree 0) and linear (degree 1) columns.
        bkd : Backend
            Computational backend.

        Returns
        -------
        active_dims : list of int
            Dimensions ``d`` that appear with nonzero degree in any index --
            the active subspace.  Determines which singular vectors form V.
        feature_map : SparseMonomialFeatureMap
            Correction features built from the degree-``>=2`` columns,
            expressed over the active dimensions (rows reindexed to
            ``active_dims``).
        """
        index_matrix = np.asarray(bkd.to_numpy(indices)).astype(int)
        total_degree = index_matrix.sum(axis=0)
        active_dims = [
            d
            for d in range(index_matrix.shape[0])
            if int(index_matrix[d, :].max()) > 0
        ]
        corr_cols = [
            c
            for c in range(index_matrix.shape[1])
            if total_degree[c] >= 2
        ]
        sub = index_matrix[active_dims, :][:, corr_cols]
        return active_dims, cls(bkd.asarray(sub), bkd)


def build_feature_map(
    nreduced: int, bkd: Backend[Array], degrees: Sequence[int] = (0, 1, 2)
) -> MonomialFeatureMap[Array]:
    """Construct a monomial feature map.

    Convenience wrapper mirroring the ``build_*`` factory style used
    elsewhere in the package.
    """
    return MonomialFeatureMap(nreduced, bkd, degrees=degrees)
