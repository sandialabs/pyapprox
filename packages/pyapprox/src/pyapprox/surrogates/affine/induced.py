r"""Sampling for stable weighted least squares.

Induced (Christoffel) sampling draws points where a basis has mass,
keeping the weighted Gram matrix close to the identity as the basis
grows. It applies to any orthonormal basis over a product measure —
polynomial chaos as much as operator learning — so it lives here beside
the bases and measures it needs rather than with any one consumer.

A sampler returns samples and their weights **together**. The weight
:math:`w = d\rho/d\mu` is a property of how a sample was drawn, so a
sampler returning samples alone would leave the caller to supply a
weight only it knows — and supplying the reciprocal by mistake still
fits noiseless in-span data perfectly, so the error survives testing.
Returning the pair makes the mistake unrepresentable.

Randomness comes from global numpy seeding, matching the rest of the
library: the marginals and joint distributions take no generator, so a
Monte Carlo sampler that accepted one could not honor it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Tuple

import numpy as np

from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.surrogates.affine.protocols.multivariate_basis import (
    EvaluableMultiIndexBasisProtocol,
    SampleableMultiIndexBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


def christoffel_function(
    basis: EvaluableMultiIndexBasisProtocol[Array],
    samples: Array,
    bkd: Backend[Array],
) -> Array:
    r"""Return the normalized Christoffel function of a basis.

    .. math:: k_\Lambda(x) = \frac{1}{N} \sum_\lambda p_\lambda(x)^2

    Normalized by the number of terms, so it integrates to one against
    the measure the basis is orthonormal under and is therefore a
    density. Its reciprocal is the least-squares weight for samples
    drawn from the induced measure.

    Parameters
    ----------
    basis : EvaluableMultiIndexBasisProtocol[Array]
        An orthonormal basis.
    samples : Array
        Points to evaluate at. Shape: (nvars, nsamples)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Christoffel values. Shape: (nsamples,)
    """
    return bkd.sum(basis(samples) ** 2, axis=1) / basis.nterms()


@dataclass(frozen=True)
class WeightedSample(Generic[Array]):
    """Coefficients drawn from a sampling measure, with their weights.

    A frozen value object holding the two together, because a weight
    only means anything alongside the sample it was computed for.

    Attributes
    ----------
    coefs : Array
        Input coefficients. Shape: (nvars, nsamples)
    weights : Array
        Least-squares weights :math:`w_i`. Shape: (nsamples,)
    """

    coefs: Array
    weights: Array


class MonteCarloSampler(Generic[Array]):
    r"""Draw coefficients from the reference measure itself.

    The sampling measure equals the reference measure, so
    :math:`d\rho/d\mu \equiv 1` and the weights are one. Unbiased, and
    the natural baseline, but the Gram matrix degrades as the basis
    grows — which is the comparison :class:`InducedSampler` exists to
    win.

    Parameters
    ----------
    rho : IndependentJoint[Array]
        The reference measure on input coefficients.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, rho: IndependentJoint[Array], bkd: Backend[Array]):
        if not isinstance(rho, IndependentJoint):
            raise TypeError(
                f"rho must be an IndependentJoint, got {type(rho).__name__}"
            )
        self._rho = rho
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def rho(self) -> IndependentJoint[Array]:
        """Return the reference measure."""
        return self._rho

    def __call__(self, nsamples: int) -> WeightedSample[Array]:
        """Draw coefficients with unit weights.

        Parameters
        ----------
        nsamples : int
            Number of samples to draw.

        Returns
        -------
        WeightedSample
            Coefficients (nvars, nsamples) and unit weights (nsamples,).
        """
        if nsamples < 1:
            raise ValueError(f"nsamples must be positive, got {nsamples}")
        return WeightedSample(
            self._rho.rvs(nsamples), self._bkd.ones((nsamples,))
        )


class InducedSampler(Generic[Array]):
    r"""Draw coefficients from the induced measure of a basis.

    The induced measure is the mixture

    .. math:: \mu = \frac{1}{N_{\mathrm{eff}}}
                    \sum_\lambda p_\lambda^2 \, d\rho
                  = k_\Lambda \, d\rho

    sampled by choosing :math:`\lambda` uniformly from :math:`\Lambda`
    and then drawing from :math:`p_\lambda^2 d\rho`. Because the basis
    is a tensor product and :math:`\rho` is a product measure, that
    second step factors into independent one-dimensional draws from
    :math:`(p^j_{\lambda_j})^2 d\rho_j`, each done by inverting a
    discrete CDF built on a Gauss rule.

    The resulting weight is :math:`w = d\rho/d\mu = 1/k_\Lambda`, the
    reciprocal of the Christoffel function. Sampling this way keeps the
    weighted Gram matrix close to the identity as the basis grows,
    where Monte Carlo sampling does not.

    The draw refers only to the basis and the reference measure. For
    operator learning that is what makes the sample count independent
    of the output dimension: the output space never enters.

    Parameters
    ----------
    basis : SampleableMultiIndexBasisProtocol[Array]
        An orthonormal basis, supplying :math:`\Lambda` and the
        univariate quadrature rules the conditionals are built on.
    rho : IndependentJoint[Array]
        The reference measure, whose marginals the basis is orthonormal
        against.
    bkd : Backend[Array]
        Computational backend.
    nquad : int
        Points in the one-dimensional Gauss rule used to discretize
        each conditional. The draw is exact only in the limit of large
        ``nquad``; too few points restricts samples to a coarse grid.
    """

    def __init__(
        self,
        basis: SampleableMultiIndexBasisProtocol[Array],
        rho: IndependentJoint[Array],
        bkd: Backend[Array],
        nquad: int = 200,
    ):
        if not isinstance(basis, SampleableMultiIndexBasisProtocol):
            raise TypeError(
                f"basis must satisfy SampleableMultiIndexBasisProtocol, "
                f"got {type(basis).__name__}"
            )
        if not isinstance(rho, IndependentJoint):
            raise TypeError(
                f"rho must be an IndependentJoint, got {type(rho).__name__}"
            )
        if nquad < 2:
            raise ValueError(f"nquad must be at least two, got {nquad}")
        self._basis = basis
        self._rho = rho
        self._bkd = bkd
        self._nquad = nquad
        self._cdfs, self._nodes = self._build_conditionals()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> SampleableMultiIndexBasisProtocol[Array]:
        """Return the basis whose induced measure is sampled."""
        return self._basis

    def rho(self) -> IndependentJoint[Array]:
        """Return the reference measure."""
        return self._rho

    def nquad(self) -> int:
        """Return the number of quadrature points per conditional."""
        return self._nquad

    def _build_conditionals(self) -> Tuple[List[Array], List[Array]]:
        r"""Tabulate the CDF of :math:`(p^j_k)^2 d\rho_j` per dimension.

        Returns cumulative sums of :math:`w_q p_k(x_q)^2` over the
        Gauss nodes, one matrix of shape (nquad, ndegrees) per
        dimension, alongside the nodes themselves. Built once at
        construction because the index set may change but the
        univariate conditionals do not.
        """
        scalar_basis = self._basis
        nvars = scalar_basis.nvars()
        cdfs, nodes = [], []
        for dim in range(nvars):
            points, weights = scalar_basis.univariate_quadrature(
                dim, self._nquad
            )
            basis_1d = scalar_basis.get_univariate_basis(dim)
            values = basis_1d(points)
            # univariate_quadrature returns weights as (npoints, 1).
            flat_weights = self._bkd.reshape(weights, (-1, 1))
            # Column k is the CDF of the kth squared basis function,
            # which is a density because the basis is orthonormal.
            cdfs.append(self._bkd.cumsum(flat_weights * values**2, axis=0))
            nodes.append(self._bkd.flatten(points))
        return cdfs, nodes

    def _sample_dimension(self, dim: int, degrees: Array) -> Array:
        """Draw one coordinate per requested degree by inverting the CDF.

        Parameters
        ----------
        dim : int
            Dimension index.
        degrees : Array
            Degree of the conditional to draw from, one per sample.
            Shape: (nsamples,)

        Returns
        -------
        Array
            Sampled coordinates. Shape: (nsamples,)
        """
        np_cdfs = self._bkd.to_numpy(self._cdfs[dim])
        np_degrees = self._bkd.to_numpy(degrees).astype(int)
        nsamples = np_degrees.shape[0]
        uniforms = np.random.uniform(0.0, 1.0, (nsamples,))

        # searchsorted per sample: each row is the CDF of its own degree
        selected = np_cdfs[:, np_degrees].T
        indices = np.array(
            [
                np.searchsorted(row, u, side="right")
                for row, u in zip(selected, uniforms)
            ]
        )
        # A uniform above the final CDF value, reachable through
        # quadrature error, would index past the last node.
        indices = np.minimum(indices, np_cdfs.shape[0] - 1)
        np_nodes = self._bkd.to_numpy(self._nodes[dim])
        return self._bkd.asarray(np_nodes[indices])

    def __call__(self, nsamples: int) -> WeightedSample[Array]:
        r"""Draw coefficients from the induced measure with their weights.

        Parameters
        ----------
        nsamples : int
            Number of samples to draw.

        Returns
        -------
        WeightedSample
            Coefficients (nvars, nsamples) and weights
            :math:`1/k_\Lambda` (nsamples,).
        """
        if nsamples < 1:
            raise ValueError(f"nsamples must be positive, got {nsamples}")
        indices = self._basis.get_indices()
        nterms = self._basis.nterms()

        # Choose lambda uniformly from Lambda, then draw each coordinate
        # from the corresponding univariate conditional.
        chosen = np.random.choice(nterms, size=nsamples, replace=True)
        selected_indices = indices[
            :, self._bkd.asarray(chosen, dtype=self._bkd.int64_dtype())
        ]
        coords = [
            self._sample_dimension(dim, selected_indices[dim])
            for dim in range(indices.shape[0])
        ]
        coefs = self._bkd.stack(coords, axis=0)
        christoffel = christoffel_function(self._basis, coefs, self._bkd)
        return WeightedSample(coefs, 1.0 / christoffel)
