r"""The operator basis :math:`V = Y_h \otimes P`.

Never formed explicitly. It exists to own one identity: for a separable
basis whose elements are :math:`p_\lambda(\hat f)\psi_j`,

.. math::

    \sum_{j,\lambda} \|p_\lambda(\hat f)\psi_j\|_Y^2
        = d_{\mathrm{out}} \sum_\lambda p_\lambda(\hat f)^2,
    \qquad N = N_{\mathrm{eff}} \, d_{\mathrm{out}}

so the sampling weight

.. math::

    w = \frac{N}{\sum \|\Phi\|^2} = \frac{1}{k_\Lambda(\hat f)},
    \qquad k_\Lambda := \frac{1}{N_{\mathrm{eff}}}
                        \sum_\lambda p_\lambda(\hat f)^2

is independent of :math:`d_{\mathrm{out}}` — it cancels exactly. That is
why the sample count depends only on :math:`N_{\mathrm{eff}}`, and it is
the reason this wrapper exists rather than the scalar basis being used
directly.
"""

from __future__ import annotations

from typing import Generic

from pyapprox.surrogates.affine.protocols.multivariate_basis import (
    EvaluableMultiIndexBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class SeparableOperatorBasis(Generic[Array]):
    r"""Operator basis formed from a scalar basis and an output space.

    Separable means every element factors as
    :math:`p_\lambda(\hat f)\,\psi_j`, with the same scalar basis
    :math:`P` used for each of the :math:`d_{\mathrm{out}}` output
    coefficients. Separability is what makes induced sampling possible:
    a sample can be drawn for :math:`P` alone, without reference to the
    output space.

    Parameters
    ----------
    scalar_basis : EvaluableMultiIndexBasisProtocol[Array]
        The basis :math:`P` over input coefficients, carrying the index
        set :math:`\Lambda`.
    noutputs : int
        The number of output coefficients :math:`d_{\mathrm{out}}`.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        scalar_basis: EvaluableMultiIndexBasisProtocol[Array],
        noutputs: int,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(scalar_basis, EvaluableMultiIndexBasisProtocol):
            raise TypeError(
                f"scalar_basis must satisfy EvaluableMultiIndexBasisProtocol, got "
                f"{type(scalar_basis).__name__}"
            )
        if noutputs < 1:
            raise ValueError(f"noutputs must be positive, got {noutputs}")
        self._scalar_basis = scalar_basis
        self._noutputs = noutputs
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def scalar_basis(self) -> EvaluableMultiIndexBasisProtocol[Array]:
        """Return the scalar basis over input coefficients."""
        return self._scalar_basis

    def noutputs(self) -> int:
        """Return the number of output coefficients d_out."""
        return self._noutputs

    def neffective(self) -> int:
        r"""Return :math:`N_{\mathrm{eff}}`, the scalar basis size.

        The sample count is governed by this rather than by the full
        basis size :math:`N = N_{\mathrm{eff}} d_{\mathrm{out}}`,
        because :math:`d_{\mathrm{out}}` cancels from the sampling
        weight.
        """
        return int(self._scalar_basis.nterms())

    def nterms(self) -> int:
        r"""Return the full basis size :math:`N_{\mathrm{eff}} d_{out}`.

        Note this *multiplies*: the operator basis has one element per
        (index, output coefficient) pair. It is reported for
        completeness; the quantity governing sample complexity is
        :meth:`neffective`.
        """
        return self.neffective() * self._noutputs

    def get_indices(self) -> Array:
        """Return the index set. Shape: (nvars, neffective)."""
        return self._scalar_basis.get_indices()

    def set_indices(self, indices: Array) -> None:
        """Replace the index set.

        Delegates to the scalar basis, so an adaptive fitter can grow
        :math:`\\Lambda` between refits without rebuilding the encoders,
        the reference measure, or the samples.
        """
        self._scalar_basis.set_indices(indices)

    def design_matrix(self, coefs: Array) -> Array:
        r"""Return the scalar design matrix :math:`A`.

        Parameters
        ----------
        coefs : Array
            Encoded input coefficients. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            :math:`A_{i\lambda} = p_\lambda(\hat f^i)`.
            Shape: (nsamples, neffective)

        Notes
        -----
        The full operator design matrix is
        :math:`A \otimes I_{d_{\mathrm{out}}}`, which is never formed:
        with the same :math:`A` for every output coefficient, the least
        squares problem is one solve with :math:`d_{\mathrm{out}}`
        right-hand sides.
        """
        return self._scalar_basis(coefs)

    def christoffel(self, coefs: Array) -> Array:
        r"""Return the normalized Christoffel function :math:`k_\Lambda`.

        .. math::

            k_\Lambda(\hat f)
                = \frac{1}{N_{\mathrm{eff}}}
                  \sum_\lambda p_\lambda(\hat f)^2

        Independent of :math:`d_{\mathrm{out}}`: the output space
        contributes a factor :math:`d_{\mathrm{out}}` to both the sum of
        squared basis norms and to :math:`N`, and the two cancel.

        The reciprocal is the sampling weight, so a caller reaching for
        this to weight a least squares fit wants ``1 / christoffel``.
        :class:`InducedSampler` returns the weight with its samples so
        the reciprocal cannot be dropped by accident.

        Parameters
        ----------
        coefs : Array
            Encoded input coefficients. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Christoffel values. Shape: (nsamples,)
        """
        basis_values = self.design_matrix(coefs)
        return self._bkd.sum(basis_values**2, axis=1) / self.neffective()

    def apply(self, params: Array, coefs: Array) -> Array:
        r"""Evaluate the operator surrogate at encoded inputs.

        Parameters
        ----------
        params : Array
            Coefficients :math:`C`. Shape: (neffective, noutputs)
        coefs : Array
            Encoded input coefficients. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Predicted output coefficients. Shape: (noutputs, nsamples)
        """
        if params.shape != (self.neffective(), self._noutputs):
            raise ValueError(
                f"params has wrong shape {params.shape}, expected "
                f"({self.neffective()}, {self._noutputs})"
            )
        return self._bkd.dot(self.design_matrix(coefs), params).T

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(neffective={self.neffective()}, "
            f"noutputs={self._noutputs})"
        )
