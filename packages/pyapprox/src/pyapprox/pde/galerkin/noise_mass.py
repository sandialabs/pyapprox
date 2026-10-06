r"""White-noise mass weightings for Gaussian random fields on a FE basis.

Discretized white noise on a finite element space has covariance equal to
the mass matrix ``M``. Sampling it needs a factor ``B`` with
:math:`B B^\top = M`: if :math:`\xi \sim N(0, I)` then
:math:`B \xi \sim N(0, M)`. ``B`` need not be square, triangular or
invertible, so it can be built without factorizing ``M`` globally.

A consumer, such as ``BiLaplacianPrior``, depends only on
``NoiseMassProtocol``; a new weighting is added by implementing it and
injecting the instance.

Do not use ``diag(M)`` as a mass: on P1 triangles the element mass is
:math:`\frac{|T|}{12}\begin{pmatrix}2&1&1\\1&2&1\\1&1&2\end{pmatrix}`, so
``diag(M)`` is exactly half the row-sum lumped mass and would halve the
variance.
"""

from typing import Any, Generic, Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp

from pyapprox.pde.galerkin.physics.helpers import ScalarMassAssembler
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.probability.protocols.covariance import (
    CovarianceOperatorProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class NoiseMassProtocol(Protocol, Generic[Array]):
    """A mass matrix ``M`` and a factor ``B`` with ``B B^T = M``."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nnoise(self) -> int:
        """Return the length of the white-noise vector ``B`` acts on."""
        ...

    def apply_factor(self, noise: Array) -> Array:
        """Return ``B noise``.

        Parameters
        ----------
        noise : Array
            Shape: ``(nnoise, ncols)``.

        Returns
        -------
        Array
            Shape: ``(ndofs, ncols)``.
        """
        ...

    def mass_matrix(self) -> Any:
        """Return ``M``, shape ``(ndofs, ndofs)``; sparse when assembled."""
        ...


def _check_noise(noise: Array, nnoise: int) -> None:
    if noise.ndim != 2 or noise.shape[0] != nnoise:
        raise ValueError(
            f"noise must have shape ({nnoise}, ncols), got {tuple(noise.shape)}"
        )


class LumpedNoiseMass(Generic[Array]):
    r"""Row-sum lumped mass: ``M_L = diag(M 1)``, ``B = M_L^{1/2}``.

    Square and diagonal, so ``nnoise`` equals the number of DOFs.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Scalar finite element basis.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self, basis: GalerkinBasisProtocol[Array], bkd: Backend[Array]
    ) -> None:
        consistent = ScalarMassAssembler(basis, bkd).mass_matrix()
        self._lumped = np.asarray(
            consistent @ np.ones(basis.ndofs()), dtype=np.float64
        ).ravel()
        self._sqrt_lumped = np.sqrt(self._lumped)
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nnoise(self) -> int:
        return int(self._lumped.shape[0])

    def apply_factor(self, noise: Array) -> Array:
        _check_noise(noise, self.nnoise())
        weighted = self._sqrt_lumped[:, None] * self._bkd.to_numpy(noise)
        return self._bkd.asarray(weighted)

    def mass_matrix(self) -> Any:
        return sp.diags(self._lumped, format="csr")

    def __repr__(self) -> str:
        return f"LumpedNoiseMass(ndofs={self.nnoise()})"


class ConsistentNoiseMass(Generic[Array]):
    r"""Consistent mass with an element-assembled sparse factor.

    The mass is a sum of element contributions,
    :math:`M = \sum_e P_e^\top M_e P_e`, where :math:`P_e` gathers element
    ``e``'s DOFs. Factoring each small local mass,
    :math:`M_e = L_e L_e^\top`, gives

    .. math::

        M = B B^\top, \qquad B = [P_1^\top L_1 \,|\, \dots \,|\,
        P_E^\top L_E],

    with no global factorization: ``B`` is sparse and costs ``O(E)``. Each
    element draws its own local noise, and DOFs shared by elements sum
    their contributions, which produces ``M``'s correlations.

    ``nnoise`` is the number of local element DOFs, ``E`` times the
    element's DOF count: about ``3E`` (roughly ``6 N``) for P1 triangles,
    so the factor is not square. When a square factor with
    ``nnoise = N`` matters, use
    ``CovarianceOperatorNoiseMass(DenseCholeskyCovarianceOperator(M))``,
    at the cost of a dense Cholesky of ``M``.

    Each local mass must be positive definite, which holds when the
    element quadrature integrates products of basis functions exactly
    (skfem's default for Lagrange elements).

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Scalar finite element basis.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self, basis: GalerkinBasisProtocol[Array], bkd: Backend[Array]
    ) -> None:
        skfem_basis = basis.skfem_basis()
        nloc = int(skfem_basis.Nbfun)
        nelem = int(skfem_basis.nelems)
        values = np.array(
            [skfem_basis.basis[ii][0].value for ii in range(nloc)]
        )
        local_mass = np.einsum(
            "iEq,jEq,Eq->Eij", values, values, skfem_basis.dx
        )
        local_factor = np.linalg.cholesky(local_mass)
        # Keep only the lower triangle of each local factor.
        row_idx, col_idx = np.tril_indices(nloc)
        element_dofs = np.asarray(skfem_basis.element_dofs)
        rows = element_dofs[row_idx, :].T
        cols = np.arange(nelem)[:, None] * nloc + col_idx[None, :]
        data = local_factor[:, row_idx, col_idx]
        self._factor = sp.csr_matrix(
            (data.ravel(), (rows.ravel(), cols.ravel())),
            shape=(int(skfem_basis.N), nelem * nloc),
        )
        self._mass = ScalarMassAssembler(basis, bkd).mass_matrix()
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nnoise(self) -> int:
        return int(self._factor.shape[1])

    def apply_factor(self, noise: Array) -> Array:
        _check_noise(noise, self.nnoise())
        weighted = self._factor @ self._bkd.to_numpy(noise)
        return self._bkd.asarray(np.asarray(weighted))

    def mass_matrix(self) -> Any:
        return self._mass

    def __repr__(self) -> str:
        return (
            f"ConsistentNoiseMass(ndofs={self._factor.shape[0]}, "
            f"nnoise={self.nnoise()})"
        )


class CovarianceOperatorNoiseMass(Generic[Array]):
    """Any covariance operator as a noise mass: ``M = covariance()``,
    ``B = L`` from ``apply``.

    Its factor is square, so ``nnoise`` equals the number of DOFs.
    ``CovarianceOperatorNoiseMass(DenseCholeskyCovarianceOperator(M))``
    gives the consistent mass with a dense Cholesky factor.

    Parameters
    ----------
    operator : CovarianceOperatorProtocol
        Covariance operator whose covariance is the mass.

    Raises
    ------
    TypeError
        If ``operator`` does not satisfy ``CovarianceOperatorProtocol``.
    """

    def __init__(self, operator: CovarianceOperatorProtocol[Array]) -> None:
        if not isinstance(operator, CovarianceOperatorProtocol):
            raise TypeError(
                "operator must satisfy CovarianceOperatorProtocol, got "
                f"{type(operator).__name__}"
            )
        self._operator = operator

    def bkd(self) -> Backend[Array]:
        return self._operator.bkd()

    def nnoise(self) -> int:
        return self._operator.nvars()

    def apply_factor(self, noise: Array) -> Array:
        _check_noise(noise, self.nnoise())
        return self._operator.apply(noise)

    def mass_matrix(self) -> Any:
        return self._operator.covariance()

    def __repr__(self) -> str:
        return f"CovarianceOperatorNoiseMass({self._operator!r})"
