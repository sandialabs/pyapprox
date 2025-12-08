"""Orthonormal polynomial basis for multivariate approximations.

This module provides OrthonormalPolynomialBasis, which extends MultiIndexBasis
for orthonormal polynomial bases with quadrature support.
"""

from typing import Generic, List, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    OrthonormalPolynomial1DProtocol,
)
from pyapprox.typing.surrogates.affine.basis.multiindex import MultiIndexBasis


class OrthonormalPolynomialBasis(MultiIndexBasis[Array], Generic[Array]):
    """Multivariate orthonormal polynomial basis.

    Extends MultiIndexBasis for orthonormal polynomial bases, providing
    access to univariate quadrature rules and tensor product quadrature.

    Parameters
    ----------
    bases_1d : List[OrthonormalPolynomial1DProtocol[Array]]
        List of orthonormal polynomial bases, one per variable.
    bkd : Backend[Array]
        Computational backend.
    indices : Array, optional
        Multi-indices specifying which basis functions to include.
        Shape: (nvars, nterms). If None, must be set later.
    """

    def __init__(
        self,
        bases_1d: List[OrthonormalPolynomial1DProtocol[Array]],
        bkd: Backend[Array],
        indices: Optional[Array] = None,
    ):
        # Verify all bases are orthonormal polynomials
        for ii, basis in enumerate(bases_1d):
            if not isinstance(basis, OrthonormalPolynomial1DProtocol):
                raise TypeError(
                    f"bases_1d[{ii}] must be OrthonormalPolynomial1DProtocol, "
                    f"got {type(basis).__name__}"
                )

        super().__init__(bases_1d, bkd, indices)

    def univariate_quadrature(
        self, dim: int, npoints: Optional[int] = None
    ) -> Tuple[Array, Array]:
        """Return univariate Gauss quadrature rule for a dimension.

        Parameters
        ----------
        dim : int
            Dimension index.
        npoints : int, optional
            Number of quadrature points. If None, uses current nterms.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints,)
        """
        basis = self._bases_1d[dim]
        if npoints is None:
            npoints = basis.nterms()
        # Ensure basis has enough terms for requested quadrature points
        if basis.nterms() < npoints:
            basis.set_nterms(npoints)
        return basis.gauss_quadrature_rule(npoints)

    def tensor_product_quadrature(
        self, npoints_1d: Optional[List[int]] = None
    ) -> Tuple[Array, Array]:
        """Return tensor product quadrature rule.

        Parameters
        ----------
        npoints_1d : List[int], optional
            Number of points per dimension. If None, uses nterms per basis.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (nvars, npoints_total)
        weights : Array
            Quadrature weights. Shape: (npoints_total,)
        """
        if npoints_1d is None:
            npoints_1d = [self._bases_1d[dd].nterms() for dd in range(self.nvars())]

        # Get univariate rules
        points_1d = []
        weights_1d = []
        for dd in range(self.nvars()):
            pts, wts = self.univariate_quadrature(dd, npoints_1d[dd])
            points_1d.append(pts[0, :])  # Remove first dim
            weights_1d.append(self._bkd.flatten(wts))  # Flatten weights

        # Build tensor product
        grids = self._bkd.meshgrid(tuple(points_1d), indexing="ij")
        points = self._bkd.stack(
            [self._bkd.flatten(g) for g in grids], axis=0
        )

        # Tensor product of weights
        weights = weights_1d[0]
        for dd in range(1, self.nvars()):
            weights = self._bkd.flatten(
                self._bkd.reshape(weights, (-1, 1)) * weights_1d[dd]
            )

        return points, weights

    def __repr__(self) -> str:
        return (
            f"OrthonormalPolynomialBasis(nvars={self.nvars()}, "
            f"nterms={self.nterms()})"
        )
