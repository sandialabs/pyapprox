"""Sparse grid to PCE converter.

This module provides converters to transform sparse grid interpolants into
Polynomial Chaos Expansions (PCE) using spectral projection.

The conversion works by:
1. Converting each tensor product subspace's Lagrange interpolant to PCE
2. Combining subspace PCEs using Smolyak coefficients

Important
---------
The ``orthonormal_bases_1d`` parameter must be created using
``create_bases_1d(marginals, bkd)`` from
``pyapprox.surrogates.affine.univariate``. This returns
``TransformedBasis1D`` objects that handle domain transforms correctly,
ensuring quadrature points are in physical domain to match the sparse
grid's Lagrange nodes.

Do NOT pass raw ``OrthonormalPolynomial1D`` objects (e.g., ``LegendrePolynomial1D``)
directly, as their ``gauss_quadrature_rule()`` returns canonical domain points
which will cause incorrect spectral projection for non-canonical domains.
"""

from typing import Dict, Generic, List, Optional, Sequence, Tuple

from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import (
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.affine.protocols import (
    PhysicalDomainBasis1DProtocol,
)
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array, Backend


class TensorProductSubspaceToPCEConverter(Generic[Array]):
    """Convert a tensor product subspace to PCE coefficients.

    Uses spectral projection to convert Lagrange interpolant basis functions
    to orthonormal polynomial coefficients.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    orthonormal_bases_1d : Sequence[PhysicalDomainBasis1DProtocol[Array]]
        Univariate bases for each dimension, created via ``create_bases_1d()``.
        These define the target PCE basis and must return physical-domain
        quadrature points from ``gauss_quadrature_rule()``.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability import UniformMarginal
    >>> from pyapprox.surrogates.affine.univariate import create_bases_1d
    >>> bkd = NumpyBkd()
    >>> marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
    >>> bases_1d = create_bases_1d(marginals, bkd)
    >>> converter = TensorProductSubspaceToPCEConverter(bkd, bases_1d)

    See Also
    --------
    create_bases_1d : Factory function to create physical-domain bases.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        orthonormal_bases_1d: Sequence[PhysicalDomainBasis1DProtocol[Array]],
    ):
        self._bkd = bkd
        self._orthonormal_bases_1d = list(orthonormal_bases_1d)
        self._nvars = len(orthonormal_bases_1d)

        # Cache for computed projection coefficients
        # Key: (dim, tuple of node values), Value: projection coefficients array
        self._cached_projection_coefs: Dict[Tuple[int, Tuple[float, ...]], Array] = {}

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def _compute_univariate_projection_coefficients(
        self,
        dim: int,
        lagrange_nodes: Array,
    ) -> Array:
        """Compute coefficients to project 1D Lagrange basis to orthonormal poly.

        For each Lagrange basis function L_j(x) defined at nodes, compute
        the coefficients c_jk such that L_j(x) ≈ sum_k c_jk * P_k(x)
        where P_k are orthonormal polynomials.

        Parameters
        ----------
        dim : int
            Dimension index.
        lagrange_nodes : Array
            Nodes defining Lagrange basis, shape (npts,)

        Returns
        -------
        Array
            Projection coefficients, shape (npts, npts)
            Entry [j, k] is coefficient of P_k in expansion of L_j.
        """
        npts = lagrange_nodes.shape[0]

        # Get Gauss quadrature for spectral projection
        # Use enough points for exact integration of polynomial products
        ortho_basis = self._orthonormal_bases_1d[dim]
        ortho_basis.set_nterms(npts + 1)
        quad_pts, quad_wts = ortho_basis.gauss_quadrature_rule(npts + 1)

        # Evaluate orthonormal polynomials at quadrature points
        ortho_basis.set_nterms(npts)
        ortho_vals = ortho_basis(quad_pts)[:, :npts]  # (nquad, npts)

        # Evaluate Lagrange basis functions at quadrature points
        lagrange_vals = self._evaluate_lagrange_basis(
            lagrange_nodes, quad_pts[0, :]
        )  # (nquad, npts)

        # Compute projection coefficients via quadrature
        # c_jk = integral(L_j * P_k * weight) ≈ sum_i w_i * L_j(x_i) * P_k(x_i)
        # coefs[j, k] = sum_i w_i * lagrange_vals[i, j] * ortho_vals[i, k]
        weighted_ortho = ortho_vals * self._bkd.reshape(quad_wts, (-1, 1))
        coefs = self._bkd.dot(lagrange_vals.T, weighted_ortho)  # (npts, npts)

        return coefs

    def _evaluate_lagrange_basis(
        self,
        nodes: Array,
        x: Array,
    ) -> Array:
        """Evaluate all Lagrange basis functions at given points.

        Parameters
        ----------
        nodes : Array
            Interpolation nodes, shape (npts,)
        x : Array
            Evaluation points, shape (neval,)

        Returns
        -------
        Array
            Lagrange basis values, shape (neval, npts)
        """
        npts = nodes.shape[0]
        neval = x.shape[0]
        result = self._bkd.zeros((neval, npts))

        for j in range(npts):
            # L_j(x) = prod_{k != j} (x - x_k) / (x_j - x_k)
            x_j = self._bkd.to_float(nodes[j])
            L_j = self._bkd.ones((neval,))
            for k in range(npts):
                if k != j:
                    x_k = self._bkd.to_float(nodes[k])
                    L_j = L_j * (x - x_k) / (x_j - x_k)
            result[:, j] = L_j

        return result

    def _get_projection_coefficients(
        self,
        dim: int,
        lagrange_nodes: Array,
    ) -> Array:
        """Get cached projection coefficients or compute them.

        Parameters
        ----------
        dim : int
            Dimension index.
        lagrange_nodes : Array
            Nodes defining Lagrange basis.

        Returns
        -------
        Array
            Projection coefficients.
        """
        # Create cache key from nodes (convert to tuple for hashing)
        flat = self._bkd.flatten(lagrange_nodes)
        nodes_key = tuple(self._bkd.to_float(n) for n in flat)
        cache_key = (dim, nodes_key)

        if cache_key not in self._cached_projection_coefs:
            flat_nodes = self._bkd.flatten(lagrange_nodes)
            self._cached_projection_coefs[cache_key] = (
                self._compute_univariate_projection_coefficients(dim, flat_nodes)
            )

        return self._cached_projection_coefs[cache_key]

    def convert_subspace(
        self,
        subspace: TensorProductSubspace[Array],
    ) -> Tuple[Array, Array]:
        """Convert a tensor product subspace to PCE coefficients.

        Parameters
        ----------
        subspace : TensorProductSubspace[Array]
            Tensor product subspace with values set.

        Returns
        -------
        indices : Array
            Multi-indices for PCE terms, shape (nvars, nterms)
        coefficients : Array
            PCE coefficients, shape (nqoi, nterms)
        """
        values = subspace.get_values()
        if values is None:
            raise ValueError("Subspace values not set")

        nqoi = values.shape[0]  # nqoi is first dimension

        # Get 1D projection coefficients for each dimension
        projection_coefs_1d: List[Array] = []
        npts_1d: List[int] = []

        for dim in range(self._nvars):
            samples_1d = subspace.get_samples_1d(dim)
            nodes = self._bkd.flatten(samples_1d)
            proj_coefs = self._get_projection_coefficients(dim, nodes)
            projection_coefs_1d.append(proj_coefs)
            npts_1d.append(nodes.shape[0])

        # Build tensor product indices
        # Total number of tensor product terms
        nterms = 1
        for npts in npts_1d:
            nterms *= npts

        # Create multi-indices
        indices = self._bkd.zeros((self._nvars, nterms), dtype=self._bkd.int64_dtype())

        # Build indices using same tensor product ordering as subspace samples
        repeat_inner = 1
        for dim in range(self._nvars - 1, -1, -1):
            npts = npts_1d[dim]
            repeat_outer = nterms // (npts * repeat_inner)

            col = 0
            for _ in range(repeat_outer):
                for pt_idx in range(npts):
                    for _ in range(repeat_inner):
                        indices[dim, col] = pt_idx
                        col += 1

            repeat_inner *= npts

        # Compute PCE coefficients via tensor product of 1D projections
        # For each tensor product index (i_0, i_1, ..., i_{d-1}):
        #   PCE_coef = sum over Lagrange indices (j_0, ..., j_{d-1}) of
        #              prod_d proj_coefs[d][j_d, i_d] * Lagrange_value[j_0,...,j_{d-1}]

        coefficients = self._bkd.zeros((nqoi, nterms))

        # The subspace values are organized in tensor product order
        # matching the samples from _build_tensor_product_samples
        nsamples = values.shape[1]  # nsamples is second dimension
        for term_idx in range(nterms):
            # Extract multi-index for this PCE term
            term_multi_idx = [
                self._bkd.to_int(indices[dim, term_idx])
                for dim in range(self._nvars)
            ]

            # Sum over all Lagrange basis functions
            for sample_idx in range(nsamples):
                # Extract multi-index for this sample (Lagrange basis)
                sample_multi_idx = self._sample_idx_to_multi_idx(sample_idx, npts_1d)

                # Compute tensor product of 1D projection coefficients
                proj_coef = 1.0
                for dim in range(self._nvars):
                    j_d = sample_multi_idx[dim]
                    i_d = term_multi_idx[dim]
                    proj_coef *= float(projection_coefs_1d[dim][j_d, i_d])

                # Add contribution from this Lagrange basis
                coefficients[:, term_idx] += proj_coef * values[:, sample_idx]

        return indices, coefficients

    def _sample_idx_to_multi_idx(
        self, sample_idx: int, npts_1d: List[int]
    ) -> List[int]:
        """Convert flat sample index to multi-index.

        Uses same ordering as TensorProductSubspace._build_tensor_product_samples.
        """
        multi_idx = []
        remaining = sample_idx
        stride = 1
        for npts in reversed(npts_1d):
            stride *= npts

        for dim in range(self._nvars):
            stride //= npts_1d[dim]
            idx = remaining // stride
            remaining = remaining % stride
            multi_idx.append(idx)

        return multi_idx


class SparseGridToPCEConverter(Generic[Array]):
    """Convert a CombinationSurrogate to a Polynomial Chaos Expansion.

    Uses spectral projection to convert each tensor product subspace's
    Lagrange interpolant to PCE, then combines using Smolyak coefficients.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    orthonormal_bases_1d : Sequence[PhysicalDomainBasis1DProtocol[Array]]
        Univariate bases for each dimension, created via ``create_bases_1d()``.
        These define the target PCE basis and must return physical-domain
        quadrature points from ``gauss_quadrature_rule()``.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability import UniformMarginal
    >>> from pyapprox.surrogates.affine.univariate import create_bases_1d
    >>> from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    ...     IsotropicSparseGridFitter,
    ... )
    >>> from pyapprox.surrogates.sparsegrids.basis_factory import (
    ...     GaussLagrangeFactory,
    ... )
    >>> from pyapprox.surrogates.affine.indices import LinearGrowthRule
    >>> bkd = NumpyBkd()
    >>> marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
    >>> factories = [GaussLagrangeFactory(m, bkd) for m in marginals]
    >>> growth = LinearGrowthRule(scale=2, shift=1)
    >>> # ... build fitter, get samples, fit ...
    >>> bases_1d = create_bases_1d(marginals, bkd)
    >>> converter = SparseGridToPCEConverter(bkd, bases_1d)
    >>> # pce = converter.convert(result.surrogate)

    See Also
    --------
    create_bases_1d : Factory function to create physical-domain bases.
    TensorProductSubspaceToPCEConverter : Lower-level subspace converter.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        orthonormal_bases_1d: Sequence[PhysicalDomainBasis1DProtocol[Array]],
    ):
        self._bkd = bkd
        self._orthonormal_bases_1d = list(orthonormal_bases_1d)
        self._nvars = len(orthonormal_bases_1d)
        self._subspace_converter = TensorProductSubspaceToPCEConverter(
            bkd, self._orthonormal_bases_1d
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def convert(
        self,
        surrogate: CombinationSurrogate[Array],
        nqoi: Optional[int] = None,
    ) -> PolynomialChaosExpansion[Array]:
        """Convert combination surrogate to Polynomial Chaos Expansion.

        Parameters
        ----------
        surrogate : CombinationSurrogate[Array]
            Fitted combination surrogate with subspace values set.
        nqoi : int, optional
            Number of quantities of interest. If None, inferred from
            surrogate.

        Returns
        -------
        PolynomialChaosExpansion[Array]
            The converted PCE.
        """
        if surrogate.nvars() != self._nvars:
            raise ValueError(
                f"Surrogate has {surrogate.nvars()} variables, "
                f"but converter has {self._nvars} bases"
            )

        subspaces = surrogate.subspaces()
        smolyak_coefs = surrogate.coefficients()

        if len(subspaces) == 0:
            raise ValueError("Sparse grid has no subspaces")

        # Determine nqoi from first subspace
        first_values = subspaces[0].get_values()
        if first_values is None:
            raise ValueError("Sparse grid values not set")
        if nqoi is None:
            nqoi = first_values.shape[0]  # nqoi is first dimension

        # Collect all unique indices and their coefficients
        # Key: multi-index tuple, Value: coefficient array
        all_indices: Dict[Tuple[int, ...], Array] = {}

        for subspace_idx, subspace in enumerate(subspaces):
            coef = self._bkd.to_float(smolyak_coefs[subspace_idx])
            if abs(coef) < 1e-14:
                continue

            indices, coefficients = self._subspace_converter.convert_subspace(subspace)

            # Combine with Smolyak coefficient
            weighted_coefficients = coef * coefficients

            # Merge into accumulated indices
            # coefficients has shape (nqoi, nterms)
            for term_idx in range(indices.shape[1]):
                idx_tuple = tuple(
                    int(indices[dim, term_idx]) for dim in range(self._nvars)
                )

                if idx_tuple in all_indices:
                    all_indices[idx_tuple] = (
                        all_indices[idx_tuple] + weighted_coefficients[:, term_idx]
                    )
                else:
                    all_indices[idx_tuple] = self._bkd.copy(
                        weighted_coefficients[:, term_idx]
                    )

        # Build final PCE
        nterms = len(all_indices)
        pce_indices = self._bkd.zeros(
            (self._nvars, nterms), dtype=self._bkd.int64_dtype()
        )
        pce_coefficients = self._bkd.zeros((nqoi, nterms))

        for j, (idx_tuple, coefs) in enumerate(all_indices.items()):
            for dim in range(self._nvars):
                pce_indices[dim, j] = idx_tuple[dim]
            pce_coefficients[:, j] = coefs

        # Determine maximum index in each dimension and set 1D basis nterms
        max_indices = [0] * self._nvars
        for idx_tuple in all_indices.keys():
            for dim in range(self._nvars):
                max_indices[dim] = max(max_indices[dim], idx_tuple[dim])

        for dim in range(self._nvars):
            # Need nterms >= max_index + 1 for evaluation
            required_nterms = max_indices[dim] + 1
            if self._orthonormal_bases_1d[dim].nterms() < required_nterms:
                self._orthonormal_bases_1d[dim].set_nterms(required_nterms)

        # Create PCE with these indices
        basis = OrthonormalPolynomialBasis(
            self._orthonormal_bases_1d, self._bkd, pce_indices
        )
        pce = PolynomialChaosExpansion(basis, self._bkd, nqoi)
        # PCE expects coefficients in (nterms, nqoi) format
        pce.set_coefficients(pce_coefficients.T)

        return pce
