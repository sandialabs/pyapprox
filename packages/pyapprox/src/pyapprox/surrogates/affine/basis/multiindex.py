"""Multi-index basis for multivariate polynomial approximations.

This module provides the MultiIndexBasis class, which combines univariate
basis functions via tensor products according to multi-indices.

The basis evaluation follows:
    φ_i(x) = ∏_d φ_{i_d}^{(d)}(x_d)

where i = (i_0, ..., i_{nvars-1}) is a multi-index and φ_j^{(d)} is the
j-th univariate basis function in dimension d.
"""

from abc import ABC
from typing import Generic, List, Optional

from pyapprox.surrogates.affine.basis.dispatch import (
    get_basis_eval_impl,
    get_basis_hessian_impl,
    get_basis_jacobian_impl,
)
from pyapprox.surrogates.affine.protocols import (
    Basis1DHasHessianProtocol,
    Basis1DHasJacobianProtocol,
    Basis1DProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class MultiIndexBasis(ABC, Generic[Array]):
    """Base class for multi-index bases.

    A multi-index basis combines univariate bases via tensor products,
    selecting specific combinations via multi-indices.

    Parameters
    ----------
    bases_1d : List[Basis1DProtocol[Array]]
        List of univariate basis functions, one per variable.
    bkd : Backend[Array]
        Computational backend.
    indices : Array, optional
        Multi-indices specifying which basis functions to include.
        Shape: (nvars, nterms). If None, must be set later.
    """

    def __init__(
        self,
        bases_1d: List[Basis1DProtocol[Array]],
        bkd: Backend[Array],
        indices: Optional[Array] = None,
    ):
        self._bkd = bkd
        self._bases_1d = bases_1d
        self._indices: Optional[Array] = None

        # Check for derivative support
        self._jacobian_supported = all(
            isinstance(b, Basis1DHasJacobianProtocol) for b in bases_1d
        )
        self._hessian_supported = all(
            isinstance(b, Basis1DHasHessianProtocol) for b in bases_1d
        )

        # Resolve dispatch at init time based on backend type
        self._eval_impl = get_basis_eval_impl(bkd)
        self._jacobian_impl = (
            get_basis_jacobian_impl(bkd) if self._jacobian_supported else None
        )
        self._hessian_impl = (
            get_basis_hessian_impl(bkd) if self._hessian_supported else None
        )

        if indices is not None:
            self.set_indices(indices)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return len(self._bases_1d)

    def nterms(self) -> int:
        """Return the number of basis terms."""
        if self._indices is None:
            return 0
        return self._indices.shape[1]

    def get_indices(self) -> Array:
        """Return the multi-indices.

        Returns
        -------
        Array
            Multi-indices. Shape: (nvars, nterms)
        """
        if self._indices is None:
            raise ValueError("Indices have not been set")
        return self._indices

    def set_indices(self, indices: Array) -> None:
        """Set the multi-indices.

        Parameters
        ----------
        indices : Array
            Multi-indices. Shape: (nvars, nterms)
        """
        if indices.ndim != 2:
            raise ValueError("indices must be 2D array")
        if indices.shape[0] != self.nvars():
            raise ValueError(
                f"indices must have shape ({self.nvars()}, nterms), got {indices.shape}"
            )

        self._indices = indices

        # Update univariate bases to have enough terms
        for dd in range(self.nvars()):
            max_degree = self._bkd.to_int(self._bkd.max(indices[dd, :]))
            # Need at least max_degree + 1 terms (for degree 0 to max_degree)
            needed_terms = max_degree + 1
            if self._bases_1d[dd].nterms() < needed_terms:
                self._bases_1d[dd].set_nterms(needed_terms)

    def get_univariate_basis(self, dim: int) -> Basis1DProtocol[Array]:
        """Return the univariate basis for a dimension.

        Parameters
        ----------
        dim : int
            Dimension index.

        Returns
        -------
        Basis1DProtocol[Array]
            Univariate basis for this dimension.
        """
        return self._bases_1d[dim]

    def jacobian_supported(self) -> bool:
        """Return True if Jacobian computation is supported."""
        return self._jacobian_supported

    def hessian_supported(self) -> bool:
        """Return True if Hessian computation is supported."""
        return self._hessian_supported

    def _basis_vals_1d(self, samples: Array) -> List[Array]:
        """Evaluate all univariate bases at samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        List[Array]
            List of univariate basis values.
            Each element has shape (nsamples, nterms_1d).
        """
        vals = []
        for dd in range(self.nvars()):
            # samples[dd:dd+1, :] has shape (1, nsamples)
            vals.append(self._bases_1d[dd](samples[dd : dd + 1, :]))
        return vals

    def _basis_jacobians_1d(self, samples: Array) -> List[Array]:
        """Evaluate first derivatives of univariate bases.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        List[Array]
            List of univariate basis jacobians.
            Each element has shape (nsamples, nterms_1d).
        """
        if not self._jacobian_supported:
            raise RuntimeError("Jacobian not supported by univariate bases")

        derivs = []
        for dd in range(self.nvars()):
            basis = self._bases_1d[dd]
            # Univariate jacobian_batch returns (nsamples, nterms_1d)
            jac = basis.jacobian_batch(samples[dd : dd + 1, :])
            derivs.append(jac)
        return derivs

    def _basis_hessians_1d(self, samples: Array) -> List[Array]:
        """Evaluate second derivatives of univariate bases.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        List[Array]
            List of univariate basis hessians.
            Each element has shape (nsamples, nterms_1d).
        """
        if not self._hessian_supported:
            raise RuntimeError("Hessian not supported by univariate bases")

        derivs = []
        for dd in range(self.nvars()):
            basis = self._bases_1d[dd]
            # Univariate hessian_batch returns (nsamples, nterms_1d)
            hess = basis.hessian_batch(samples[dd : dd + 1, :])
            derivs.append(hess)
        return derivs

    def __call__(self, samples: Array) -> Array:
        """Evaluate the basis at sample points.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Basis values. Shape: (nsamples, nterms)
        """
        if self._indices is None:
            raise ValueError("Indices have not been set")

        basis_vals_1d = self._basis_vals_1d(samples)
        return self._eval_impl(
            basis_vals_1d,
            self._indices,
            self.nvars(),
            self._bkd,
        )

    def jacobian_batch(self, samples: Array) -> Array:
        """Compute Jacobians of basis functions.

        For multivariate basis functions, the Jacobian w.r.t. x_d is:
            ∂φ_i/∂x_d = (∂φ_{i_d}^{(d)}/∂x_d) ∏_{k≠d} φ_{i_k}^{(k)}

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Jacobians. Shape: (nsamples, nterms, nvars)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (nvars, nsamples).
        """
        if self._indices is None:
            raise ValueError("Indices have not been set")
        if not self._jacobian_supported:
            raise RuntimeError("Jacobian not supported by univariate bases")

        basis_vals_1d = self._basis_vals_1d(samples)
        basis_derivs_1d = self._basis_jacobians_1d(samples)
        return self._jacobian_impl(
            basis_vals_1d,
            basis_derivs_1d,
            self._indices,
            self.nvars(),
            self._bkd,
        )

    def hessian_batch(self, samples: Array) -> Array:
        """Compute Hessians of basis functions.

        For diagonal terms (d == k):
            ∂²φ_i/∂x_d² = (∂²φ_{i_d}^{(d)}/∂x_d²) ∏_{l≠d} φ_{i_l}^{(l)}

        For off-diagonal terms (d ≠ k):
            ∂²φ_i/∂x_d∂x_k = (∂φ_{i_d}^{(d)}/∂x_d)(∂φ_{i_k}^{(k)}/∂x_k) ∏_{l≠d,k}
            φ_{i_l}^{(l)}

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Hessians. Shape: (nsamples, nterms, nvars, nvars)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (nvars, nsamples).
        """
        if self._indices is None:
            raise ValueError("Indices have not been set")
        if not self._hessian_supported:
            raise RuntimeError("Hessian not supported by univariate bases")

        basis_vals_1d = self._basis_vals_1d(samples)
        basis_derivs_1d = self._basis_jacobians_1d(samples)
        basis_hess_1d = self._basis_hessians_1d(samples)

        return self._hessian_impl(
            basis_vals_1d,
            basis_derivs_1d,
            basis_hess_1d,
            self._indices,
            self.nvars(),
            self._bkd,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nvars={self.nvars()}, nterms={self.nterms()})"
        )
