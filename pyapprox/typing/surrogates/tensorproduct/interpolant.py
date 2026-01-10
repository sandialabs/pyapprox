"""Tensor product interpolation using 1D interpolation bases.

This module provides a general-purpose tensor product interpolant that can be
used independently of sparse grids. It requires 1D bases that satisfy the
InterpolationBasis1DProtocol.
"""

from typing import Generic, List, Optional, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.cartesian import (
    cartesian_product_indices,
    cartesian_product_samples,
)
from pyapprox.typing.surrogates.tensorproduct.protocols import (
    InterpolationBasis1DProtocol,
    Basis1DHasJacobianProtocol,
    Basis1DHasHessianProtocol,
)


class TensorProductInterpolant(Generic[Array]):
    """Tensor product interpolant using 1D interpolation bases.

    This class implements tensor product interpolation using Lagrange or other
    interpolation bases that satisfy InterpolationBasis1DProtocol. It strictly
    requires bases with a `get_samples` method - orthogonal polynomial bases
    (Legendre, Hermite, etc.) are NOT accepted directly.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend (NumPy or PyTorch).
    bases_1d : Sequence[InterpolationBasis1DProtocol[Array]]
        Univariate interpolation bases for each dimension.
    nterms_1d : Sequence[int]
        Number of interpolation points in each dimension.

    Raises
    ------
    TypeError
        If any basis does not satisfy InterpolationBasis1DProtocol.
    ValueError
        If lengths of bases_1d and nterms_1d don't match.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import (
    ...     LagrangeBasis1D, LegendrePolynomial1D
    ... )
    >>> bkd = NumpyBkd()
    >>> poly = LegendrePolynomial1D(bkd)
    >>> poly.set_nterms(10)
    >>> basis = LagrangeBasis1D(bkd, poly.gauss_quadrature_rule)
    >>> interp = TensorProductInterpolant(bkd, [basis, basis], [3, 4])
    """

    def __init__(
        self,
        bkd: Backend[Array],
        bases_1d: Sequence[InterpolationBasis1DProtocol[Array]],
        nterms_1d: Sequence[int],
    ):
        if len(bases_1d) != len(nterms_1d):
            raise ValueError(
                f"Length mismatch: bases_1d has {len(bases_1d)} elements, "
                f"nterms_1d has {len(nterms_1d)} elements"
            )

        # Validate that all bases satisfy the protocol
        for i, basis in enumerate(bases_1d):
            if not isinstance(basis, InterpolationBasis1DProtocol):
                raise TypeError(
                    f"Basis at index {i} does not satisfy "
                    f"InterpolationBasis1DProtocol. "
                    f"Got type {type(basis).__name__}. "
                    f"Use LagrangeBasis1D or another interpolation basis."
                )

        self._bkd = bkd
        self._bases_1d = bases_1d
        self._nterms_1d = list(nterms_1d)
        self._values: Optional[Array] = None

        # Initialize each basis with the number of terms
        for basis, nterms in zip(self._bases_1d, self._nterms_1d):
            basis.set_nterms(nterms)

        # Get 1D samples from each basis
        self._samples_1d: List[Array] = []
        for basis, nterms in zip(self._bases_1d, self._nterms_1d):
            samples = basis.get_samples(nterms)
            self._samples_1d.append(samples)

        # Generate tensor product indices for vectorized evaluation
        self._tp_indices = cartesian_product_indices(self._nterms_1d, bkd)

        # Build tensor product samples
        self._samples = cartesian_product_samples(self._samples_1d, bkd)
        self._nsamples = self._samples.shape[1]

        # Detect derivative support from bases
        self._jacobian_supported = all(
            isinstance(b, Basis1DHasJacobianProtocol) for b in self._bases_1d
        )
        self._hessian_supported = all(
            isinstance(b, Basis1DHasHessianProtocol) for b in self._bases_1d
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables (dimensions)."""
        return len(self._bases_1d)

    def nsamples(self) -> int:
        """Return the total number of interpolation points."""
        return self._nsamples

    def nqoi(self) -> int:
        """Return the number of quantities of interest, or 0 if not set."""
        if self._values is None:
            return 0
        return self._values.shape[0]

    def get_samples(self) -> Array:
        """Return interpolation node locations.

        Returns
        -------
        Array
            Sample locations with shape (nvars, nsamples).
        """
        return self._bkd.copy(self._samples)

    def get_samples_1d(self, dim: int) -> Array:
        """Return 1D interpolation nodes for a specific dimension.

        Parameters
        ----------
        dim : int
            Dimension index.

        Returns
        -------
        Array
            1D sample locations with shape (1, nterms_1d[dim]).
        """
        return self._bkd.copy(self._samples_1d[dim])

    def get_values(self) -> Optional[Array]:
        """Return function values at samples, if set."""
        return self._values

    def set_values(self, values: Array) -> None:
        """Set function values at interpolation nodes.

        Parameters
        ----------
        values : Array
            Function values with shape (nqoi, nsamples).

        Raises
        ------
        ValueError
            If values shape is incompatible with nsamples.
        """
        if values.shape[1] != self._nsamples:
            raise ValueError(
                f"Expected {self._nsamples} samples, got {values.shape[1]}"
            )
        self._values = self._bkd.copy(values)

    def _basis_vals_1d(self, samples: Array) -> List[Array]:
        """Evaluate all 1D bases at samples.

        Parameters
        ----------
        samples : Array
            Sample points with shape (nvars, npoints).

        Returns
        -------
        List[Array]
            List of basis values, each with shape (npoints, nterms_1d[d]).
        """
        vals = []
        for dd in range(self.nvars()):
            # samples[dd:dd+1, :] has shape (1, npoints)
            vals.append(self._bases_1d[dd](samples[dd : dd + 1, :]))
        return vals

    def __call__(self, samples: Array) -> Array:
        """Evaluate the interpolant at given samples.

        Uses vectorized tensor product evaluation: evaluate all 1D bases once,
        then combine via fancy indexing and element-wise multiplication.

        Parameters
        ----------
        samples : Array
            Evaluation points with shape (nvars, npoints).

        Returns
        -------
        Array
            Interpolant values with shape (nqoi, npoints).

        Raises
        ------
        ValueError
            If values have not been set.
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        # Evaluate all 1D bases at samples (vectorized)
        # Each element has shape (npoints, nterms_1d[d])
        basis_vals_1d = self._basis_vals_1d(samples)

        # Build interpolation matrix via tensor product using fancy indexing
        # interp_mat[s, i] = prod_d basis_vals_1d[d][s, indices[d, i]]
        interp_mat = basis_vals_1d[0][:, self._tp_indices[0, :]]

        for dd in range(1, self.nvars()):
            interp_mat = interp_mat * basis_vals_1d[dd][:, self._tp_indices[dd, :]]

        # Apply to values: (nqoi, nsamples) @ (nsamples, npoints) = (nqoi, npoints)
        return self._values @ interp_mat.T

    # Derivative support methods

    def jacobian_supported(self) -> bool:
        """Return whether Jacobian computation is supported."""
        return self._jacobian_supported

    def hessian_supported(self) -> bool:
        """Return whether Hessian computation is supported."""
        return self._hessian_supported

    def _basis_jacobians_1d(self, samples: Array) -> List[Array]:
        """Evaluate first derivatives of all 1D bases.

        Parameters
        ----------
        samples : Array
            Sample points with shape (nvars, npoints).

        Returns
        -------
        List[Array]
            List of derivatives, each with shape (npoints, nterms_1d[d]).
        """
        if not self._jacobian_supported:
            raise RuntimeError("Jacobian not supported by univariate bases")

        derivs = []
        for dd in range(self.nvars()):
            jac = self._bases_1d[dd].jacobian_batch(samples[dd : dd + 1, :])
            derivs.append(jac)
        return derivs

    def _basis_hessians_1d(self, samples: Array) -> List[Array]:
        """Evaluate second derivatives of all 1D bases.

        Parameters
        ----------
        samples : Array
            Sample points with shape (nvars, npoints).

        Returns
        -------
        List[Array]
            List of second derivatives, each with shape (npoints, nterms_1d[d]).
        """
        if not self._hessian_supported:
            raise RuntimeError("Hessian not supported by univariate bases")

        derivs = []
        for dd in range(self.nvars()):
            hess = self._bases_1d[dd].hessian_batch(samples[dd : dd + 1, :])
            derivs.append(hess)
        return derivs

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample point.

        Uses the product rule on the tensor product structure.

        Parameters
        ----------
        sample : Array
            Single evaluation point with shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian matrix with shape (nqoi, nvars).

        Raises
        ------
        ValueError
            If values have not been set.
        RuntimeError
            If Jacobian is not supported by the bases.
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")
        if not self._jacobian_supported:
            raise RuntimeError("Jacobian not supported by univariate bases")

        nvars = self.nvars()
        nqoi = self._values.shape[0]

        # Get 1D basis values and derivatives (single sample)
        basis_vals_1d = self._basis_vals_1d(sample)
        basis_derivs_1d = self._basis_jacobians_1d(sample)

        jacobian = self._bkd.zeros((nqoi, nvars))

        for dim in range(nvars):
            # Build tensor product: derivative in dim, values in other dims
            interp_deriv = basis_derivs_1d[dim][0, self._tp_indices[dim, :]]

            for dd in range(nvars):
                if dd != dim:
                    interp_deriv = (
                        interp_deriv * basis_vals_1d[dd][0, self._tp_indices[dd, :]]
                    )

            # Contract with values: (nqoi, nsamples) @ (nsamples,) = (nqoi,)
            jacobian[:, dim] = self._values @ interp_deriv

        return jacobian

    def hessian(self, sample: Array) -> Array:
        """Compute Hessian at a single sample point.

        Only valid when nqoi == 1. For multiple QoIs, use whvp().

        Parameters
        ----------
        sample : Array
            Single evaluation point with shape (nvars, 1).

        Returns
        -------
        Array
            Hessian matrix with shape (nvars, nvars).

        Raises
        ------
        ValueError
            If values have not been set or nqoi > 1.
        RuntimeError
            If Hessian is not supported by the bases.
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")
        if self._values.shape[0] != 1:
            raise ValueError(
                f"hessian() only valid for nqoi=1, got nqoi={self._values.shape[0]}. "
                "Use whvp() for multi-QoI."
            )
        if not self._hessian_supported:
            raise RuntimeError("Hessian not supported by univariate bases")

        nvars = self.nvars()

        basis_vals_1d = self._basis_vals_1d(sample)
        basis_derivs_1d = self._basis_jacobians_1d(sample)
        basis_hess_1d = self._basis_hessians_1d(sample)

        hessian = self._bkd.zeros((nvars, nvars))
        values_q = self._values[0, :]  # Shape: (nsamples,), nqoi must be 1

        for dim1 in range(nvars):
            for dim2 in range(dim1, nvars):
                if dim1 == dim2:
                    # Diagonal: second derivative in dimension dim1
                    interp_deriv = basis_hess_1d[dim1][0, self._tp_indices[dim1, :]]
                else:
                    # Off-diagonal: first derivatives in both dimensions
                    interp_deriv = (
                        basis_derivs_1d[dim1][0, self._tp_indices[dim1, :]]
                        * basis_derivs_1d[dim2][0, self._tp_indices[dim2, :]]
                    )

                # Multiply by values in remaining dimensions
                for dd in range(nvars):
                    if dd != dim1 and dd != dim2:
                        interp_deriv = (
                            interp_deriv
                            * basis_vals_1d[dd][0, self._tp_indices[dd, :]]
                        )

                val = self._bkd.dot(interp_deriv, values_q)
                hessian[dim1, dim2] = val
                if dim1 != dim2:
                    hessian[dim2, dim1] = val

        return hessian

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product efficiently.

        Computes H @ v where H is the Hessian, without explicitly forming H.
        Only valid when nqoi == 1. For multiple QoIs, use whvp().

        Parameters
        ----------
        sample : Array
            Single evaluation point with shape (nvars, 1).
        vec : Array
            Direction vector with shape (nvars, 1).

        Returns
        -------
        Array
            Hessian-vector product with shape (nvars, 1).

        Raises
        ------
        ValueError
            If values have not been set or nqoi > 1.
        RuntimeError
            If HVP is not supported by the bases.
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")
        if self._values.shape[0] != 1:
            raise ValueError(
                f"hvp() only valid for nqoi=1, got nqoi={self._values.shape[0]}. "
                "Use whvp() for multi-QoI."
            )
        if not self._hessian_supported:
            raise RuntimeError("HVP not supported by univariate bases")

        nvars = self.nvars()
        vec_flat = vec.flatten()

        basis_vals_1d = self._basis_vals_1d(sample)
        basis_derivs_1d = self._basis_jacobians_1d(sample)
        basis_hess_1d = self._basis_hessians_1d(sample)

        values_q = self._values[0, :]  # Shape: (nsamples,), nqoi must be 1
        result = self._bkd.zeros((nvars, 1))

        for dim1 in range(nvars):
            row_sum = 0.0

            for dim2 in range(nvars):
                v_j = float(vec_flat[dim2])
                if abs(v_j) < 1e-14:
                    continue

                if dim1 == dim2:
                    interp_deriv = basis_hess_1d[dim1][0, self._tp_indices[dim1, :]]
                else:
                    interp_deriv = (
                        basis_derivs_1d[dim1][0, self._tp_indices[dim1, :]]
                        * basis_derivs_1d[dim2][0, self._tp_indices[dim2, :]]
                    )

                for dd in range(nvars):
                    if dd != dim1 and dd != dim2:
                        interp_deriv = (
                            interp_deriv
                            * basis_vals_1d[dd][0, self._tp_indices[dd, :]]
                        )

                H_ij = float(self._bkd.dot(interp_deriv, values_q))
                row_sum += H_ij * v_j

            result[dim1, 0] = row_sum

        return result

    def _hvp_for_qoi(
        self, sample: Array, vec: Array, qoi_idx: int,
        basis_vals_1d: List[Array],
        basis_derivs_1d: List[Array],
        basis_hess_1d: List[Array],
    ) -> Array:
        """Compute HVP for a specific QoI index (internal helper).

        This is used by whvp to compute weighted HVP across QoIs.
        """
        nvars = self.nvars()
        vec_flat = vec.flatten()
        values_q = self._values[qoi_idx, :]  # Shape: (nsamples,)
        result = self._bkd.zeros((nvars, 1))

        for dim1 in range(nvars):
            row_sum = 0.0

            for dim2 in range(nvars):
                v_j = float(vec_flat[dim2])
                if abs(v_j) < 1e-14:
                    continue

                if dim1 == dim2:
                    interp_deriv = basis_hess_1d[dim1][0, self._tp_indices[dim1, :]]
                else:
                    interp_deriv = (
                        basis_derivs_1d[dim1][0, self._tp_indices[dim1, :]]
                        * basis_derivs_1d[dim2][0, self._tp_indices[dim2, :]]
                    )

                for dd in range(nvars):
                    if dd != dim1 and dd != dim2:
                        interp_deriv = (
                            interp_deriv
                            * basis_vals_1d[dd][0, self._tp_indices[dd, :]]
                        )

                H_ij = float(self._bkd.dot(interp_deriv, values_q))
                row_sum += H_ij * v_j

            result[dim1, 0] = row_sum

        return result

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Compute weighted Hessian-vector product.

        Computes sum_q weights[q] * H_q @ v where H_q is the Hessian for QoI q.
        This is useful for multi-QoI optimization.

        Parameters
        ----------
        sample : Array
            Single evaluation point with shape (nvars, 1).
        vec : Array
            Direction vector with shape (nvars, 1).
        weights : Array
            Weights for each QoI. Shape: (nqoi, 1), (1, nqoi), or (nqoi,).

        Returns
        -------
        Array
            Weighted Hessian-vector product with shape (nvars, 1).

        Raises
        ------
        ValueError
            If values have not been set.
        RuntimeError
            If WHVP is not supported by the bases.
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")
        if not self._hessian_supported:
            raise RuntimeError("WHVP not supported by univariate bases")

        nqoi = self._values.shape[0]
        result = self._bkd.zeros((self.nvars(), 1))

        weights_flat = weights.flatten()

        # Precompute basis evaluations once for all QoIs
        basis_vals_1d = self._basis_vals_1d(sample)
        basis_derivs_1d = self._basis_jacobians_1d(sample)
        basis_hess_1d = self._basis_hessians_1d(sample)

        for qoi_idx in range(nqoi):
            w = float(weights_flat[qoi_idx])
            if abs(w) > 1e-14:
                hvp_q = self._hvp_for_qoi(
                    sample, vec, qoi_idx,
                    basis_vals_1d, basis_derivs_1d, basis_hess_1d
                )
                result = result + w * hvp_q

        return result

    def __repr__(self) -> str:
        nterms_str = ",".join(str(n) for n in self._nterms_1d)
        return (
            f"TensorProductInterpolant(nvars={self.nvars()}, "
            f"nterms_1d=[{nterms_str}], nsamples={self._nsamples})"
        )
