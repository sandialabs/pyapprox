"""Tensor product interpolation using 1D interpolation bases.

This module provides a general-purpose tensor product interpolant that can be
used independently of sparse grids. It requires 1D bases that satisfy the
InterpolationBasis1DProtocol.
"""

from typing import Generic, List, Optional, Sequence

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.surrogates.tensorproduct.dispatch import (
    get_tp_eval_impl,
)
from pyapprox.surrogates.tensorproduct.protocols import (
    Basis1DHasHessianProtocol,
    Basis1DHasJacobianProtocol,
    InterpolationBasis1DProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.cartesian import (
    cartesian_product_indices,
    cartesian_product_samples,
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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.affine.univariate import (
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

        # Reject shared basis objects with different nterms (set_nterms on
        # a shared object causes silent corruption of earlier dimensions)
        unique_ids = {id(b) for b in bases_1d}
        if len(unique_ids) < len(bases_1d):
            nterms_set = set(nterms_1d)
            if len(nterms_set) > 1:
                raise ValueError(
                    "Same basis object passed for multiple dimensions with "
                    "different nterms_1d values. Use separate basis instances "
                    "per dimension when nterms_1d differ."
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

        # Detect derivative support from bases, keeping the narrowed base
        # lists so downstream evaluation needs no casts
        jac_bases: List[Basis1DHasJacobianProtocol[Array]] = []
        hess_bases: List[Basis1DHasHessianProtocol[Array]] = []
        for b in self._bases_1d:
            if isinstance(b, Basis1DHasJacobianProtocol):
                jac_bases.append(b)
            if isinstance(b, Basis1DHasHessianProtocol):
                hess_bases.append(b)
        self._jac_bases_1d: Optional[List[Basis1DHasJacobianProtocol[Array]]] = (
            jac_bases if len(jac_bases) == len(self._bases_1d) else None
        )
        self._hess_bases_1d: Optional[List[Basis1DHasHessianProtocol[Array]]] = (
            hess_bases if len(hess_bases) == len(self._bases_1d) else None
        )
        self._jacobian_supported = self._jac_bases_1d is not None
        self._hessian_supported = self._hess_bases_1d is not None

        # Select accelerated evaluation strategy based on backend
        self._tp_eval_impl = get_tp_eval_impl(bkd)

        self._derivs: Derivatives[Array] = self._build_derivatives()

    def _build_derivatives(self) -> Derivatives[Array]:
        """Capability bundle; rebuilt by set_values(), the nqoi decision point.

        Before values are set nqoi is unknown, so the scalar-only fields
        (hessian/hvp) are included optimistically when the bases support
        second derivatives; invoking them before set_values() raises, and
        set_values() rebuilds the bundle to match the actual nqoi.
        """
        if not self._jacobian_supported:
            return Derivatives.none()
        if not self._hessian_supported:
            return Derivatives.first_order(
                jacobian=self._jacobian,
                jacobian_batch=self._jacobian_batch,
            )
        if self._values is not None and self._values.shape[0] != 1:
            # vector-valued: second order only through the weighted form
            return Derivatives.second_order_weighted(
                jacobian=self._jacobian,
                jacobian_batch=self._jacobian_batch,
                whvp=self._whvp,
                whvp_batch=self._whvp_batch,
            )
        # jacobian + materialized hessian + hvp + whvp is an unusual
        # combination, so the raw constructor is used
        return Derivatives(
            jacobian=self._jacobian,
            jacobian_batch=self._jacobian_batch,
            hessian=self._hessian,
            hessian_batch=self._hessian_batch,
            hvp=self._hvp,
            hvp_batch=self._hvp_batch,
            whvp=self._whvp,
            whvp_batch=self._whvp_batch,
        )

    def derivatives(self) -> Derivatives[Array]:
        """Return the derivative bundle."""
        return self._derivs

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables (dimensions)."""
        return len(self._bases_1d)

    def nsamples(self) -> int:
        """Return the total number of interpolation points."""
        return int(self._nsamples)

    def nqoi(self) -> int:
        """Return the number of quantities of interest, or 0 if not set."""
        if self._values is None:
            return 0
        return int(self._values.shape[0])

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
        # nqoi is now known; rebuild the capability bundle to match
        self._derivs = self._build_derivatives()

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

        basis_vals_1d = self._basis_vals_1d(samples)
        return self._tp_eval_impl(
            basis_vals_1d,
            self._values,
            self._nterms_1d,
            self._bkd,
        )

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
        if self._jac_bases_1d is None:
            raise RuntimeError("Jacobian not supported by univariate bases")

        derivs = []
        for dd in range(self.nvars()):
            jac = self._jac_bases_1d[dd].jacobian_batch(samples[dd : dd + 1, :])
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
        if self._hess_bases_1d is None:
            raise RuntimeError("Hessian not supported by univariate bases")

        derivs = []
        for dd in range(self.nvars()):
            hess = self._hess_bases_1d[dd].hessian_batch(samples[dd : dd + 1, :])
            derivs.append(hess)
        return derivs

    def _jacobian(self, sample: Array) -> Array:
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

    def _second_derivative_terms(self, samples: Array) -> List[List[Array]]:
        """Tensor product second-derivative weights for every (i, j) pair.

        Entry [i][j] has shape (npoints, nterms) and holds the tensor
        product basis differentiated twice in dimension i when i == j,
        or once each in i and j otherwise. Contracting it with a value
        vector gives that Hessian entry at every point.
        """
        nvars = self.nvars()
        basis_vals_1d = self._basis_vals_1d(samples)
        basis_derivs_1d = self._basis_jacobians_1d(samples)
        basis_hess_1d = self._basis_hessians_1d(samples)

        terms: List[List[Array]] = [
            [self._bkd.zeros((1, 1))] * nvars for _ in range(nvars)
        ]
        for dim1 in range(nvars):
            for dim2 in range(dim1, nvars):
                if dim1 == dim2:
                    term = basis_hess_1d[dim1][:, self._tp_indices[dim1, :]]
                else:
                    term = (
                        basis_derivs_1d[dim1][:, self._tp_indices[dim1, :]]
                        * basis_derivs_1d[dim2][:, self._tp_indices[dim2, :]]
                    )
                for dd in range(nvars):
                    if dd != dim1 and dd != dim2:
                        term = (
                            term
                            * basis_vals_1d[dd][:, self._tp_indices[dd, :]]
                        )
                terms[dim1][dim2] = term
                terms[dim2][dim1] = term
        return terms

    def _hessian_from_terms(
        self, terms: List[List[Array]], values: Array, npoints: int
    ) -> Array:
        """Contract second-derivative terms with one value vector.

        Parameters
        ----------
        terms : List[List[Array]]
            From ``_second_derivative_terms``.
        values : Array
            Values for a single quantity of interest, shape (nterms,).
        npoints : int
            Number of evaluation points.

        Returns
        -------
        Array
            Hessians with shape (npoints, nvars, nvars).
        """
        nvars = self.nvars()
        hessian = self._bkd.zeros((npoints, nvars, nvars))
        for dim1 in range(nvars):
            for dim2 in range(dim1, nvars):
                entries = terms[dim1][dim2] @ values
                hessian[:, dim1, dim2] = entries
                if dim1 != dim2:
                    hessian[:, dim2, dim1] = entries
        return hessian

    def _jacobian_batch(self, samples: Array) -> Array:
        """Compute Jacobians at many sample points at once.

        The one-dimensional bases already evaluate a whole batch, so
        this is the same product rule as ``_jacobian`` with the points
        axis carried through rather than indexed away.

        Parameters
        ----------
        samples : Array
            Evaluation points with shape (nvars, npoints).

        Returns
        -------
        Array
            Jacobians with shape (npoints, nqoi, nvars).

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
        npoints = samples.shape[1]

        basis_vals_1d = self._basis_vals_1d(samples)
        basis_derivs_1d = self._basis_jacobians_1d(samples)

        jacobian = self._bkd.zeros((npoints, nqoi, nvars))
        for dim in range(nvars):
            term = basis_derivs_1d[dim][:, self._tp_indices[dim, :]]
            for dd in range(nvars):
                if dd != dim:
                    term = term * basis_vals_1d[dd][:, self._tp_indices[dd, :]]
            # (npoints, nterms) @ (nterms, nqoi) -> (npoints, nqoi)
            jacobian[:, :, dim] = term @ self._bkd.transpose(self._values)
        return jacobian

    def _hessian_batch(self, samples: Array) -> Array:
        """Compute Hessians at many sample points at once (nqoi=1).

        Parameters
        ----------
        samples : Array
            Evaluation points with shape (nvars, npoints).

        Returns
        -------
        Array
            Hessians with shape (npoints, nvars, nvars).

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
                "hessian_batch requires nqoi == 1; use whvp_batch otherwise"
            )
        if not self._hessian_supported:
            raise RuntimeError("Hessian not supported by univariate bases")

        terms = self._second_derivative_terms(samples)
        return self._hessian_from_terms(
            terms, self._values[0, :], samples.shape[1]
        )

    def _hvp_batch(self, samples: Array, vecs: Array) -> Array:
        """Hessian-vector products at many points (nqoi=1).

        Parameters
        ----------
        samples : Array
            Evaluation points with shape (nvars, npoints).
        vecs : Array
            Directions with shape (nvars, npoints), one per point.

        Returns
        -------
        Array
            Products with shape (npoints, nvars).
        """
        hessians = self._hessian_batch(samples)
        return self._bkd.einsum("pij,jp->pi", hessians, vecs)

    def _whvp_batch(
        self, samples: Array, vecs: Array, weights: Array
    ) -> Array:
        """Weighted Hessian-vector products at many points.

        sum_q w_q H_q v is linear in the values, so weighting the
        values by w first collapses the quantities of interest into a
        single vector and the rest is the nqoi == 1 computation.

        Parameters
        ----------
        samples : Array
            Evaluation points with shape (nvars, npoints).
        vecs : Array
            Directions with shape (nvars, npoints), one per point.
        weights : Array
            QoI weights, shape (nqoi, 1), (1, nqoi) or (nqoi,).

        Returns
        -------
        Array
            Products with shape (npoints, nvars).

        Raises
        ------
        ValueError
            If values have not been set.
        RuntimeError
            If the bases do not support second derivatives.
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")
        if not self._hessian_supported:
            raise RuntimeError("WHVP not supported by univariate bases")

        weighted_values = self._bkd.flatten(weights) @ self._values
        terms = self._second_derivative_terms(samples)
        hessians = self._hessian_from_terms(
            terms, weighted_values, samples.shape[1]
        )
        return self._bkd.einsum("pij,jp->pi", hessians, vecs)

    def _hessian(self, sample: Array) -> Array:
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
                            interp_deriv * basis_vals_1d[dd][0, self._tp_indices[dd, :]]
                        )

                val = self._bkd.dot(interp_deriv, values_q)
                hessian[dim1, dim2] = val
                if dim1 != dim2:
                    hessian[dim2, dim1] = val

        return hessian

    def _hvp(self, sample: Array, vec: Array) -> Array:
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
        vec_flat = self._bkd.flatten(vec)

        basis_vals_1d = self._basis_vals_1d(sample)
        basis_derivs_1d = self._basis_jacobians_1d(sample)
        basis_hess_1d = self._basis_hessians_1d(sample)

        values_q = self._values[0, :]  # Shape: (nsamples,), nqoi must be 1
        result = self._bkd.zeros((nvars, 1))

        for dim1 in range(nvars):
            row_sum: Array = self._bkd.asarray(0.0)

            for dim2 in range(nvars):
                v_j = vec_flat[dim2]

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
                            interp_deriv * basis_vals_1d[dd][0, self._tp_indices[dd, :]]
                        )

                H_ij = self._bkd.dot(interp_deriv, values_q)
                row_sum = row_sum + H_ij * v_j

            result[dim1, 0] = row_sum

        return result

    def _hvp_for_qoi(
        self,
        sample: Array,
        vec: Array,
        qoi_idx: int,
        basis_vals_1d: List[Array],
        basis_derivs_1d: List[Array],
        basis_hess_1d: List[Array],
    ) -> Array:
        """Compute HVP for a specific QoI index (internal helper).

        This is used by whvp to compute weighted HVP across QoIs.
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        nvars = self.nvars()
        vec_flat = self._bkd.flatten(vec)
        values_q = self._values[qoi_idx, :]  # Shape: (nsamples,)
        result = self._bkd.zeros((nvars, 1))

        for dim1 in range(nvars):
            row_sum: Array = self._bkd.asarray(0.0)

            for dim2 in range(nvars):
                v_j = vec_flat[dim2]

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
                            interp_deriv * basis_vals_1d[dd][0, self._tp_indices[dd, :]]
                        )

                H_ij = self._bkd.dot(interp_deriv, values_q)
                row_sum = row_sum + H_ij * v_j

            result[dim1, 0] = row_sum

        return result

    def _whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
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

        weights_flat = self._bkd.flatten(weights)

        # Precompute basis evaluations once for all QoIs
        basis_vals_1d = self._basis_vals_1d(sample)
        basis_derivs_1d = self._basis_jacobians_1d(sample)
        basis_hess_1d = self._basis_hessians_1d(sample)

        for qoi_idx in range(nqoi):
            w = weights_flat[qoi_idx]
            hvp_q = self._hvp_for_qoi(
                sample, vec, qoi_idx, basis_vals_1d, basis_derivs_1d, basis_hess_1d
            )
            result = result + w * hvp_q

        return result

    def __repr__(self) -> str:
        nterms_str = ",".join(str(n) for n in self._nterms_1d)
        return (
            f"TensorProductInterpolant(nvars={self.nvars()}, "
            f"nterms_1d=[{nterms_str}], nsamples={self._nsamples})"
        )
