"""Tensor product subspace for sparse grids.

A subspace represents a single tensor product of univariate interpolations,
identified by a multi-index specifying the level in each dimension.
"""

from typing import Generic, List, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    Basis1DProtocol,
    IndexGrowthRuleProtocol,
)


def _lagrange_derivatives_1d(
    nodes: Array,
    pt_idx: int,
    x: Array,
    bkd: "Backend[Array]",
) -> Tuple[Array, Array, Array]:
    """Compute 1D Lagrange basis function and its derivatives at points.

    Parameters
    ----------
    nodes : Array
        Interpolation nodes, shape (npts,)
    pt_idx : int
        Index of the interpolation point
    x : Array
        Evaluation points, shape (npoints,)
    bkd : Backend[Array]
        Computational backend

    Returns
    -------
    L : Array
        Lagrange basis values, shape (npoints,)
    dL : Array
        First derivatives, shape (npoints,)
    d2L : Array
        Second derivatives, shape (npoints,)
    """
    npoints = x.shape[0]
    npts = nodes.shape[0]
    x_i = float(nodes[pt_idx])

    # Compute Lagrange basis L_i(x) = prod_{j!=i} (x - x_j) / (x_i - x_j)
    L = bkd.ones((npoints,))
    for j in range(npts):
        if j != pt_idx:
            x_j = float(nodes[j])
            L = L * (x - x_j) / (x_i - x_j)

    # Compute first derivative using product rule:
    # dL/dx = sum_k [ prod_{j!=i,j!=k} (x-x_j)/(x_i-x_j) * 1/(x_i-x_k) ]
    dL = bkd.zeros((npoints,))
    for k in range(npts):
        if k != pt_idx:
            term = bkd.ones((npoints,))
            for j in range(npts):
                if j != pt_idx and j != k:
                    x_j = float(nodes[j])
                    term = term * (x - x_j) / (x_i - x_j)
            x_k = float(nodes[k])
            term = term / (x_i - x_k)
            dL = dL + term

    # Compute second derivative using product rule on first derivative:
    # d2L/dx2 = sum_k sum_m [ prod_{j!=i,j!=k,j!=m} ... ]
    d2L = bkd.zeros((npoints,))
    for k in range(npts):
        if k != pt_idx:
            for m in range(npts):
                if m != pt_idx and m != k:
                    term = bkd.ones((npoints,))
                    for j in range(npts):
                        if j != pt_idx and j != k and j != m:
                            x_j = float(nodes[j])
                            term = term * (x - x_j) / (x_i - x_j)
                    x_k = float(nodes[k])
                    x_m = float(nodes[m])
                    term = term / ((x_i - x_k) * (x_i - x_m))
                    d2L = d2L + term

    return L, dL, d2L


class TensorProductSubspace(Generic[Array]):
    """Single tensor product subspace in sparse grid.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    index : Array
        Multi-index identifying this subspace, shape (nvars,)
    univariate_bases : List[Basis1DProtocol[Array]]
        Univariate bases for each dimension.
    growth_rule : IndexGrowthRuleProtocol
        Rule mapping level to number of points.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
    >>> from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule
    >>> bkd = NumpyBkd()
    >>> basis = LegendrePolynomial1D(bkd)
    >>> growth = LinearGrowthRule()
    >>> index = bkd.asarray([1, 2])
    >>> subspace = TensorProductSubspace(bkd, index, [basis, basis], growth)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        index: Array,
        univariate_bases: List[Basis1DProtocol[Array]],
        growth_rule: IndexGrowthRuleProtocol,
    ):
        self._bkd = bkd
        self._index = bkd.copy(index)
        self._univariate_bases = univariate_bases
        self._growth_rule = growth_rule
        self._values: Optional[Array] = None

        # Compute 1D samples and number of points per dimension
        self._1d_samples: List[Array] = []
        self._npts_1d: List[int] = []
        for dim in range(self.nvars()):
            level = int(index[dim])
            npts = growth_rule(level)
            self._npts_1d.append(npts)

            # Get quadrature points for this level
            basis = univariate_bases[dim]
            basis.set_nterms(npts)
            samples_1d, _ = basis.gauss_quadrature_rule(npts)
            self._1d_samples.append(samples_1d)

        # Build tensor product samples
        self._samples = self._build_tensor_product_samples()
        self._nsamples = self._samples.shape[1]

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def get_index(self) -> Array:
        """Return the multi-index identifying this subspace."""
        return self._bkd.copy(self._index)

    def nvars(self) -> int:
        """Return the number of variables."""
        return len(self._univariate_bases)

    def nsamples(self) -> int:
        """Return the number of samples in this subspace."""
        return self._nsamples

    def get_samples(self) -> Array:
        """Return sample locations for this subspace."""
        return self._bkd.copy(self._samples)

    def get_values(self) -> Optional[Array]:
        """Return function values at samples, if set."""
        return self._values

    def set_values(self, values: Array) -> None:
        """Set function values at samples."""
        if values.shape[0] != self._nsamples:
            raise ValueError(
                f"Expected {self._nsamples} values, got {values.shape[0]}"
            )
        self._values = self._bkd.copy(values)

    def _build_tensor_product_samples(self) -> Array:
        """Build tensor product of 1D sample locations."""
        nvars = self.nvars()

        # Compute total number of samples
        total = 1
        for npts in self._npts_1d:
            total *= npts

        # Build tensor product grid
        samples = self._bkd.zeros((nvars, total))

        # Use meshgrid-like approach
        repeat_inner = 1
        for dim in range(nvars - 1, -1, -1):
            npts = self._npts_1d[dim]
            samples_1d = self._1d_samples[dim].flatten()

            repeat_outer = total // (npts * repeat_inner)

            idx = 0
            for _ in range(repeat_outer):
                for pt_idx in range(npts):
                    for _ in range(repeat_inner):
                        samples[dim, idx] = samples_1d[pt_idx]
                        idx += 1

            repeat_inner *= npts

        return samples

    def __call__(self, samples: Array) -> Array:
        """Evaluate subspace interpolant at given samples.

        Parameters
        ----------
        samples : Array
            Evaluation points of shape (nvars, npoints)

        Returns
        -------
        Array
            Interpolant values of shape (npoints, nqoi)
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        npoints = samples.shape[1]
        nqoi = self._values.shape[1]

        # Evaluate 1D bases at each sample point
        basis_vals_1d: List[Array] = []
        for dim in range(self.nvars()):
            npts = self._npts_1d[dim]
            self._univariate_bases[dim].set_nterms(npts)
            vals = self._univariate_bases[dim](samples[dim:dim+1, :])
            basis_vals_1d.append(vals)  # Shape: (npoints, npts)

        # Compute Lagrange interpolation coefficients
        # For each sample point, compute the tensor product of 1D Lagrange values
        result = self._bkd.zeros((npoints, nqoi))

        # Build interpolation matrix
        # Each row corresponds to a sample point
        # Each column corresponds to a tensor product basis function
        interp_mat = self._bkd.ones((npoints, self._nsamples))

        idx = 0
        repeat_inner = 1
        for dim in range(self.nvars() - 1, -1, -1):
            npts = self._npts_1d[dim]
            repeat_outer = self._nsamples // (npts * repeat_inner)

            col = 0
            for _ in range(repeat_outer):
                for pt_idx in range(npts):
                    for _ in range(repeat_inner):
                        # Lagrange basis for this point
                        interp_mat[:, col] *= self._lagrange_basis_1d(
                            dim, pt_idx, samples[dim:dim+1, :]
                        )
                        col += 1

            repeat_inner *= npts

        # Apply interpolation
        result = interp_mat @ self._values

        return result

    def _lagrange_basis_1d(
        self,
        dim: int,
        pt_idx: int,
        samples: Array,
    ) -> Array:
        """Compute 1D Lagrange basis function at given samples.

        Parameters
        ----------
        dim : int
            Dimension index
        pt_idx : int
            Index of the interpolation point
        samples : Array
            Evaluation points, shape (1, npoints)

        Returns
        -------
        Array
            Lagrange basis values, shape (npoints,)
        """
        nodes = self._1d_samples[dim].flatten()
        npoints = samples.shape[1]

        x_i = float(nodes[pt_idx])
        result = self._bkd.ones((npoints,))

        for j, x_j in enumerate(nodes):
            if j != pt_idx:
                x_j = float(x_j)
                result = result * (samples[0, :] - x_j) / (x_i - x_j)

        return result

    def jacobian_supported(self) -> bool:
        """Return whether Jacobian computation is supported."""
        return True

    def hessian_supported(self) -> bool:
        """Return whether Hessian computation is supported."""
        return True

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample point.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)

        Returns
        -------
        Array
            Jacobian matrix of shape (nqoi, nvars)
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        nvars = self.nvars()
        nqoi = self._values.shape[1]

        # Compute 1D Lagrange basis values and derivatives at the sample point
        L_1d: List[Array] = []
        dL_1d: List[Array] = []

        for dim in range(nvars):
            nodes = self._1d_samples[dim].flatten()
            x = sample[dim, :]
            L, dL, _ = _lagrange_derivatives_1d(nodes, 0, x, self._bkd)
            # We need all basis functions, not just one
            L_all = self._bkd.zeros((self._npts_1d[dim],))
            dL_all = self._bkd.zeros((self._npts_1d[dim],))
            for pt_idx in range(self._npts_1d[dim]):
                L, dL, _ = _lagrange_derivatives_1d(nodes, pt_idx, x, self._bkd)
                L_all[pt_idx] = L[0]
                dL_all[pt_idx] = dL[0]
            L_1d.append(L_all)
            dL_1d.append(dL_all)

        # Compute Jacobian using product rule on tensor product
        # f(x) = sum_i c_i * prod_d L_{i_d}(x_d)
        # df/dx_d = sum_i c_i * dL_{i_d}/dx_d * prod_{k!=d} L_{i_k}(x_k)
        jacobian = self._bkd.zeros((nqoi, nvars))

        for dim in range(nvars):
            # Build tensor product: dL in dim, L in other dims
            interp_deriv = self._bkd.ones((self._nsamples,))

            repeat_inner = 1
            for d in range(nvars - 1, -1, -1):
                npts = self._npts_1d[d]
                repeat_outer = self._nsamples // (npts * repeat_inner)

                col = 0
                for _ in range(repeat_outer):
                    for pt_idx in range(npts):
                        for _ in range(repeat_inner):
                            if d == dim:
                                interp_deriv[col] *= dL_1d[d][pt_idx]
                            else:
                                interp_deriv[col] *= L_1d[d][pt_idx]
                            col += 1
                repeat_inner *= npts

            # Apply to values
            for q in range(nqoi):
                jacobian[q, dim] = self._bkd.dot(
                    interp_deriv, self._values[:, q]
                )

        return jacobian

    def hessian(self, sample: Array, qoi_idx: int = 0) -> Array:
        """Compute Hessian at a single sample point for one QoI.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)
        qoi_idx : int
            Index of quantity of interest. Default: 0.

        Returns
        -------
        Array
            Hessian matrix of shape (nvars, nvars)
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        nvars = self.nvars()

        # Compute 1D Lagrange basis values and derivatives at the sample point
        L_1d: List[Array] = []
        dL_1d: List[Array] = []
        d2L_1d: List[Array] = []

        for dim in range(nvars):
            nodes = self._1d_samples[dim].flatten()
            x = sample[dim, :]
            L_all = self._bkd.zeros((self._npts_1d[dim],))
            dL_all = self._bkd.zeros((self._npts_1d[dim],))
            d2L_all = self._bkd.zeros((self._npts_1d[dim],))
            for pt_idx in range(self._npts_1d[dim]):
                L, dL, d2L = _lagrange_derivatives_1d(
                    nodes, pt_idx, x, self._bkd
                )
                L_all[pt_idx] = L[0]
                dL_all[pt_idx] = dL[0]
                d2L_all[pt_idx] = d2L[0]
            L_1d.append(L_all)
            dL_1d.append(dL_all)
            d2L_1d.append(d2L_all)

        # Compute Hessian using product rule
        # d2f/dx_d1 dx_d2 = sum_i c_i * ...
        hessian = self._bkd.zeros((nvars, nvars))
        values_q = self._values[:, qoi_idx]

        for dim1 in range(nvars):
            for dim2 in range(dim1, nvars):
                # Build tensor product for this Hessian entry
                interp_deriv = self._bkd.ones((self._nsamples,))

                repeat_inner = 1
                for d in range(nvars - 1, -1, -1):
                    npts = self._npts_1d[d]
                    repeat_outer = self._nsamples // (npts * repeat_inner)

                    col = 0
                    for _ in range(repeat_outer):
                        for pt_idx in range(npts):
                            for _ in range(repeat_inner):
                                if dim1 == dim2 and d == dim1:
                                    # Second derivative in same direction
                                    interp_deriv[col] *= d2L_1d[d][pt_idx]
                                elif d == dim1 or d == dim2:
                                    # First derivative
                                    interp_deriv[col] *= dL_1d[d][pt_idx]
                                else:
                                    # No derivative
                                    interp_deriv[col] *= L_1d[d][pt_idx]
                                col += 1
                    repeat_inner *= npts

                val = self._bkd.dot(interp_deriv, values_q)
                hessian[dim1, dim2] = val
                if dim1 != dim2:
                    hessian[dim2, dim1] = val

        return hessian

    def hvp(self, sample: Array, vec: Array, qoi_idx: int = 0) -> Array:
        """Compute Hessian-vector product efficiently without forming Hessian.

        Computes H @ v where H is the Hessian, without explicitly forming H.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)
        vec : Array
            Direction vector of shape (nvars, 1)
        qoi_idx : int
            Index of quantity of interest. Default: 0.

        Returns
        -------
        Array
            Hessian-vector product of shape (nvars, 1)
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        nvars = self.nvars()
        vec_flat = vec.flatten()

        # Compute 1D Lagrange basis values and derivatives at the sample point
        L_1d: List[Array] = []
        dL_1d: List[Array] = []
        d2L_1d: List[Array] = []

        for dim in range(nvars):
            nodes = self._1d_samples[dim].flatten()
            x = sample[dim, :]
            L_all = self._bkd.zeros((self._npts_1d[dim],))
            dL_all = self._bkd.zeros((self._npts_1d[dim],))
            d2L_all = self._bkd.zeros((self._npts_1d[dim],))
            for pt_idx in range(self._npts_1d[dim]):
                L, dL, d2L = _lagrange_derivatives_1d(
                    nodes, pt_idx, x, self._bkd
                )
                L_all[pt_idx] = L[0]
                dL_all[pt_idx] = dL[0]
                d2L_all[pt_idx] = d2L[0]
            L_1d.append(L_all)
            dL_1d.append(dL_all)
            d2L_1d.append(d2L_all)

        values_q = self._values[:, qoi_idx]
        result = self._bkd.zeros((nvars, 1))

        # Compute H @ v efficiently by computing each row dot v
        # (H @ v)_i = sum_j H_ij * v_j
        # For tensor product interpolation, we can compute this more efficiently
        # by grouping terms that share the same tensor product structure
        for dim1 in range(nvars):
            row_sum = 0.0

            for dim2 in range(nvars):
                v_j = float(vec_flat[dim2])
                if abs(v_j) < 1e-14:
                    continue

                # Build tensor product for H[dim1, dim2]
                interp_deriv = self._bkd.ones((self._nsamples,))

                repeat_inner = 1
                for d in range(nvars - 1, -1, -1):
                    npts = self._npts_1d[d]
                    repeat_outer = self._nsamples // (npts * repeat_inner)

                    col = 0
                    for _ in range(repeat_outer):
                        for pt_idx in range(npts):
                            for _ in range(repeat_inner):
                                if dim1 == dim2 and d == dim1:
                                    # Second derivative in same direction
                                    interp_deriv[col] *= d2L_1d[d][pt_idx]
                                elif d == dim1 or d == dim2:
                                    # First derivative
                                    interp_deriv[col] *= dL_1d[d][pt_idx]
                                else:
                                    # No derivative
                                    interp_deriv[col] *= L_1d[d][pt_idx]
                                col += 1
                    repeat_inner *= npts

                H_ij = float(self._bkd.dot(interp_deriv, values_q))
                row_sum += H_ij * v_j

            result[dim1, 0] = row_sum

        return result

    def whvp(
        self,
        sample: Array,
        vec: Array,
        weights: Array,
    ) -> Array:
        """Compute weighted Hessian-vector product efficiently.

        Computes sum_q weights[q] * H_q @ v where H_q is the Hessian for QoI q.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)
        vec : Array
            Direction vector of shape (nvars, 1)
        weights : Array
            Weights for each QoI. Can be shape (nqoi, 1), (1, nqoi), or (nqoi,).

        Returns
        -------
        Array
            Weighted Hessian-vector product of shape (nvars, 1)
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        nqoi = self._values.shape[1]
        result = self._bkd.zeros((self.nvars(), 1))

        # Handle different weight shapes
        weights_flat = weights.flatten()

        for qoi_idx in range(nqoi):
            w = float(weights_flat[qoi_idx])
            if abs(w) > 1e-14:
                result = result + w * self.hvp(sample, vec, qoi_idx=qoi_idx)

        return result

    def __repr__(self) -> str:
        index_str = ",".join(str(int(i)) for i in self._index)
        return (
            f"TensorProductSubspace(index=[{index_str}], "
            f"nsamples={self._nsamples})"
        )
