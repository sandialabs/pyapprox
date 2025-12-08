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

    def __repr__(self) -> str:
        index_str = ",".join(str(int(i)) for i in self._index)
        return (
            f"TensorProductSubspace(index=[{index_str}], "
            f"nsamples={self._nsamples})"
        )
